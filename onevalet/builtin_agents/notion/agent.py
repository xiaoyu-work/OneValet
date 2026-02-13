"""
NotionDomainAgent - Domain agent for all Notion operations.

Replaces NotionCreatePageAgent + NotionUpdatePageAgent + 3 standalone tools
(notion_search, notion_read_page, notion_query_database) with a single agent
that has its own mini ReAct loop.
"""

import os
import logging
from typing import Any, Dict, List

from onevalet import valet
from onevalet.agents.domain_agent import DomainAgent, DomainTool, DomainToolContext

from .client import NotionClient

logger = logging.getLogger(__name__)


# =============================================================================
# Helper functions (reused from tools/notion.py)
# =============================================================================

def _blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    """Convert Notion blocks to readable text."""
    parts = []
    for block in blocks:
        block_type = block.get("type", "")
        type_data = block.get(block_type, {})

        if block_type == "to_do":
            checked = type_data.get("checked", False)
            rich_text = type_data.get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich_text)
            marker = "[x]" if checked else "[ ]"
            parts.append(f"{marker} {text}")
        elif block_type in ("heading_1", "heading_2", "heading_3"):
            level = {"heading_1": "#", "heading_2": "##", "heading_3": "###"}[block_type]
            rich_text = type_data.get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich_text)
            if text:
                parts.append(f"{level} {text}")
        elif block_type == "bulleted_list_item":
            rich_text = type_data.get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich_text)
            if text:
                parts.append(f"- {text}")
        elif block_type == "numbered_list_item":
            rich_text = type_data.get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich_text)
            if text:
                parts.append(f"1. {text}")
        elif block_type == "divider":
            parts.append("---")
        elif block_type == "code":
            rich_text = type_data.get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich_text)
            lang = type_data.get("language", "")
            if text:
                parts.append(f"```{lang}\n{text}\n```")
        else:
            rich_text = type_data.get("rich_text", [])
            if rich_text:
                text = "".join(rt.get("plain_text", "") for rt in rich_text)
                if text:
                    parts.append(text)
    return "\n".join(parts)


def _get_page_title(page: Dict[str, Any]) -> str:
    """Extract title from a page object."""
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            title_parts = prop.get("title", [])
            return "".join(t.get("plain_text", "") for t in title_parts)
    return "Untitled"


def _extract_property_value(prop: Dict[str, Any]) -> str:
    """Extract display value from a Notion property."""
    prop_type = prop.get("type", "")

    if prop_type == "title":
        return "".join(t.get("plain_text", "") for t in prop.get("title", []))
    elif prop_type == "rich_text":
        return "".join(t.get("plain_text", "") for t in prop.get("rich_text", []))
    elif prop_type == "number":
        val = prop.get("number")
        return str(val) if val is not None else ""
    elif prop_type == "select":
        sel = prop.get("select")
        return sel.get("name", "") if sel else ""
    elif prop_type == "multi_select":
        return ", ".join(s.get("name", "") for s in prop.get("multi_select", []))
    elif prop_type == "date":
        date = prop.get("date")
        if date:
            start = date.get("start", "")
            end = date.get("end", "")
            return f"{start} → {end}" if end else start
        return ""
    elif prop_type == "checkbox":
        return "Yes" if prop.get("checkbox") else "No"
    elif prop_type == "url":
        return prop.get("url", "")
    elif prop_type == "email":
        return prop.get("email", "")
    elif prop_type == "phone_number":
        return prop.get("phone_number", "")
    elif prop_type == "status":
        status = prop.get("status")
        return status.get("name", "") if status else ""
    elif prop_type in ("created_time", "last_edited_time"):
        return prop.get(prop_type, "")[:10]
    elif prop_type == "people":
        return ", ".join(p.get("name", p.get("id", "")) for p in prop.get("people", []))
    return ""


# =============================================================================
# Tool executors
# =============================================================================

async def notion_search(args: dict, context: DomainToolContext) -> str:
    """Search Notion pages and databases by keyword."""
    query = args.get("query", "")
    filter_type = args.get("filter_type")
    page_size = args.get("page_size", 10)

    if not os.getenv("NOTION_API_KEY"):
        return "Error: Notion API key not configured. Please add it in Settings."

    try:
        client = NotionClient()
        data = await client.search(query=query, filter_type=filter_type, page_size=page_size)
        results = data.get("results", [])

        if not results:
            if query:
                return f'No results found in Notion for "{query}".'
            return "Your Notion workspace appears empty."

        items = []
        for i, result in enumerate(results, 1):
            obj_type = result.get("object", "page")
            page_id = result.get("id", "")
            title = _get_page_title(result) if obj_type == "page" else "".join(
                t.get("plain_text", "") for t in result.get("title", [])
            )
            last_edited = result.get("last_edited_time", "")[:10]
            items.append(f'{i}. [{obj_type}] "{title or "Untitled"}" (id: {page_id}, edited: {last_edited})')

        header = f'Found {len(results)} results for "{query}":' if query else f"Notion pages ({len(results)}):"
        return header + "\n" + "\n".join(items)
    except Exception as e:
        logger.error(f"Notion search failed: {e}", exc_info=True)
        return f"Error searching Notion: {e}"


async def notion_read_page(args: dict, context: DomainToolContext) -> str:
    """Read the full content of a Notion page."""
    page_id = args.get("page_id", "")
    if not page_id:
        return "Error: page_id is required."
    if not os.getenv("NOTION_API_KEY"):
        return "Error: Notion API key not configured. Please add it in Settings."

    try:
        client = NotionClient()
        page = await client.get_page(page_id)
        title = _get_page_title(page)
        blocks = await client.get_page_content(page_id)
        content = _blocks_to_text(blocks)

        if not content.strip():
            return f'Page "{title}" exists but has no content.'
        return f'Page: "{title}"\n\nContent:\n{content}'
    except Exception as e:
        logger.error(f"Notion read page failed: {e}", exc_info=True)
        return f"Error reading Notion page: {e}"


async def notion_query_database(args: dict, context: DomainToolContext) -> str:
    """Query a Notion database and return rows."""
    database_id = args.get("database_id", "")
    filter_obj = args.get("filter")
    sorts = args.get("sorts")
    page_size = args.get("page_size", 20)

    if not database_id:
        return "Error: database_id is required."
    if not os.getenv("NOTION_API_KEY"):
        return "Error: Notion API key not configured. Please add it in Settings."

    try:
        client = NotionClient()
        data = await client.query_database(
            database_id=database_id, filter=filter_obj, sorts=sorts, page_size=page_size,
        )
        results = data.get("results", [])
        if not results:
            return "No rows found in this database."

        rows = []
        for i, page in enumerate(results, 1):
            props = page.get("properties", {})
            fields = []
            for prop_name, prop_val in props.items():
                display = _extract_property_value(prop_val)
                if display:
                    fields.append(f"  {prop_name}: {display}")
            page_id = page.get("id", "")
            rows.append(f"{i}. (id: {page_id})\n" + "\n".join(fields))

        return f"Database query returned {len(results)} rows:\n\n" + "\n\n".join(rows)
    except Exception as e:
        logger.error(f"Notion database query failed: {e}", exc_info=True)
        return f"Error querying Notion database: {e}"


async def notion_create_page(args: dict, context: DomainToolContext) -> str:
    """Create a new Notion page."""
    title = args.get("title", "")
    content = args.get("content", "")
    parent_name = args.get("parent", "")

    if not title:
        return "Error: title is required."
    if not os.getenv("NOTION_API_KEY"):
        return "Error: Notion API key not configured. Please add it in Settings."

    try:
        client = NotionClient()
        parent_id = None

        # Resolve parent page if specified
        if parent_name:
            data = await client.search(query=parent_name, filter_type="page", page_size=1)
            results = data.get("results", [])
            if results:
                parent_id = results[0]["id"]

        if not parent_id:
            # Use first available page as parent
            data = await client.search(filter_type="page", page_size=1)
            results = data.get("results", [])
            if not results:
                return "No pages found in Notion workspace to use as parent."
            parent_id = results[0]["id"]

        page = await client.create_page(
            parent_id=parent_id, title=title, content=content, parent_type="page_id",
        )
        url = page.get("url", "")
        return f'Created Notion page "{title}".\n{url}'
    except Exception as e:
        logger.error(f"Failed to create Notion page: {e}", exc_info=True)
        return "Couldn't create the Notion page. Please check your API key and permissions."


async def notion_update_page(args: dict, context: DomainToolContext) -> str:
    """Update an existing Notion page by appending content."""
    page_title = args.get("page_title", "")
    content = args.get("content", "")

    if not page_title:
        return "Error: page_title is required."
    if not content:
        return "Error: content is required."
    if not os.getenv("NOTION_API_KEY"):
        return "Error: Notion API key not configured. Please add it in Settings."

    try:
        client = NotionClient()
        data = await client.search(query=page_title, filter_type="page", page_size=3)
        results = data.get("results", [])
        if not results:
            return f'Couldn\'t find a Notion page matching "{page_title}".'

        page_id = results[0]["id"]
        resolved_title = _get_page_title(results[0])

        blocks = NotionClient.text_to_blocks(content)
        await client.append_blocks(page_id, blocks)
        return f'Updated "{resolved_title}" with new content.'
    except Exception as e:
        logger.error(f"Failed to update Notion page: {e}", exc_info=True)
        return "Couldn't update the Notion page. Please check your API key and permissions."


# =============================================================================
# Approval preview functions
# =============================================================================

async def _create_page_preview(args: dict, context) -> str:
    title = args.get("title", "")
    content = args.get("content", "")
    parent = args.get("parent", "workspace")
    preview = content[:100] + "..." if len(content) > 100 else content
    return f"Create Notion page?\n\nTitle: {title}\nParent: {parent}\nContent: {preview}"


async def _update_page_preview(args: dict, context) -> str:
    page_title = args.get("page_title", "")
    content = args.get("content", "")
    preview = content[:100] + "..." if len(content) > 100 else content
    return f'Update Notion page "{page_title}"?\n\nAdd content: {preview}'


# =============================================================================
# Domain Agent
# =============================================================================

@valet(capabilities=["notion"])
class NotionDomainAgent(DomainAgent):
    """Search, read, create, and update Notion pages and databases. Use when the user mentions Notion, their notes, wiki, or knowledge base in Notion."""

    max_domain_turns = 5

    domain_system_prompt = """\
You are a Notion workspace assistant with access to Notion tools.

Available tools:
- notion_search: Search pages and databases by keyword.
- notion_read_page: Read full content of a page by ID.
- notion_query_database: Query a database to get rows with properties.
- notion_create_page: Create a new page with title and content.
- notion_update_page: Update an existing page by appending content.

Instructions:
1. For searches (find pages, list workspace), call notion_search with short keywords (1-2 words).
2. To read a page's content, first search for its ID, then call notion_read_page.
3. To query a database, first search for the database ID, then call notion_query_database.
4. To create a new page, call notion_create_page with title and content.
5. To update/edit a page, call notion_update_page with the page title and new content.
6. If the user's request is ambiguous, ask for clarification WITHOUT calling any tools.
7. After getting tool results, provide a clear summary to the user."""

    domain_tools = [
        DomainTool(
            name="notion_search",
            description="Search Notion workspace for pages and databases by keyword. Use short keywords (1-2 words).",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Short search keyword (1-2 words)"},
                    "filter_type": {"type": "string", "enum": ["page", "database"], "description": "Filter by type (optional)"},
                    "page_size": {"type": "integer", "description": "Max results (default 10)"},
                },
                "required": [],
            },
            executor=notion_search,
        ),
        DomainTool(
            name="notion_read_page",
            description="Read the full content of a Notion page by its ID. Use notion_search first to find the page ID.",
            parameters={
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "The Notion page ID to read"},
                },
                "required": ["page_id"],
            },
            executor=notion_read_page,
        ),
        DomainTool(
            name="notion_query_database",
            description="Query a Notion database to get rows with properties. Use notion_search with filter_type='database' first.",
            parameters={
                "type": "object",
                "properties": {
                    "database_id": {"type": "string", "description": "The database ID to query"},
                    "filter": {"type": "object", "description": "Notion filter object (optional)"},
                    "sorts": {"type": "array", "items": {"type": "object"}, "description": "Sort objects (optional)"},
                    "page_size": {"type": "integer", "description": "Max rows (default 20)"},
                },
                "required": ["database_id"],
            },
            executor=notion_query_database,
        ),
        DomainTool(
            name="notion_create_page",
            description="Create a new Notion page with title and content.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Page title"},
                    "content": {"type": "string", "description": "Page content in plain text"},
                    "parent": {"type": "string", "description": "Parent page name (optional, defaults to workspace)"},
                },
                "required": ["title"],
            },
            executor=notion_create_page,
            needs_approval=True,
            get_preview=_create_page_preview,
        ),
        DomainTool(
            name="notion_update_page",
            description="Update an existing Notion page by appending content. Searches by title.",
            parameters={
                "type": "object",
                "properties": {
                    "page_title": {"type": "string", "description": "Title of the page to update"},
                    "content": {"type": "string", "description": "Content to append"},
                },
                "required": ["page_title", "content"],
            },
            executor=notion_update_page,
            needs_approval=True,
            get_preview=_update_page_preview,
        ),
    ]

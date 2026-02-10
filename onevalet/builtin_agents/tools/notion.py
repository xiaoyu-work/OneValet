"""
Notion Tools - Search, read pages, and query databases via Notion API.

These tools are used by the orchestrator's ReAct loop so the LLM can
chain multiple Notion operations (search → read → answer) naturally.

Requires NOTION_API_KEY environment variable (Internal Integration Token).
"""

import os
import logging
from typing import Any, Dict, List

from onevalet.tools import ToolRegistry, ToolDefinition, ToolCategory, ToolExecutionContext

logger = logging.getLogger(__name__)


def _get_client():
    """Get a NotionClient instance (lazy import to avoid circular deps)."""
    from onevalet.builtin_agents.notion.client import NotionClient
    return NotionClient()


def _blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    """Convert Notion blocks to readable text, preserving to_do checkbox state."""
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
    elif prop_type == "formula":
        formula = prop.get("formula", {})
        f_type = formula.get("type", "")
        return str(formula.get(f_type, ""))
    elif prop_type == "relation":
        return f"({len(prop.get('relation', []))} relations)"
    elif prop_type == "rollup":
        rollup = prop.get("rollup", {})
        r_type = rollup.get("type", "")
        return str(rollup.get(r_type, ""))
    elif prop_type in ("created_time", "last_edited_time"):
        return prop.get(prop_type, "")[:10]
    elif prop_type in ("created_by", "last_edited_by"):
        person = prop.get(prop_type, {})
        return person.get("name", person.get("id", ""))
    elif prop_type == "people":
        return ", ".join(p.get("name", p.get("id", "")) for p in prop.get("people", []))
    elif prop_type == "files":
        return f"({len(prop.get('files', []))} files)"

    return ""


# ──────────────────────────────────────────────────────────────
# Tool Executors
# ──────────────────────────────────────────────────────────────

async def notion_search_executor(args: dict, context: ToolExecutionContext = None) -> str:
    """Search Notion pages and databases by keyword."""
    query = args.get("query", "")
    filter_type = args.get("filter_type")
    page_size = args.get("page_size", 10)

    api_key = os.getenv("NOTION_API_KEY")
    if not api_key:
        return "Error: Notion API key not configured. Please add it in Settings."

    try:
        client = _get_client()
        data = await client.search(query=query, filter_type=filter_type, page_size=page_size)
        results = data.get("results", [])

        if not results:
            if query:
                return f"No results found in Notion for \"{query}\"."
            return "Your Notion workspace appears empty, or the integration doesn't have access to any pages."

        items = []
        for i, result in enumerate(results, 1):
            obj_type = result.get("object", "page")
            page_id = result.get("id", "")

            if obj_type == "page":
                title = _get_page_title(result)
            elif obj_type == "database":
                title_parts = result.get("title", [])
                title = "".join(t.get("plain_text", "") for t in title_parts)
            else:
                title = "Untitled"

            last_edited = result.get("last_edited_time", "")[:10]
            items.append(f"{i}. [{obj_type}] \"{title or 'Untitled'}\" (id: {page_id}, edited: {last_edited})")

        header = f"Found {len(results)} results for \"{query}\":" if query else f"Notion pages ({len(results)}):"
        return header + "\n" + "\n".join(items)

    except Exception as e:
        logger.error(f"Notion search failed: {e}", exc_info=True)
        return f"Error searching Notion: {e}"


async def notion_read_page_executor(args: dict, context: ToolExecutionContext = None) -> str:
    """Read the full content of a Notion page."""
    page_id = args.get("page_id", "")

    if not page_id:
        return "Error: page_id is required."

    api_key = os.getenv("NOTION_API_KEY")
    if not api_key:
        return "Error: Notion API key not configured. Please add it in Settings."

    try:
        client = _get_client()

        # Get page metadata for title
        page = await client.get_page(page_id)
        title = _get_page_title(page)

        # Get page content blocks
        blocks = await client.get_page_content(page_id)
        content = _blocks_to_text(blocks)

        if not content.strip():
            return f"Page \"{title}\" exists but has no content."

        return f"Page: \"{title}\"\n\nContent:\n{content}"

    except Exception as e:
        logger.error(f"Notion read page failed: {e}", exc_info=True)
        return f"Error reading Notion page: {e}"


async def notion_query_database_executor(args: dict, context: ToolExecutionContext = None) -> str:
    """Query a Notion database and return rows with their properties."""
    database_id = args.get("database_id", "")
    filter_obj = args.get("filter")
    sorts = args.get("sorts")
    page_size = args.get("page_size", 20)

    if not database_id:
        return "Error: database_id is required."

    api_key = os.getenv("NOTION_API_KEY")
    if not api_key:
        return "Error: Notion API key not configured. Please add it in Settings."

    try:
        client = _get_client()
        data = await client.query_database(
            database_id=database_id,
            filter=filter_obj,
            sorts=sorts,
            page_size=page_size,
        )
        results = data.get("results", [])

        if not results:
            return "No rows found in this database (with the given filter)."

        rows = []
        for i, page in enumerate(results, 1):
            props = page.get("properties", {})
            fields = []
            for prop_name, prop_val in props.items():
                display = _extract_property_value(prop_val)
                if display:
                    fields.append(f"  {prop_name}: {display}")
            page_id = page.get("id", "")
            row_text = f"{i}. (id: {page_id})\n" + "\n".join(fields)
            rows.append(row_text)

        return f"Database query returned {len(results)} rows:\n\n" + "\n\n".join(rows)

    except Exception as e:
        logger.error(f"Notion database query failed: {e}", exc_info=True)
        return f"Error querying Notion database: {e}"


# ──────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────

def register_notion_tools() -> None:
    """Register Notion tools with the global ToolRegistry."""
    registry = ToolRegistry.get_instance()

    registry.register(ToolDefinition(
        name="notion_search",
        description=(
            "Search the user's Notion workspace for pages and databases by keyword. "
            "Returns a list of matching items with their IDs, titles, and types. "
            "Use this first to find pages, then use notion_read_page to read content. "
            "IMPORTANT: Use short, simple keywords (1-2 words) for best results. "
            "For example, search 'koi' instead of 'koi checklist'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short search keyword (1-2 words work best, e.g. 'koi' not 'koi checklist')",
                },
                "filter_type": {
                    "type": "string",
                    "enum": ["page", "database"],
                    "description": "Filter results by type (optional)",
                },
                "page_size": {
                    "type": "integer",
                    "description": "Max number of results (default 10, max 100)",
                    "default": 10,
                },
            },
            "required": [],
        },
        executor=notion_search_executor,
        category=ToolCategory.CUSTOM,
    ))

    registry.register(ToolDefinition(
        name="notion_read_page",
        description=(
            "Read the full content of a Notion page by its ID. "
            "Returns the page title and all content blocks (text, headings, "
            "to-do items with [x]/[ ] checked state, lists, code blocks, etc.). "
            "Use notion_search first to find the page ID."
        ),
        parameters={
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "The Notion page ID to read",
                },
            },
            "required": ["page_id"],
        },
        executor=notion_read_page_executor,
        category=ToolCategory.CUSTOM,
    ))

    registry.register(ToolDefinition(
        name="notion_query_database",
        description=(
            "Query a Notion database to get rows with their properties. "
            "Supports optional Notion filter and sort objects. "
            "Use notion_search with filter_type='database' first to find the database ID."
        ),
        parameters={
            "type": "object",
            "properties": {
                "database_id": {
                    "type": "string",
                    "description": "The Notion database ID to query",
                },
                "filter": {
                    "type": "object",
                    "description": "Notion filter object (optional, e.g. {\"property\": \"Status\", \"select\": {\"equals\": \"Done\"}})",
                },
                "sorts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "description": "A Notion sort object with 'property' and 'direction' keys",
                    },
                    "description": "Notion sort objects (optional, e.g. [{\"property\": \"Created\", \"direction\": \"descending\"}])",
                },
                "page_size": {
                    "type": "integer",
                    "description": "Max rows to return (default 20)",
                    "default": 20,
                },
            },
            "required": ["database_id"],
        },
        executor=notion_query_database_executor,
        category=ToolCategory.CUSTOM,
    ))

    logger.info("Notion tools registered")

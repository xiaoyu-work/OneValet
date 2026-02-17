"""
GoogleWorkspaceAgent - Domain agent for all Google Workspace operations.

Replaces GoogleDocsCreateAgent + GoogleSheetsWriteAgent + 3 standalone tools
(google_drive_search, google_docs_read, google_sheets_read) with a single agent
that has its own mini ReAct loop.
"""

import json
import logging
from typing import Annotated, Any, Dict, List, Optional

from onevalet import valet
from onevalet.standard_agent import StandardAgent, AgentToolContext
from onevalet.tool_decorator import tool

from .client import GoogleWorkspaceClient

logger = logging.getLogger(__name__)


# =============================================================================
# Auth helper
# =============================================================================

async def _get_token(context: AgentToolContext):
    """Get Google OAuth token using the agent-style helper."""
    from .auth import get_google_token_for_agent
    return await get_google_token_for_agent(context.tenant_id)


# =============================================================================
# Tool executors
# =============================================================================

@tool
async def google_drive_search(
    query: Annotated[str, "Short search keyword"] = "",
    file_type: Annotated[Optional[str], "Filter by type (optional)"] = None,
    page_size: Annotated[int, "Max results (default 10)"] = 10,
    *,
    context: AgentToolContext,
) -> str:
    """Search Google Drive for files (documents, spreadsheets, folders). Returns names, IDs, types."""
    token, error = await _get_token(context)
    if error:
        return error

    try:
        client = GoogleWorkspaceClient(token)
        files = await client.drive_search(query=query, file_type=file_type, page_size=page_size)

        if not files:
            return f'No files found in Google Drive for "{query}".' if query else "No files found in Google Drive."

        items = []
        for i, f in enumerate(files, 1):
            name = f.get("name", "Untitled")
            file_id = f.get("id", "")
            mime = f.get("mimeType", "")
            modified = f.get("modifiedTime", "")[:10]
            file_type_label = client.format_mime_type(mime)
            items.append(f'{i}. "{name}" (id: {file_id}, type: {file_type_label}, modified: {modified})')

        header = f'Found {len(files)} files for "{query}":' if query else f"Google Drive files ({len(files)}):"
        return header + "\n" + "\n".join(items)
    except Exception as e:
        logger.error(f"Google Drive search failed: {e}", exc_info=True)
        return f"Error searching Google Drive: {e}"


@tool
async def google_docs_read(
    document_id: Annotated[str, "The Google Doc document ID"],
    *,
    context: AgentToolContext,
) -> str:
    """Read the full text of a Google Doc by its ID. Use google_drive_search first."""
    if not document_id:
        return "Error: document_id is required."

    token, error = await _get_token(context)
    if error:
        return error

    try:
        client = GoogleWorkspaceClient(token)
        doc = await client.docs_get(document_id)
        title = doc.get("title", "Untitled")
        text = client.docs_to_text(doc)

        if not text.strip():
            return f'Document: "{title}"\n\nThe document is empty.'
        return f'Document: "{title}"\n\nContent:\n{text}'
    except Exception as e:
        logger.error(f"Google Docs read failed: {e}", exc_info=True)
        return f"Error reading Google Doc: {e}"


@tool
async def google_sheets_read(
    spreadsheet_id: Annotated[str, "The Spreadsheet ID"],
    range: Annotated[str, "Cell range like 'Sheet1!A1:D10' (optional)"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Read data from a Google Spreadsheet by ID. Specify range or omit for first sheet."""
    range_ = range

    if not spreadsheet_id:
        return "Error: spreadsheet_id is required."

    token, error = await _get_token(context)
    if error:
        return error

    try:
        client = GoogleWorkspaceClient(token)

        if not range_:
            metadata = await client.sheets_get_metadata(spreadsheet_id)
            sheets = metadata.get("sheets", [])
            if not sheets:
                return "The spreadsheet has no sheets."
            sheet_name = sheets[0].get("properties", {}).get("title", "Sheet1")
            range_ = sheet_name

        data = await client.sheets_get_values(spreadsheet_id, range_)
        values = data.get("values", [])

        if not values:
            return f'No data found in range "{range_}".'

        # Format as aligned text table
        col_count = max(len(row) for row in values)
        col_widths = [0] * col_count
        for row in values:
            for j, cell in enumerate(row):
                col_widths[j] = max(col_widths[j], len(str(cell)))

        lines = []
        for i, row in enumerate(values):
            cells = []
            for j in range(col_count):
                val = str(row[j]) if j < len(row) else ""
                cells.append(val.ljust(col_widths[j]))
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
                lines.append(sep)

        title = data.get("range", range_)
        return f"Spreadsheet range: {title}\n\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"Google Sheets read failed: {e}", exc_info=True)
        return f"Error reading Google Sheet: {e}"


# =============================================================================
# Approval preview functions
# =============================================================================

async def _docs_create_preview(args: dict, context) -> str:
    title = args.get("title", "Untitled")
    content = args.get("content", "")
    preview = content[:200] + "..." if len(content) > 200 else content
    return f"Create Google Doc?\n\nTitle: {title}\nContent preview:\n{preview}"


async def _sheets_write_preview(args: dict, context) -> str:
    name = args.get("spreadsheet_name", "")
    range_ = args.get("range", "")
    values = args.get("values", "[]")
    preview = values[:300] + "..." if isinstance(values, str) and len(values) > 300 else str(values)[:300]
    return f"Write to Google Sheet?\n\nSpreadsheet: {name}\nRange: {range_}\nData:\n{preview}"


@tool(needs_approval=True, get_preview=_docs_create_preview)
async def google_docs_create(
    title: Annotated[str, "Document title"],
    content: Annotated[str, "Document content in plain text"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Create a new Google Doc with title and content."""
    token, error = await _get_token(context)
    if error:
        return error

    try:
        client = GoogleWorkspaceClient(token)
        doc = await client.docs_create(title=title, body_text=content)
        doc_id = doc.get("documentId", "")
        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"
        return f'Created Google Doc "{title}".\nURL: {doc_url}'
    except Exception as e:
        logger.error(f"Failed to create Google Doc: {e}", exc_info=True)
        return f"Failed to create Google Doc: {e}"


@tool(needs_approval=True, get_preview=_sheets_write_preview)
async def google_sheets_write(
    spreadsheet_name: Annotated[str, "Name of the spreadsheet"],
    range: Annotated[str, "Cell range in A1 notation (e.g. Sheet1!A1:C10)"],
    values: Annotated[str, 'JSON array of arrays, e.g. [["Name","Age"],["Alice",30]]'],
    *,
    context: AgentToolContext,
) -> str:
    """Write data to a Google Spreadsheet. Searches by name to find the spreadsheet."""
    range_ = range

    if not spreadsheet_name:
        return "Error: spreadsheet_name is required."
    if not range_:
        return "Error: range is required (e.g. 'Sheet1!A1:C10')."

    # Parse values JSON
    values_str = values
    try:
        values = json.loads(values_str) if isinstance(values_str, str) else values_str
        if not isinstance(values, list) or not all(isinstance(row, list) for row in values):
            return 'Invalid data format. Values must be a JSON array of arrays, e.g. [["Name","Age"],["Alice",30]]'
    except json.JSONDecodeError as e:
        return f"Invalid JSON in values: {e}"

    token, error = await _get_token(context)
    if error:
        return error

    try:
        client = GoogleWorkspaceClient(token)

        # Resolve spreadsheet ID by name
        files = await client.drive_search(query=spreadsheet_name, file_type="spreadsheet", page_size=3)
        if not files:
            return f'Couldn\'t find a spreadsheet matching "{spreadsheet_name}".'

        spreadsheet_id = files[0]["id"]
        resolved_name = files[0].get("name", spreadsheet_name)

        result = await client.sheets_update_values(
            spreadsheet_id=spreadsheet_id, range_=range_, values=values,
        )
        updated_range = result.get("updatedRange", range_)
        updated_cells = result.get("updatedCells", len(values))
        sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
        return f'Updated "{resolved_name}" range {updated_range} ({updated_cells} cells).\nURL: {sheet_url}'
    except Exception as e:
        logger.error(f"Failed to write to Google Sheet: {e}", exc_info=True)
        return f"Failed to write to Google Sheet: {e}"


# =============================================================================
# Domain Agent
# =============================================================================

@valet(capabilities=["google_workspace", "google_docs", "google_sheets", "google_drive"])
class GoogleWorkspaceAgent(StandardAgent):
    """Search, read, create, and write Google Drive files, Docs, and Sheets. Use when the user mentions Google Docs, Sheets, Drive, or their documents and spreadsheets."""

    max_turns = 5

    domain_system_prompt = """\
You are a Google Workspace assistant with access to Drive, Docs, and Sheets tools.

Available tools:
- google_drive_search: Search files in Google Drive by keyword.
- google_docs_read: Read full text of a Google Doc by ID.
- google_sheets_read: Read data from a Google Spreadsheet by ID.
- google_docs_create: Create a new Google Doc with title and content.
- google_sheets_write: Write data to a Google Spreadsheet.

Instructions:
1. To find files, use google_drive_search with short keywords.
2. To read a document, first search for its ID, then call google_docs_read.
3. To read a spreadsheet, first search for its ID, then call google_sheets_read.
4. To create a new document, call google_docs_create with title and content.
5. To write spreadsheet data, call google_sheets_write with the spreadsheet name, range, and values.
6. If the user's request is ambiguous, ask for clarification WITHOUT calling any tools.
7. After getting tool results, provide a clear summary to the user."""

    tools = [
        google_drive_search,
        google_docs_read,
        google_sheets_read,
        google_docs_create,
        google_sheets_write,
    ]

"""
Google Workspace Tools - Search Drive, read Docs, and read Sheets.

These tools are used by the orchestrator's ReAct loop so the LLM can
chain multiple Google Workspace operations (search → read → answer) naturally.

Requires Google OAuth credentials connected via Settings.
"""

import logging
from typing import Any, Dict, List

from onevalet.tools import ToolRegistry, ToolDefinition, ToolCategory, ToolExecutionContext

logger = logging.getLogger(__name__)


def _get_client(access_token: str):
    """Get a GoogleWorkspaceClient instance (lazy import to avoid circular deps)."""
    from onevalet.builtin_agents.google_workspace.client import GoogleWorkspaceClient
    return GoogleWorkspaceClient(access_token)


async def _get_token(context: ToolExecutionContext):
    """Get a valid Google token, returning (token, None) or (None, error_message)."""
    from onevalet.builtin_agents.google_workspace.auth import get_google_token
    return await get_google_token(context)


# ──────────────────────────────────────────────────────────────
# Tool Executors
# ──────────────────────────────────────────────────────────────

async def google_drive_search_executor(args: dict, context: ToolExecutionContext = None) -> str:
    """Search the user's Google Drive for files."""
    query = args.get("query", "")
    file_type = args.get("file_type")
    page_size = args.get("page_size", 10)

    token, error = await _get_token(context)
    if error:
        return error

    try:
        client = _get_client(token)
        files = await client.drive_search(query=query, file_type=file_type, page_size=page_size)

        if not files:
            if query:
                return f'No files found in Google Drive for "{query}".'
            return "No files found in Google Drive."

        items = []
        for i, f in enumerate(files, 1):
            name = f.get("name", "Untitled")
            file_id = f.get("id", "")
            mime = f.get("mimeType", "")
            modified = f.get("modifiedTime", "")[:10]
            file_type_label = client.format_mime_type(mime)
            items.append(f"{i}. \"{name}\" (id: {file_id}, type: {file_type_label}, modified: {modified})")

        header = f'Found {len(files)} files for "{query}":' if query else f"Google Drive files ({len(files)}):"
        return header + "\n" + "\n".join(items)

    except Exception as e:
        logger.error(f"Google Drive search failed: {e}", exc_info=True)
        return f"Error searching Google Drive: {e}"


async def google_docs_read_executor(args: dict, context: ToolExecutionContext = None) -> str:
    """Read the full text content of a Google Doc."""
    document_id = args.get("document_id", "")

    if not document_id:
        return "Error: document_id is required."

    token, error = await _get_token(context)
    if error:
        return error

    try:
        client = _get_client(token)
        doc = await client.docs_get(document_id)
        title = doc.get("title", "Untitled")
        text = client.docs_to_text(doc)

        if not text.strip():
            return f'Document: "{title}"\n\nThe document is empty.'

        return f'Document: "{title}"\n\nContent:\n{text}'

    except Exception as e:
        logger.error(f"Google Docs read failed: {e}", exc_info=True)
        return f"Error reading Google Doc: {e}"


async def google_sheets_read_executor(args: dict, context: ToolExecutionContext = None) -> str:
    """Read data from a Google Spreadsheet."""
    spreadsheet_id = args.get("spreadsheet_id", "")
    range_ = args.get("range", "")

    if not spreadsheet_id:
        return "Error: spreadsheet_id is required."

    token, error = await _get_token(context)
    if error:
        return error

    try:
        client = _get_client(token)

        # If no range specified, read all values from the first sheet
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
            return f"No data found in range \"{range_}\"."

        # Format as aligned text table with | separators
        # Calculate column widths
        col_count = max(len(row) for row in values)
        col_widths = [0] * col_count
        for row in values:
            for j, cell in enumerate(row):
                col_widths[j] = max(col_widths[j], len(str(cell)))

        # Build table
        lines = []
        for i, row in enumerate(values):
            cells = []
            for j in range(col_count):
                val = str(row[j]) if j < len(row) else ""
                cells.append(val.ljust(col_widths[j]))
            lines.append("| " + " | ".join(cells) + " |")
            # Add separator after header row
            if i == 0:
                sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
                lines.append(sep)

        title = data.get("range", range_)
        return f"Spreadsheet range: {title}\n\n" + "\n".join(lines)

    except Exception as e:
        logger.error(f"Google Sheets read failed: {e}", exc_info=True)
        return f"Error reading Google Sheet: {e}"


# ──────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────

def register_google_workspace_tools() -> None:
    """Register Google Workspace tools with the global ToolRegistry."""
    registry = ToolRegistry.get_instance()

    registry.register(ToolDefinition(
        name="google_drive_search",
        description=(
            "Search the user's Google Drive for files (documents, spreadsheets, folders). "
            "Returns file names, IDs, types, and modification dates. "
            "Use short keywords for best results. "
            "Use this first to find file IDs, then use google_docs_read or google_sheets_read."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short search keyword",
                },
                "file_type": {
                    "type": "string",
                    "enum": ["document", "spreadsheet", "folder"],
                    "description": "Filter by file type (optional)",
                },
                "page_size": {
                    "type": "integer",
                    "description": "Max number of results (default 10, max 100)",
                    "default": 10,
                },
            },
            "required": [],
        },
        executor=google_drive_search_executor,
        category=ToolCategory.CUSTOM,
    ))

    registry.register(ToolDefinition(
        name="google_docs_read",
        description=(
            "Read the full text content of a Google Doc by its ID. "
            "Returns the document title and all text content. "
            "Use google_drive_search first to find the document ID."
        ),
        parameters={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The Google Doc document ID",
                },
            },
            "required": ["document_id"],
        },
        executor=google_docs_read_executor,
        category=ToolCategory.CUSTOM,
    ))

    registry.register(ToolDefinition(
        name="google_sheets_read",
        description=(
            "Read data from a Google Spreadsheet. "
            "Returns cell values as a formatted table. "
            "Use google_drive_search first to find the spreadsheet ID. "
            "Specify a range like 'Sheet1!A1:D10' or omit to read the first sheet."
        ),
        parameters={
            "type": "object",
            "properties": {
                "spreadsheet_id": {
                    "type": "string",
                    "description": "The Google Spreadsheet ID",
                },
                "range": {
                    "type": "string",
                    "description": "Cell range to read, e.g. 'Sheet1!A1:D10' (optional, reads first sheet if omitted)",
                },
            },
            "required": ["spreadsheet_id"],
        },
        executor=google_sheets_read_executor,
        category=ToolCategory.CUSTOM,
    ))

    logger.info("Google Workspace tools registered")

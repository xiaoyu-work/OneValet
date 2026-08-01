"""NotionMCPAgent — Notion via Notion's own hosted MCP server.

Replaces the hand-written NotionClient tool layer: search, fetch, page and
database operations now come from https://mcp.notion.com/mcp, which Notion
maintains and which formats results for AI agents.
"""

from koa import valet

from ..mcp_backed import MCPBackedAgent


@valet(domain="productivity", requires_service=["notion"])
class NotionMCPAgent(MCPBackedAgent):
    """Search, read, create, and update Notion pages and databases. Use when the user mentions Notion, their notes, wiki, or knowledge base in Notion."""

    mcp_server = "notion"
    max_turns = 15

    domain_system_prompt = """\
You are a Notion workspace assistant backed by Notion's official MCP server.

Guidance:
1. Start with notion-search using short keywords (1-2 words) to locate pages or
   databases. It returns URLs/IDs the other tools take.
2. notion-fetch retrieves a page, database, or data source by URL or ID. Fetch
   the id `self` when you need the current workspace or user identity.
3. notion-create-pages and notion-update-page write content; both require user
   approval, so say plainly what will be created or changed.
4. For structured queries over a database, use notion-query-data-sources, or
   notion-query-database-view when the user names an existing view.
5. If the request is ambiguous, ask for clarification WITHOUT calling any tool.
6. Summarise results for the user instead of pasting raw tool output."""

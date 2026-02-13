"""
OneValet Built-in Tools - Common tools for agent use

Provides:
- google_search: Web search via Google Custom Search API
- important_dates: CRUD for birthdays, anniversaries, etc.
- user_tools: User profile and connected accounts lookup
- notion: Search, read pages, and query databases in Notion
- google_workspace: Search Drive, read Docs, and read Sheets

All tools use the standard onevalet tool pattern and CredentialStore
for credential access.

Usage:
    from onevalet.builtin_agents.tools import register_all_builtin_tools

    register_all_builtin_tools()
"""

from .google_search import register_google_search_tools
from .important_dates import register_important_dates_tools
from .user_tools import register_user_tools


def register_all_builtin_tools() -> None:
    """Register all built-in tools with the global ToolRegistry.

    Note: Notion and Google Workspace tools are no longer registered here.
    They have been consolidated into NotionDomainAgent and
    GoogleWorkspaceDomainAgent respectively.
    """
    register_google_search_tools()
    register_important_dates_tools()
    register_user_tools()


__all__ = [
    "register_all_builtin_tools",
    "register_google_search_tools",
    "register_important_dates_tools",
    "register_user_tools",
]

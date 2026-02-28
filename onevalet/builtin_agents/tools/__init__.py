"""
OneValet Built-in Tools - Common tools for agent use

Provides executor functions and schemas for orchestrator-level tools:
- google_search: Web search via Google Custom Search API
- web_fetch: Fetch and extract readable content from a URL
- important_dates: CRUD for birthdays, anniversaries, etc.
- user_tools: User profile and connected accounts lookup
"""

from .google_search import google_search_executor, GOOGLE_SEARCH_SCHEMA
from .web_fetch import web_fetch_executor, WEB_FETCH_SCHEMA
from .important_dates import IMPORTANT_DATES_TOOL_DEFS
from .user_tools import (
    get_user_accounts_executor,
    get_user_profile_executor,
    GET_USER_ACCOUNTS_SCHEMA,
    GET_USER_PROFILE_SCHEMA,
)

__all__ = [
    "google_search_executor",
    "GOOGLE_SEARCH_SCHEMA",
    "web_fetch_executor",
    "WEB_FETCH_SCHEMA",
    "IMPORTANT_DATES_TOOL_DEFS",
    "get_user_accounts_executor",
    "get_user_profile_executor",
    "GET_USER_ACCOUNTS_SCHEMA",
    "GET_USER_PROFILE_SCHEMA",
]

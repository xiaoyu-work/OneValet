"""
Notion integration for Koa

NotionMCPAgent handles all Notion operations (search, fetch, create, update)
through Notion's official hosted MCP server.
"""

from .agent import NotionMCPAgent

__all__ = [
    "NotionMCPAgent",
]

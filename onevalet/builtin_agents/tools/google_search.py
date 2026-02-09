"""
Google Search Tool - Web search via Google Custom Search API

Requires environment variables:
- GOOGLE_SEARCH_API_KEY: Your Google API key
- GOOGLE_SEARCH_ENGINE_ID: Your Custom Search Engine ID
"""

import os
import logging

import httpx

from onevalet.tools import ToolRegistry, ToolDefinition, ToolCategory, ToolExecutionContext

logger = logging.getLogger(__name__)


async def google_search_executor(args: dict, context: ToolExecutionContext = None) -> str:
    """Search the web using Google Custom Search API."""
    query = args.get("query", "")
    num_results = args.get("num_results", 5)

    if not query:
        return "Error: No search query provided."

    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        return (
            "Error: Google Search API not configured. "
            "Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables."
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": api_key,
                    "cx": search_engine_id,
                    "q": query,
                    "num": min(num_results, 10),
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                logger.error(f"Google Search API error: {response.status_code} - {response.text}")
                return f"Error: Search failed with status {response.status_code}"

            data = response.json()
            items = data.get("items", [])

            if not items:
                return f"No results found for '{query}'."

            output = []
            for i, item in enumerate(items, 1):
                title = item.get("title", "No title")
                link = item.get("link", "")
                snippet = item.get("snippet", "").replace("\n", " ")
                output.append(
                    f"{i}. {title}\n"
                    f"   URL: {link}\n"
                    f"   {snippet}"
                )

            total_results = data.get("searchInformation", {}).get("totalResults", "?")
            return (
                f"Found approximately {total_results} results for '{query}'.\n"
                f"Top {len(items)} results:\n\n" + "\n\n".join(output)
            )

    except httpx.TimeoutException:
        return "Error: Search request timed out"
    except Exception as e:
        logger.error(f"Google Search error: {e}", exc_info=True)
        return f"Error: {e}"


def register_google_search_tools() -> None:
    """Register google_search tool with the global ToolRegistry."""
    registry = ToolRegistry.get_instance()

    registry.register(ToolDefinition(
        name="google_search",
        description=(
            "Search the web using Google. "
            "Returns titles, URLs, and snippets of top results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (max 10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        executor=google_search_executor,
        category=ToolCategory.WEB,
    ))

    logger.info("Google search tool registered")

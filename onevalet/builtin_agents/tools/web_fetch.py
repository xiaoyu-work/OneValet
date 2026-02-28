"""
Web Fetch Tool - Fetch and extract readable content from a URL.

Uses trafilatura for high-quality article extraction (readability-style),
with httpx for async fetching. Includes SSRF guard and in-memory cache.
"""

import hashlib
import logging
import re
import time
from typing import Dict, Optional, Tuple

import httpx
import trafilatura

from onevalet.models import AgentToolContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple in-memory cache (URL -> (content, expire_ts))
# ---------------------------------------------------------------------------

_cache: Dict[str, Tuple[str, float]] = {}
_CACHE_TTL_S = 300  # 5 minutes
_MAX_CACHE_ENTRIES = 64


def _cache_get(url: str) -> Optional[str]:
    key = hashlib.sha256(url.encode()).hexdigest()
    entry = _cache.get(key)
    if entry and entry[1] > time.time():
        return entry[0]
    _cache.pop(key, None)
    return None


def _cache_set(url: str, content: str) -> None:
    if len(_cache) >= _MAX_CACHE_ENTRIES:
        oldest_key = min(_cache, key=lambda k: _cache[k][1])
        _cache.pop(oldest_key, None)
    key = hashlib.sha256(url.encode()).hexdigest()
    _cache[key] = (content, time.time() + _CACHE_TTL_S)


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

_BLOCKED_PATTERNS = re.compile(
    r"^https?://"
    r"(localhost|127\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|0\.0\.0\.0|\[::1?\])",
    re.IGNORECASE,
)


def _is_safe_url(url: str) -> bool:
    return not _BLOCKED_PATTERNS.match(url)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

_MAX_CONTENT_CHARS = 12000
_USER_AGENT = (
    "Mozilla/5.0 (compatible; OneValet/1.0; +https://github.com/xiaoyu-work/onevalet)"
)


async def web_fetch_executor(args: dict, context: AgentToolContext = None) -> str:
    """Fetch a URL and return its content as readable text."""
    url = (args.get("url") or "").strip()
    include_links = args.get("include_links", False)

    if not url:
        return "Error: url is required."

    # Ensure scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # SSRF guard
    if not _is_safe_url(url):
        return "Error: cannot fetch internal or private network URLs."

    # Check cache
    cached = _cache_get(url)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=20.0,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = await client.get(url)

        if resp.status_code >= 400:
            return f"Error: HTTP {resp.status_code} fetching {url}"

        content_type = resp.headers.get("content-type", "")

        # Plain text or JSON — return directly
        if "text/plain" in content_type or "application/json" in content_type:
            text = resp.text[:_MAX_CONTENT_CHARS]
            _cache_set(url, text)
            return text

        # HTML → extract with trafilatura
        raw_html = resp.text
        text = trafilatura.extract(
            raw_html,
            url=url,
            include_links=include_links,
            include_tables=True,
            output_format="txt",
            favor_recall=True,
        )

        if not text or len(text.strip()) < 30:
            # Fallback: try with less strict settings
            text = trafilatura.extract(
                raw_html,
                url=url,
                include_links=include_links,
                include_tables=True,
                output_format="txt",
                favor_recall=True,
                no_fallback=False,
            )

        if not text or len(text.strip()) < 30:
            return (
                f"Fetched {url} but could not extract meaningful content. "
                "The page may require JavaScript or be behind a login."
            )

        # Truncate
        if len(text) > _MAX_CONTENT_CHARS:
            text = text[:_MAX_CONTENT_CHARS] + "\n\n[Content truncated]"

        # Extract title if available
        metadata = trafilatura.extract(
            raw_html, url=url, output_format="xmltei",
        )
        title = ""
        if metadata:
            import re as _re
            m = _re.search(r"<title[^>]*>([^<]+)</title>", metadata)
            if m:
                title = m.group(1).strip()

        header = f"Content from {url}"
        if title:
            header = f"{title}\nSource: {url}"

        result = f"{header}\n\n{text}"
        _cache_set(url, result)
        return result

    except httpx.TimeoutException:
        return f"Error: request to {url} timed out (20s)."
    except httpx.ConnectError as e:
        return f"Error: could not connect to {url}: {e}"
    except Exception as e:
        logger.error(f"web_fetch error for {url}: {e}", exc_info=True)
        return f"Error fetching {url}: {e}"


WEB_FETCH_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The URL to fetch and read content from.",
        },
        "include_links": {
            "type": "boolean",
            "description": "Whether to include hyperlinks in the extracted text (default false).",
            "default": False,
        },
    },
    "required": ["url"],
}

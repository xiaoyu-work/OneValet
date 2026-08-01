"""Cloud storage tools — search, inspect, download, share files, and check usage.

All tools fan out across every connected provider (Google Drive, OneDrive,
Dropbox) unless the user names one. A provider that fails is reported rather
than silently dropped, so a partial result never looks like an empty one.
"""

import logging
from datetime import datetime
from typing import Annotated, Any, Dict, List, Tuple

from koa.models import AgentToolContext

from ...tool_decorator import tool

logger = logging.getLogger(__name__)


# ── provider plumbing ────────────────────────────────────────────────


async def _accounts(tenant_id: str, provider_spec: str = "") -> List[dict]:
    """Connected accounts, narrowed to one provider when the user named it."""
    from koa.providers.cloud_storage.resolver import CloudStorageResolver

    if provider_spec and provider_spec.lower() != "all":
        account = await CloudStorageResolver.resolve(tenant_id, provider_spec)
        return [account] if account else []
    return await CloudStorageResolver.resolve_all(tenant_id)


def _factory():
    from koa.providers.cloud_storage.factory import CloudStorageProviderFactory

    return CloudStorageProviderFactory


def _failed(account: dict, reason: str, error: str = "") -> dict:
    return {
        "provider": account.get("provider", ""),
        "email": account.get("email", ""),
        "reason": reason,
        "error": error,
    }


async def _usable_provider(account: dict, failures: List[dict]):
    """The live provider for an account, or None with the reason recorded."""
    provider = _factory().create_provider(account)
    if not provider:
        failures.append(_failed(account, "unsupported_provider"))
        return None
    if not await provider.ensure_valid_token():
        failures.append(_failed(account, "token_expired"))
        return None
    return provider


async def _find_one_file(
    tenant_id: str, query: str, provider_spec: str = ""
) -> Tuple[Any, Dict[str, Any], dict]:
    """First file matching *query*, with the provider and account that hold it."""
    for account in await _accounts(tenant_id, provider_spec):
        provider = _factory().create_provider(account)
        if not provider or not await provider.ensure_valid_token():
            continue
        try:
            result = await provider.search_files(query=query, max_results=1)
            if result.get("success") and result.get("data"):
                return provider, result["data"][0], account
        except Exception as e:
            logger.error(f"Lookup failed on {account.get('provider')}: {e}", exc_info=True)
    return None, {}, {}


# ── formatting ───────────────────────────────────────────────────────


def _format_size(size) -> str:
    from koa.providers.cloud_storage.base import BaseCloudStorageProvider

    return BaseCloudStorageProvider.format_size(size)


def _format_date(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        from dateutil import parser as date_parser

        dt = date_parser.parse(date_str)
        if dt.year == datetime.now().year:
            return dt.strftime("%b %d").lstrip("0")
        return dt.strftime("%b %d, %Y").lstrip("0")
    except Exception:
        return date_str


def _append_failures(parts: List[str], failures: List[Dict]) -> None:
    """Explain each unreachable account in terms the user can act on."""
    for failure in failures:
        display = failure.get("email") or failure.get("provider") or "cloud storage"
        reason = failure.get("reason", "unknown")
        if reason == "token_expired":
            parts.append(
                f"\nI lost access to your {display} account. Could you reconnect it in settings?"
            )
        elif reason == "unsupported_provider":
            parts.append(f"\nSorry, I can't access {display} yet - that provider isn't supported.")
        else:
            parts.append(f"\nI had trouble checking {display}. Want me to try again later?")


def _format_files(
    files: List[Dict],
    searched: List[Dict],
    failures: List[Dict],
    action_label: str = "",
) -> str:
    if not files and not failures:
        return f"No files found for {action_label}." if action_label else "No files found."

    parts: List[str] = []
    multi_provider = len(searched) > 1

    if not files:
        parts.append(f"No files found for {action_label}." if action_label else "No files found.")
    else:
        parts.append(f"Found {len(files)} file(s):\n")
        for i, f in enumerate(files, 1):
            name = f.get("name", "Untitled")
            size = f.get("size")
            size_str = f" - {_format_size(size)}" if size is not None else ""
            modified = f.get("modified", "")
            date_str = f" - modified {_format_date(modified)}" if modified else ""
            display = f.get("_provider_display", "")
            prefix = f"[{display}] " if multi_provider and display else ""
            parts.append(f"{i}. {prefix}{name}{size_str}{date_str}")

    _append_failures(parts, failures)
    return "\n".join(parts)


async def _collect_files(accounts: List[dict], fetch, failures: List[dict], fail_reason: str):
    """Run *fetch* on every account, tagging results with their provider."""
    all_files: List[Dict] = []
    for account in accounts:
        provider = await _usable_provider(account, failures)
        if not provider:
            continue
        try:
            result = await fetch(provider)
            if result.get("success"):
                for f in result.get("data", []):
                    f["_provider"] = account.get("provider", "")
                    f["_provider_display"] = provider.get_provider_display_name()
                    all_files.append(f)
            else:
                failures.append(_failed(account, fail_reason, result.get("error")))
        except Exception as e:
            logger.error(f"{fail_reason} on {account.get('provider')}: {e}", exc_info=True)
            failures.append(_failed(account, "query_failed", str(e)))
    all_files.sort(key=lambda f: f.get("modified", ""), reverse=True)
    return all_files


_NO_ACCOUNTS = "No cloud storage accounts found. Please connect one in settings."


# ── tools ────────────────────────────────────────────────────────────


@tool(risk_level="read", category="cloud_storage")
async def search_files(
    query: Annotated[str, "What to search for — a file name or keywords."],
    provider: Annotated[
        str, "Restrict to one provider: google, onedrive, or dropbox. Empty searches all."
    ] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Search the user's cloud storage for files matching a query."""
    if not query.strip():
        return "What would you like to search for?"

    accounts = await _accounts(context.tenant_id, provider)
    if not accounts:
        return _NO_ACCOUNTS

    failures: List[dict] = []
    files = await _collect_files(
        accounts, lambda p: p.search_files(query=query), failures, "search_failed"
    )
    return _format_files(files, accounts, failures, f'search "{query}"')


@tool(risk_level="read", category="cloud_storage")
async def list_recent_files(
    provider: Annotated[
        str, "Restrict to one provider: google, onedrive, or dropbox. Empty lists all."
    ] = "",
    *,
    context: AgentToolContext,
) -> str:
    """List the user's most recently modified cloud storage files."""
    accounts = await _accounts(context.tenant_id, provider)
    if not accounts:
        return _NO_ACCOUNTS

    failures: List[dict] = []
    files = await _collect_files(
        accounts, lambda p: p.list_recent_files(), failures, "list_failed"
    )
    return _format_files(files, accounts, failures, "recent files")


@tool(risk_level="read", category="cloud_storage")
async def get_file_info(
    query: Annotated[str, "Name or keywords identifying the file."],
    provider: Annotated[str, "Restrict to one provider: google, onedrive, or dropbox."] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Get details about a file: type, size, path, sharing status, and link."""
    if not query.strip():
        return "Which file would you like info about?"

    found_provider, file_data, _ = await _find_one_file(context.tenant_id, query, provider)
    if not found_provider or not file_data.get("id"):
        return f'Could not find a file matching "{query}".'

    try:
        info = await found_provider.get_file_info(file_data["id"])
    except Exception as e:
        logger.error(f"File info failed: {e}", exc_info=True)
        return f'Could not read details for "{query}".'
    if not info.get("success"):
        return f'Could not find a file matching "{query}".'

    data = info["data"]
    parts = [f"[{found_provider.get_provider_display_name()}] {data.get('name', 'Unknown')}"]
    if data.get("type"):
        parts.append(f"Type: {data['type']}")
    if data.get("size") is not None:
        parts.append(f"Size: {_format_size(data['size'])}")
    if data.get("modified"):
        parts.append(f"Modified: {_format_date(data['modified'])}")
    if data.get("path"):
        parts.append(f"Path: {data['path']}")
    if data.get("shared"):
        parts.append("Shared: Yes")
    if data.get("url"):
        parts.append(f"Link: {data['url']}")
    return "\n".join(parts)


@tool(risk_level="read", category="cloud_storage")
async def get_download_link(
    query: Annotated[str, "Name or keywords identifying the file to download."],
    provider: Annotated[str, "Restrict to one provider: google, onedrive, or dropbox."] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Get a download link for a file in the user's cloud storage."""
    if not query.strip():
        return "Which file would you like to download?"

    found_provider, file_data, _ = await _find_one_file(context.tenant_id, query, provider)
    if not found_provider or not file_data.get("id"):
        return f'Could not find a file matching "{query}" to download.'

    try:
        result = await found_provider.get_download_link(file_data["id"])
    except Exception as e:
        logger.error(f"Download link failed: {e}", exc_info=True)
        return f'Could not get a download link for "{query}".'
    if not result.get("success"):
        return f'Could not find a file matching "{query}" to download.'

    name = file_data.get("name", query)
    url = result["data"].get("url", "")
    msg = f"[{found_provider.get_provider_display_name()}] {name}\nDownload: {url}"
    if result["data"].get("expires"):
        msg += f"\n(link expires: {result['data']['expires']})"
    return msg


async def _preview_share_file(args: dict, context: AgentToolContext) -> str:
    """Resolve the file first so the user approves a named file, not a guess."""
    query = args.get("query", "")
    target = args.get("target", "")
    provider_name = ""
    file_name = query or "file"

    try:
        found_provider, file_data, _ = await _find_one_file(
            context.tenant_id, query, args.get("provider", "")
        )
        if found_provider:
            provider_name = f" on {found_provider.get_provider_display_name()}"
            file_name = file_data.get("name", query)
    except Exception as e:
        logger.error(f"Share preview lookup failed: {e}")

    return (
        f'Share "{file_name}"{provider_name} with {target}?\n\n'
        "(yes / no / or describe changes)"
    )


@tool(
    needs_approval=True,
    risk_level="write",
    category="cloud_storage",
    get_preview=_preview_share_file,
)
async def share_file(
    query: Annotated[str, "Name or keywords identifying the file to share."],
    target: Annotated[str, "Email address of the person to share the file with."],
    provider: Annotated[str, "Restrict to one provider: google, onedrive, or dropbox."] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Share a cloud storage file with someone by email."""
    if not query.strip():
        return "Which file would you like to share?"
    if not target.strip():
        return "Who would you like to share the file with? (email address)"

    found_provider, file_data, _ = await _find_one_file(context.tenant_id, query, provider)
    if not found_provider or not file_data.get("id"):
        return f'Could not find a file matching "{query}" to share.'

    try:
        result = await found_provider.share_file(file_id=file_data["id"], email=target)
    except Exception as e:
        logger.error(f"Share failed: {e}", exc_info=True)
        return "Something went wrong sharing the file. Want to try again?"

    if not result.get("success"):
        return f"Couldn't share the file: {result.get('error', 'Unknown error')}"

    name = file_data.get("name", query)
    msg = f"Shared [{found_provider.get_provider_display_name()}] {name} with {target}"
    share_url = result.get("data", {}).get("url", "")
    if share_url:
        msg += f"\nLink: {share_url}"
    return msg


@tool(risk_level="read", category="cloud_storage")
async def get_storage_usage(
    *,
    context: AgentToolContext,
) -> str:
    """Show how much space is used across the user's cloud storage accounts."""
    accounts = await _accounts(context.tenant_id)
    if not accounts:
        return _NO_ACCOUNTS

    usage_parts: List[str] = []
    failures: List[dict] = []

    for account in accounts:
        provider = await _usable_provider(account, failures)
        if not provider:
            continue
        try:
            result = await provider.get_storage_usage()
            if result.get("success"):
                data = result["data"]
                usage_parts.append(
                    f"[{provider.get_provider_display_name()}] "
                    f"{_format_size(data.get('used', 0))} / {_format_size(data.get('total', 0))} "
                    f"({data.get('percent', 0):.1f}% used)"
                )
            else:
                failures.append(_failed(account, "usage_failed", result.get("error")))
        except Exception as e:
            logger.error(f"Usage check failed on {account.get('provider')}: {e}", exc_info=True)
            failures.append(_failed(account, "query_failed", str(e)))

    if not usage_parts and not failures:
        return _NO_ACCOUNTS

    parts: List[str] = []
    if usage_parts:
        parts.append("Storage usage:\n")
        parts.extend(usage_parts)
    _append_failures(parts, failures)
    return "\n".join(parts)

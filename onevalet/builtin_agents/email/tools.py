"""
Email tools for EmailAgent.

Extracted from legacy email agents (SendEmailAgent, ReadEmailAgent,
ReplyEmailAgent, DeleteEmailAgent, ArchiveEmailAgent, MarkReadEmailAgent).
"""
import logging
import html
from typing import Annotated, Any, Dict, List, Optional

from onevalet.tool_decorator import tool
from onevalet.models import AgentToolContext

logger = logging.getLogger(__name__)


# ============================================================
# Shared helpers
# ============================================================

async def _resolve_provider(tenant_id: str, account_spec: str = "primary"):
    """Resolve a single email account and create its provider.

    Returns (account, provider, error_message).
    On success error_message is None; on failure provider is None.
    """
    from onevalet.providers.email.resolver import AccountResolver
    from onevalet.providers.email.factory import EmailProviderFactory

    account = await AccountResolver.resolve_account(tenant_id, account_spec)
    if not account:
        return None, None, f"No email account found for '{account_spec}'."

    provider = EmailProviderFactory.create_provider(account)
    if not provider:
        email = account.get("account_identifier", account_spec)
        return account, None, f"Unsupported email provider for {email}."

    if not await provider.ensure_valid_token():
        email = account.get("account_identifier", account_spec)
        return account, None, f"Lost access to {email}. Please reconnect in settings."

    return account, provider, None


async def _resolve_all_providers(tenant_id: str, account_specs=None):
    """Resolve multiple email accounts and create providers."""
    from onevalet.providers.email.resolver import AccountResolver
    from onevalet.providers.email.factory import EmailProviderFactory

    if not account_specs:
        account_specs = ["all"]

    accounts = await AccountResolver.resolve_accounts(tenant_id, account_specs)
    if not accounts:
        return [], ["No email accounts found. Please connect one in settings."]

    providers = []
    errors = []
    for account in accounts:
        provider = EmailProviderFactory.create_provider(account)
        if not provider:
            errors.append(f"{account.get('account_name', 'unknown')}: unsupported provider")
            continue
        if not await provider.ensure_valid_token():
            errors.append(f"{account.get('account_identifier', 'unknown')}: token expired, reconnect in settings")
            continue
        providers.append((account, provider))

    return providers, errors


def _format_sender(sender_raw: str) -> str:
    """Extract display name from 'Name <email>' format."""
    sender = html.unescape(sender_raw)
    if "<" in sender:
        return sender.split("<")[0].strip().strip('"')
    return sender


# ============================================================
# search_emails
# ============================================================

@tool
async def search_emails(
    query: Annotated[Optional[str], "Search keywords (subject, content)"] = None,
    sender: Annotated[Optional[str], "Filter by sender name or email"] = None,
    unread_only: Annotated[bool, "Only show unread emails (default: true)"] = True,
    days_back: Annotated[int, "Days to search back (default: 7)"] = 7,
    date_range: Annotated[Optional[str], "Date range like 'today', 'yesterday', 'last week'"] = None,
    accounts: Annotated[Optional[str], "Account to search: 'all', 'primary', or account name"] = None,
    max_results: Annotated[int, "Max results to return (default: 15)"] = 15,
    category: Annotated[Optional[str], "Inbox category filter: 'primary' (default), 'social', 'promotions', 'updates', or 'all'"] = "primary",
    *,
    context: AgentToolContext,
) -> str:
    """Search emails across connected accounts. Returns email list with message_ids."""
    tenant_id = context.tenant_id

    if category and category.lower() != "all":
        include_categories = [category.lower()]
    else:
        include_categories = None

    account_list = [accounts] if isinstance(accounts, str) else accounts

    providers, errors = await _resolve_all_providers(tenant_id, account_list)
    if not providers:
        return "; ".join(errors) if errors else "No email accounts available."

    all_emails: List[Dict[str, Any]] = []
    for account, provider in providers:
        try:
            effective_query = query
            meta_keywords = {"unread", "new", "recent", "latest", "all", "emails", "email", "inbox", "check"}
            if effective_query and effective_query.lower().strip() in meta_keywords:
                effective_query = None

            result = await provider.search_emails(
                query=effective_query,
                sender=sender,
                date_range=date_range,
                unread_only=unread_only,
                days_back=days_back,
                include_categories=include_categories,
                max_results=max_results,
            )
            if result.get("success"):
                emails = result.get("data", [])
                for email in emails:
                    email["_account_name"] = account["account_name"]
                    email["_account_email"] = account["account_identifier"]
                all_emails.extend(emails)
            else:
                errors.append(f"{account['account_name']}: {result.get('error', 'search failed')}")
        except Exception as e:
            errors.append(f"{account['account_name']}: {e}")

    if not all_emails:
        msg = "No emails found matching your search."
        if errors:
            msg += f"\nWarnings: {'; '.join(errors)}"
        return msg

    lines = [f"Found {len(all_emails)} email(s):"]
    for i, email in enumerate(all_emails[:max_results], 1):
        sender_name = _format_sender(email.get("sender", "Unknown"))
        subject = html.unescape(email.get("subject", "No subject"))
        snippet = html.unescape(email.get("snippet", ""))[:100]
        unread_mark = " [UNREAD]" if email.get("unread") else ""
        msg_id = email.get("message_id", "")
        acct = email.get("_account_name", "")

        lines.append(f"{i}. From: {sender_name} | Subject: {subject}{unread_mark}")
        if snippet:
            lines.append(f"   Preview: {snippet}")
        lines.append(f"   [message_id: {msg_id}, account: {acct}]")

    if len(all_emails) > max_results:
        lines.append(f"\n... and {len(all_emails) - max_results} more email(s).")
    if errors:
        lines.append(f"\nWarnings: {'; '.join(errors)}")

    return "\n".join(lines)


# ============================================================
# send_email (needs_approval)
# ============================================================

async def _preview_send_email(args: dict, context) -> str:
    """Preview for send_email approval."""
    to = args.get("to", "")
    subject = args.get("subject", "Quick note")
    body = args.get("body", "")
    user_profile = context.user_profile or {}
    first_name = user_profile.get("first_name", "")
    if first_name:
        body_preview = f"{body}\n\nThanks,\n{first_name}"
    else:
        body_preview = f"{body}\n\nThanks"
    return f"Email Draft:\nTo: {to}\nSubject: {subject}\n\n{body_preview}\n\n---\nSend this?"


@tool(needs_approval=True, get_preview=_preview_send_email)
async def send_email(
    to: Annotated[str, "Recipient email address"],
    body: Annotated[str, "Email body content (plain text)"],
    subject: Annotated[str, "Email subject line"] = "Quick note",
    from_account: Annotated[str, "Account to send from (default: 'primary')"] = "primary",
    *,
    context: AgentToolContext,
) -> str:
    """Send an email. Requires recipient, subject, and body."""
    tenant_id = context.tenant_id

    if not to:
        return "Error: recipient email address is required."
    if not body:
        return "Error: email body is required."

    account, provider, error = await _resolve_provider(tenant_id, from_account)
    if error:
        return error

    user_profile = context.user_profile or {}
    first_name = user_profile.get("first_name", "")
    if first_name:
        body_with_sig = f"{body}\n\nThanks,\n{first_name}"
    else:
        body_with_sig = f"{body}\n\nThanks"

    try:
        result = await provider.send_email(to=to, subject=subject, body=body_with_sig)
        if result.get("success"):
            return f"Email sent to {to}."
        else:
            return f"Failed to send: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.error(f"send_email failed: {e}", exc_info=True)
        return f"Error sending email: {e}"


# ============================================================
# reply_email (needs_approval)
# ============================================================

async def _preview_reply_email(args: dict, context) -> str:
    """Preview for reply_email approval."""
    body = args.get("body", "")
    reply_all = args.get("reply_all", False)
    suffix = " (reply all)" if reply_all else ""
    return f"Reply Draft{suffix}:\n\n{body}\n\n---\nSend this reply?"


@tool(needs_approval=True, get_preview=_preview_reply_email)
async def reply_email(
    message_id: Annotated[str, "Message ID of the email to reply to (from search_emails)"],
    body: Annotated[str, "Reply content"],
    reply_all: Annotated[bool, "Reply to all recipients (default: false)"] = False,
    account: Annotated[str, "Account name (from search_emails results)"] = "primary",
    *,
    context: AgentToolContext,
) -> str:
    """Reply to an email. Use message_id from search_emails results."""
    tenant_id = context.tenant_id

    if not message_id:
        return "Error: message_id is required. Use search_emails first to find it."
    if not body:
        return "Error: reply body is required."

    account_obj, provider, error = await _resolve_provider(tenant_id, account)
    if error:
        return error

    if not hasattr(provider, "reply_email"):
        return "Reply not supported for this email provider."

    try:
        result = await provider.reply_email(
            original_message_id=message_id, body=body, reply_all=reply_all,
        )
        if result.get("success"):
            replied_to = result.get("replied_to", "")
            return f"Reply sent{' to ' + replied_to if replied_to else ''}."
        else:
            return f"Failed to reply: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.error(f"reply_email failed: {e}", exc_info=True)
        return f"Error replying: {e}"


# ============================================================
# delete_emails (needs_approval)
# ============================================================

async def _preview_delete_emails(args: dict, context) -> str:
    """Preview for delete_emails approval."""
    message_ids = args.get("message_ids", [])
    description = args.get("description", f"{len(message_ids)} email(s)")
    permanent = args.get("permanent", False)
    action = "Permanently delete" if permanent else "Delete"
    return f"{action} {description}?"


@tool(needs_approval=True, get_preview=_preview_delete_emails)
async def delete_emails(
    message_ids: Annotated[List[str], "List of message IDs to delete"],
    permanent: Annotated[bool, "Permanently delete instead of trash (default: false)"] = False,
    account: Annotated[str, "Account name (from search_emails results)"] = "primary",
    description: Annotated[str, "Human-readable description for preview (e.g. '3 emails from Amazon')"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Delete emails by message IDs from search_emails results."""
    tenant_id = context.tenant_id

    if not message_ids:
        return "Error: no message_ids provided. Use search_emails first."

    account_obj, provider, error = await _resolve_provider(tenant_id, account)
    if error:
        return error

    try:
        result = await provider.delete_emails(message_ids=message_ids, permanent=permanent)
        if result.get("success"):
            count = result.get("deleted_count", len(message_ids))
            action = "permanently deleted" if permanent else "moved to trash"
            return f"Done! {action.capitalize()} {count} email(s)."
        else:
            return f"Failed to delete: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.error(f"delete_emails failed: {e}", exc_info=True)
        return f"Error deleting emails: {e}"


# ============================================================
# archive_emails (needs_approval)
# ============================================================

async def _preview_archive_emails(args: dict, context) -> str:
    """Preview for archive_emails approval."""
    message_ids = args.get("message_ids", [])
    description = args.get("description", f"{len(message_ids)} email(s)")
    return f"Archive {description}?"


@tool(needs_approval=True, get_preview=_preview_archive_emails)
async def archive_emails(
    message_ids: Annotated[List[str], "List of message IDs to archive"],
    account: Annotated[str, "Account name (from search_emails results)"] = "primary",
    description: Annotated[str, "Human-readable description for preview"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Archive emails by message IDs from search_emails results."""
    tenant_id = context.tenant_id

    if not message_ids:
        return "Error: no message_ids provided. Use search_emails first."

    account_obj, provider, error = await _resolve_provider(tenant_id, account)
    if error:
        return error

    try:
        result = await provider.archive_emails(message_ids=message_ids)
        if result.get("success"):
            count = result.get("archived_count", len(message_ids))
            return f"Done! Archived {count} email(s)."
        else:
            return f"Failed to archive: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.error(f"archive_emails failed: {e}", exc_info=True)
        return f"Error archiving emails: {e}"


# ============================================================
# mark_as_read
# ============================================================

@tool
async def mark_as_read(
    message_ids: Annotated[List[str], "List of message IDs to mark as read"],
    account: Annotated[str, "Account name (from search_emails results)"] = "primary",
    *,
    context: AgentToolContext,
) -> str:
    """Mark emails as read by message IDs."""
    tenant_id = context.tenant_id

    if not message_ids:
        return "Error: no message_ids provided. Use search_emails first."

    account_obj, provider, error = await _resolve_provider(tenant_id, account)
    if error:
        return error

    try:
        result = await provider.mark_as_read(message_ids)
        if result.get("success"):
            return f"Marked {len(message_ids)} email(s) as read."
        else:
            return f"Failed to mark as read: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.error(f"mark_as_read failed: {e}", exc_info=True)
        return f"Error marking emails as read: {e}"

"""
Email tools for EmailAgent domain agent.

Extracted from legacy email agents (SendEmailAgent, ReadEmailAgent,
ReplyEmailAgent, DeleteEmailAgent, ArchiveEmailAgent, MarkReadEmailAgent).
"""
import logging
import html
from typing import Any, Dict, List

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

async def search_emails(args: dict, context) -> str:
    """Search emails across connected accounts."""
    tenant_id = context.tenant_id
    query = args.get("query")
    sender = args.get("sender")
    unread_only = args.get("unread_only", True)
    days_back = args.get("days_back", 7)
    date_range = args.get("date_range")
    accounts = args.get("accounts")
    max_results = args.get("max_results", 15)
    include_categories = args.get("include_categories")

    if isinstance(accounts, str):
        accounts = [accounts]

    providers, errors = await _resolve_all_providers(tenant_id, accounts)
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

async def send_email(args: dict, context) -> str:
    """Send an email."""
    tenant_id = context.tenant_id
    to = args.get("to", "")
    subject = args.get("subject", "Quick note")
    body = args.get("body", "")
    from_account = args.get("from_account", "primary")

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


# ============================================================
# reply_email (needs_approval)
# ============================================================

async def reply_email(args: dict, context) -> str:
    """Reply to an email by message_id."""
    tenant_id = context.tenant_id
    message_id = args.get("message_id", "")
    body = args.get("body", "")
    reply_all = args.get("reply_all", False)
    account_spec = args.get("account", "primary")

    if not message_id:
        return "Error: message_id is required. Use search_emails first to find it."
    if not body:
        return "Error: reply body is required."

    account, provider, error = await _resolve_provider(tenant_id, account_spec)
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


async def _preview_reply_email(args: dict, context) -> str:
    """Preview for reply_email approval."""
    body = args.get("body", "")
    reply_all = args.get("reply_all", False)
    suffix = " (reply all)" if reply_all else ""
    return f"Reply Draft{suffix}:\n\n{body}\n\n---\nSend this reply?"


# ============================================================
# delete_emails (needs_approval)
# ============================================================

async def delete_emails(args: dict, context) -> str:
    """Delete emails by message IDs."""
    tenant_id = context.tenant_id
    message_ids = args.get("message_ids", [])
    permanent = args.get("permanent", False)
    account_spec = args.get("account", "primary")

    if not message_ids:
        return "Error: no message_ids provided. Use search_emails first."

    account, provider, error = await _resolve_provider(tenant_id, account_spec)
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


async def _preview_delete_emails(args: dict, context) -> str:
    """Preview for delete_emails approval."""
    message_ids = args.get("message_ids", [])
    description = args.get("description", f"{len(message_ids)} email(s)")
    permanent = args.get("permanent", False)
    action = "Permanently delete" if permanent else "Delete"
    return f"{action} {description}?"


# ============================================================
# archive_emails (needs_approval)
# ============================================================

async def archive_emails(args: dict, context) -> str:
    """Archive emails by message IDs."""
    tenant_id = context.tenant_id
    message_ids = args.get("message_ids", [])
    account_spec = args.get("account", "primary")

    if not message_ids:
        return "Error: no message_ids provided. Use search_emails first."

    account, provider, error = await _resolve_provider(tenant_id, account_spec)
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


async def _preview_archive_emails(args: dict, context) -> str:
    """Preview for archive_emails approval."""
    message_ids = args.get("message_ids", [])
    description = args.get("description", f"{len(message_ids)} email(s)")
    return f"Archive {description}?"


# ============================================================
# mark_as_read
# ============================================================

async def mark_as_read(args: dict, context) -> str:
    """Mark emails as read by message IDs."""
    tenant_id = context.tenant_id
    message_ids = args.get("message_ids", [])
    account_spec = args.get("account", "primary")

    if not message_ids:
        return "Error: no message_ids provided. Use search_emails first."

    account, provider, error = await _resolve_provider(tenant_id, account_spec)
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

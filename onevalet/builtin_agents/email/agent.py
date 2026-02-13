"""
EmailAgent - Domain agent for all email-related requests.

Replaces SendEmailAgent, ReadEmailAgent, ReplyEmailAgent, DeleteEmailAgent,
ArchiveEmailAgent, MarkReadEmailAgent, and EmailSummaryAgent with a single
agent that has its own mini ReAct loop.
"""
from datetime import datetime

from onevalet import valet
from onevalet.agents.domain_agent import DomainAgent, DomainTool

from .tools import (
    search_emails,
    send_email, _preview_send_email,
    reply_email, _preview_reply_email,
    delete_emails, _preview_delete_emails,
    archive_emails, _preview_archive_emails,
    mark_as_read,
)


@valet(capabilities=["email"])
class EmailDomainAgent(DomainAgent):
    """Read, send, reply, delete, and archive emails. Use when the user mentions email, inbox, messages, or wants to send/check/reply to any email."""

    max_domain_turns = 6

    _SYSTEM_PROMPT_TEMPLATE = """\
You are an email assistant with access to real-time email tools.

Available tools:
- search_emails: Search emails across connected accounts. Returns message_ids needed by other tools.
- send_email: Compose and send a new email (requires approval).
- reply_email: Reply to a specific email by message_id (requires approval).
- delete_emails: Delete emails by message_ids (requires approval).
- archive_emails: Archive emails by message_ids (requires approval).
- mark_as_read: Mark emails as read by message_ids.

Today's date: {today} ({weekday})

Instructions:
1. For reading/checking emails: call search_emails. Default to unread emails from primary inbox.
2. For sending: collect recipient email, subject, and body. Generate a clear, concise email body \
based on the user's intent. Then call send_email.
3. For replying: first call search_emails to find the target email and get its message_id, \
then call reply_email.
4. For deleting/archiving: first call search_emails, then use the message_ids from results \
to call delete_emails or archive_emails. Include a description of what's being deleted/archived.
5. For mark as read: use message_ids from a previous search_emails call.
6. If critical info is missing (like recipient email for sending), ASK the user \
in your text response WITHOUT calling any tools.
7. When composing emails, keep them simple and direct — match the user's tone.
8. Always use message_id and account values from search_emails results when calling other tools."""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime('%Y-%m-%d'),
            weekday=now.strftime('%A'),
        )

    domain_tools = [
        DomainTool(
            name="search_emails",
            description="Search emails across connected accounts. Returns email list with message_ids.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (subject, content)",
                    },
                    "sender": {
                        "type": "string",
                        "description": "Filter by sender name or email",
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "Only show unread emails (default: true)",
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "Days to search back (default: 7)",
                    },
                    "date_range": {
                        "type": "string",
                        "description": "Date range like 'today', 'yesterday', 'last week'",
                    },
                    "accounts": {
                        "type": "string",
                        "description": "Account to search: 'all', 'primary', or account name",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return (default: 15)",
                    },
                },
                "required": [],
            },
            executor=search_emails,
        ),
        DomainTool(
            name="send_email",
            description="Send an email. Requires recipient, subject, and body.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content (plain text)",
                    },
                    "from_account": {
                        "type": "string",
                        "description": "Account to send from (default: 'primary')",
                    },
                },
                "required": ["to", "body"],
            },
            executor=send_email,
            needs_approval=True,
            get_preview=_preview_send_email,
        ),
        DomainTool(
            name="reply_email",
            description="Reply to an email. Use message_id from search_emails results.",
            parameters={
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "Message ID of the email to reply to (from search_emails)",
                    },
                    "body": {
                        "type": "string",
                        "description": "Reply content",
                    },
                    "reply_all": {
                        "type": "boolean",
                        "description": "Reply to all recipients (default: false)",
                    },
                    "account": {
                        "type": "string",
                        "description": "Account name (from search_emails results)",
                    },
                },
                "required": ["message_id", "body"],
            },
            executor=reply_email,
            needs_approval=True,
            get_preview=_preview_reply_email,
        ),
        DomainTool(
            name="delete_emails",
            description="Delete emails by message IDs from search_emails results.",
            parameters={
                "type": "object",
                "properties": {
                    "message_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of message IDs to delete",
                    },
                    "permanent": {
                        "type": "boolean",
                        "description": "Permanently delete instead of trash (default: false)",
                    },
                    "account": {
                        "type": "string",
                        "description": "Account name (from search_emails results)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description for preview (e.g. '3 emails from Amazon')",
                    },
                },
                "required": ["message_ids"],
            },
            executor=delete_emails,
            needs_approval=True,
            get_preview=_preview_delete_emails,
        ),
        DomainTool(
            name="archive_emails",
            description="Archive emails by message IDs from search_emails results.",
            parameters={
                "type": "object",
                "properties": {
                    "message_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of message IDs to archive",
                    },
                    "account": {
                        "type": "string",
                        "description": "Account name (from search_emails results)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description for preview",
                    },
                },
                "required": ["message_ids"],
            },
            executor=archive_emails,
            needs_approval=True,
            get_preview=_preview_archive_emails,
        ),
        DomainTool(
            name="mark_as_read",
            description="Mark emails as read by message IDs.",
            parameters={
                "type": "object",
                "properties": {
                    "message_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of message IDs to mark as read",
                    },
                    "account": {
                        "type": "string",
                        "description": "Account name (from search_emails results)",
                    },
                },
                "required": ["message_ids"],
            },
            executor=mark_as_read,
        ),
    ]

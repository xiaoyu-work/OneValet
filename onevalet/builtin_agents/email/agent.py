"""
EmailAgent - Domain agent for all email-related requests.

Replaces SendEmailAgent, ReadEmailAgent, ReplyEmailAgent, DeleteEmailAgent,
ArchiveEmailAgent, MarkReadEmailAgent, and EmailSummaryAgent with a single
agent that has its own mini ReAct loop.
"""
from datetime import datetime

from onevalet import valet
from onevalet.standard_agent import StandardAgent, AgentTool

from .tools import (
    search_emails,
    send_email, _preview_send_email,
    reply_email, _preview_reply_email,
    delete_emails, _preview_delete_emails,
    archive_emails, _preview_archive_emails,
    mark_as_read,
)


@valet(capabilities=["email"])
class EmailAgent(StandardAgent):
    """Read, send, reply, delete, and archive emails. Use when the user mentions email, inbox, messages, or wants to send/check/reply to any email."""

    max_domain_turns = 6

    _SYSTEM_PROMPT_TEMPLATE = """\
Email management tools are available for this task. Today is {today} ({weekday}).

Tool reference:
- search_emails: Find emails, returns message_ids for use with other tools.
- send_email: Send a new email (approval required). Needs: to, subject, body.
- reply_email: Reply by message_id (approval required). Needs: message_id, body.
- delete_emails: Delete by message_ids (approval required).
- archive_emails: Archive by message_ids (approval required).
- mark_as_read: Mark as read by message_ids.

Guidelines:
1. Reading emails: call search_emails. Default to unread from primary inbox.
2. Sending: need recipient email, subject, and body. If any is missing, ask in one sentence.
3. Replying: search_emails first to get message_id, then reply_email.
4. Deleting/archiving: search_emails first, then use message_ids.
5. If only a name is given, search_emails for their address. If not found, ask the user.
6. Only write what the user asked. Do not guess email content from prior context.
7. Always use message_id and account from search_emails results.
8. Be concise."""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime('%Y-%m-%d'),
            weekday=now.strftime('%A'),
        )

    domain_tools = [
        AgentTool(
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
                    "category": {
                        "type": "string",
                        "description": "Inbox category filter: 'primary' (default), 'social', 'promotions', 'updates', or 'all'",
                    },
                },
                "required": [],
            },
            executor=search_emails,
        ),
        AgentTool(
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
        AgentTool(
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
        AgentTool(
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
        AgentTool(
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
        AgentTool(
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

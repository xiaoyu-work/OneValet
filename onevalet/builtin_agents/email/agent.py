"""
EmailAgent - Agent for all email-related requests.

Replaces SendEmailAgent, ReadEmailAgent, ReplyEmailAgent, DeleteEmailAgent,
ArchiveEmailAgent, MarkReadEmailAgent with a single agent that has its own
mini ReAct loop.
"""
from datetime import datetime

from onevalet import valet
from onevalet.constants import EMAIL_SERVICES
from onevalet.standard_agent import StandardAgent

from .tools import (
    search_emails,
    send_email,
    reply_email,
    delete_emails,
    archive_emails,
    mark_as_read,
)


@valet(domain="communication", requires_service=list(EMAIL_SERVICES))
class EmailAgent(StandardAgent):
    """Read, send, reply, delete, and archive emails. Use when the user mentions email, inbox, messages, or wants to send/check/reply to any email."""

    max_turns = 6

    _SYSTEM_PROMPT_TEMPLATE = """\
Email management tools are available for this task. Today is {today} ({weekday}).

Tool reference:
- search_emails: Find emails, returns message_ids for use with other tools.
- send_email: Send a new email. Needs: to, subject, body.
- reply_email: Reply by message_id. Needs: message_id, body.
- delete_emails: Delete by message_ids.
- archive_emails: Archive by message_ids.
- mark_as_read: Mark as read by message_ids.

Guidelines:
1. Reading emails: call search_emails. Default to unread from primary inbox.
2. Sending: once you have recipient, subject, and body, call send_email directly. If any is missing, ask in one sentence.
3. Replying: search_emails first to get message_id, then call reply_email directly.
4. Deleting/archiving: search_emails first, then call delete_emails or archive_emails directly with the message_ids.
5. If only a name is given, search_emails for their address. If not found, ask the user.
6. Only write what the user asked. Do not guess email content from prior context.
7. Always use message_id and account from search_emails results.
8. Be concise.
9. Do NOT ask for confirmation before calling a tool. Call the tool directly once you have enough information."""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime('%Y-%m-%d'),
            weekday=now.strftime('%A'),
        )

    tools = (search_emails, send_email, reply_email, delete_emails, archive_emails, mark_as_read)

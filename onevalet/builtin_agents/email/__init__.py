"""
Email agents for OneValet

Provides agents for reading, sending, deleting, replying, archiving,
and managing email preferences.
"""

from .read import ReadEmailAgent
from .send import SendEmailAgent
from .delete import DeleteEmailAgent
from .reply import ReplyEmailAgent
from .archive import ArchiveEmailAgent
from .mark_read import MarkReadEmailAgent
from .importance import EmailImportanceAgent
from .preference import EmailPreferenceAgent
from .summary import EmailSummaryAgent

__all__ = [
    "ReadEmailAgent",
    "SendEmailAgent",
    "DeleteEmailAgent",
    "ReplyEmailAgent",
    "ArchiveEmailAgent",
    "MarkReadEmailAgent",
    "EmailImportanceAgent",
    "EmailPreferenceAgent",
    "EmailSummaryAgent",
]

"""
Email agents for OneValet

Provides a domain agent for managing email (read, send, reply, delete, archive)
and agents for email importance evaluation and preference management.
"""

from .agent import EmailDomainAgent
from .importance import EmailImportanceAgent
from .preference import EmailPreferenceAgent

__all__ = [
    "EmailDomainAgent",
    "EmailImportanceAgent",
    "EmailPreferenceAgent",
]

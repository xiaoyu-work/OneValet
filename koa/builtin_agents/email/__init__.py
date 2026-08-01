"""
Email agents for Koa

Provides an agent for managing email (read, send, reply, delete, archive)
and an agent for managing email importance rules and notification preferences.
"""

from .agent import EmailAgent
from .importance_rules import SYSTEM_RULES
from .preference import EmailPreferenceAgent

__all__ = [
    "EmailAgent",
    "EmailPreferenceAgent",
    "SYSTEM_RULES",
]

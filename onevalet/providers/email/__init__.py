"""
Email providers - Gmail, Outlook, etc.
"""

from .base import BaseEmailProvider
from .factory import EmailProviderFactory
from .resolver import AccountResolver

__all__ = ["BaseEmailProvider", "EmailProviderFactory", "AccountResolver"]

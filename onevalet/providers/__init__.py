"""
OneValet Providers - Email, Calendar, SMS, and Shipment providers

Providers receive credentials dicts directly. They do not query any database.
The calling code (agent or orchestrator) is responsible for fetching credentials
from CredentialStore and passing them to the provider.
"""

from .email.factory import EmailProviderFactory
from .calendar.factory import CalendarProviderFactory
from .sms.factory import SMSProviderFactory
from .email.resolver import AccountResolver

__all__ = [
    "EmailProviderFactory",
    "CalendarProviderFactory",
    "SMSProviderFactory",
    "AccountResolver",
]

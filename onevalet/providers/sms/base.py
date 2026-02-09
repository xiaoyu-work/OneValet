"""
Base SMS Provider - Abstract base class for all SMS providers
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseSMSProvider(ABC):
    """
    Abstract base class for SMS providers.

    All SMS providers must implement:
    - send_sms(to, body) - Send SMS message
    - is_enabled() - Check if provider is configured and ready
    """

    def __init__(self):
        self.provider_name = self.__class__.__name__

    @abstractmethod
    async def send_sms(self, to: str, body: str) -> bool:
        """
        Send SMS message.

        Args:
            to: Recipient phone number (E.164 format, e.g., +1234567890)
            body: Message content

        Returns:
            True if sent successfully, False otherwise
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if provider is configured and enabled.

        Returns:
            True if provider can send SMS, False otherwise
        """
        pass

    def normalize_phone_number(self, phone: str) -> str:
        """Normalize phone number to E.164 format."""
        if not phone.startswith("+"):
            return f"+{phone}"
        return phone

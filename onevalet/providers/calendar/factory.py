"""
Calendar Provider Factory - Creates appropriate calendar provider instances

Registry pattern for managing multiple calendar providers (Google Calendar, Outlook, etc.)
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .base import BaseCalendarProvider

logger = logging.getLogger(__name__)


class CalendarProviderFactory:
    """Factory for creating calendar provider instances."""

    _providers: Dict[str, type] = {}

    @classmethod
    def register_provider(cls, provider_name: str, provider_class: type):
        """
        Register a calendar provider implementation.

        Args:
            provider_name: Provider identifier (e.g., "google", "microsoft")
            provider_class: Provider class (must inherit from BaseCalendarProvider)
        """
        if not issubclass(provider_class, BaseCalendarProvider):
            raise ValueError("Provider class must inherit from BaseCalendarProvider")

        cls._providers[provider_name.lower()] = provider_class
        logger.info(f"Registered calendar provider: {provider_name}")

    @classmethod
    def create_provider(
        cls,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ) -> Optional[BaseCalendarProvider]:
        """
        Create a calendar provider instance from credentials dict.

        Args:
            credentials: Credentials dict with 'provider' field
            on_token_refreshed: Optional callback for token refresh persistence

        Returns:
            Provider instance or None if provider not supported
        """
        provider_name = credentials.get("provider", "").lower()

        if provider_name not in cls._providers:
            logger.error(f"Unsupported calendar provider: {provider_name}")
            logger.info(f"Supported providers: {list(cls._providers.keys())}")
            return None

        provider_class = cls._providers[provider_name]
        logger.info(f"Creating {provider_class.__name__} for account {credentials.get('account_name')}")
        return provider_class(credentials, on_token_refreshed)

    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of registered provider names."""
        return list(cls._providers.keys())


def _register_providers():
    """Auto-register available calendar providers."""
    from .google import GoogleCalendarProvider
    CalendarProviderFactory.register_provider("google", GoogleCalendarProvider)


_register_providers()

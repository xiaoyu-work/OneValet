"""
Email Provider Factory - Create appropriate email provider instances

Factory pattern for instantiating the correct provider class based on
the credentials dict's provider field (gmail, outlook, etc.)
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .base import BaseEmailProvider

logger = logging.getLogger(__name__)


class EmailProviderFactory:
    """Factory for creating email provider instances."""

    _providers: Dict[str, type] = {}

    @classmethod
    def register_provider(cls, provider_name: str, provider_class: type):
        """
        Register a provider implementation.

        Args:
            provider_name: Provider identifier (e.g., "gmail", "outlook")
            provider_class: Provider class (must inherit from BaseEmailProvider)
        """
        if not issubclass(provider_class, BaseEmailProvider):
            raise TypeError(f"{provider_class} must inherit from BaseEmailProvider")

        cls._providers[provider_name.lower()] = provider_class
        logger.info(f"Registered email provider: {provider_name}")

    @classmethod
    def create_provider(
        cls,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ) -> Optional[BaseEmailProvider]:
        """
        Create appropriate provider instance from credentials dict.

        Args:
            credentials: Credentials dict (must have 'provider' field)
            on_token_refreshed: Optional callback for token refresh persistence

        Returns:
            Provider instance or None if provider not supported
        """
        provider_name = credentials.get("provider", "").lower()

        if not provider_name:
            logger.error("Credentials missing 'provider' field")
            return None

        provider_class = cls._providers.get(provider_name)
        if not provider_class:
            logger.error(f"Unsupported email provider: {provider_name}")
            logger.info(f"Available providers: {list(cls._providers.keys())}")
            return None

        try:
            provider = provider_class(credentials, on_token_refreshed)
            logger.info(f"Created {provider_name} provider for {credentials.get('account_name')}")
            return provider
        except Exception as e:
            logger.error(f"Failed to create {provider_name} provider: {e}", exc_info=True)
            return None

    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of supported provider names."""
        return list(cls._providers.keys())


def _register_providers():
    """Auto-register all available email providers."""
    try:
        from .gmail import GmailProvider
        EmailProviderFactory.register_provider("google", GmailProvider)
    except ImportError as e:
        logger.warning(f"Google provider not available: {e}")

    try:
        from .outlook import OutlookProvider
        EmailProviderFactory.register_provider("outlook", OutlookProvider)
        EmailProviderFactory.register_provider("microsoft", OutlookProvider)
    except ImportError as e:
        logger.warning(f"Outlook provider not available: {e}")


_register_providers()

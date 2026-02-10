"""
Image Provider Factory - Create appropriate image provider instances

Factory pattern for instantiating the correct provider class based on
the credentials dict's provider field (openai, azure, gemini, seedream, etc.)
"""

import logging
from typing import Dict, List, Optional

from .base import BaseImageProvider

logger = logging.getLogger(__name__)


class ImageProviderFactory:
    """Factory for creating image provider instances."""

    _providers: Dict[str, type] = {}

    @classmethod
    def register_provider(cls, provider_name: str, provider_class: type):
        """
        Register a provider implementation.

        Args:
            provider_name: Provider identifier (e.g., "openai", "azure", "gemini", "seedream")
            provider_class: Provider class (must inherit from BaseImageProvider)
        """
        if not issubclass(provider_class, BaseImageProvider):
            raise TypeError(f"{provider_class} must inherit from BaseImageProvider")

        cls._providers[provider_name.lower()] = provider_class
        logger.info(f"Registered image provider: {provider_name}")

    @classmethod
    def create_provider(cls, credentials: dict) -> Optional[BaseImageProvider]:
        """
        Create appropriate provider instance from credentials dict.

        Args:
            credentials: Credentials dict (must have 'provider' field)

        Returns:
            Provider instance or None if provider not supported
        """
        provider_name = credentials.get("provider", "").lower()

        if not provider_name:
            logger.error("Credentials missing 'provider' field")
            return None

        provider_class = cls._providers.get(provider_name)
        if not provider_class:
            logger.error(f"Unsupported image provider: {provider_name}")
            logger.info(f"Available providers: {list(cls._providers.keys())}")
            return None

        try:
            provider = provider_class(credentials)
            logger.info(f"Created {provider_name} image provider")
            return provider
        except Exception as e:
            logger.error(f"Failed to create {provider_name} image provider: {e}", exc_info=True)
            return None

    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of supported provider names."""
        return list(cls._providers.keys())


def _register_providers():
    """Auto-register all available image providers."""
    try:
        from .openai_image import OpenAIImageProvider
        ImageProviderFactory.register_provider("openai", OpenAIImageProvider)
    except ImportError as e:
        logger.warning(f"OpenAI image provider not available: {e}")

    try:
        from .azure_image import AzureImageProvider
        ImageProviderFactory.register_provider("azure", AzureImageProvider)
    except ImportError as e:
        logger.warning(f"Azure image provider not available: {e}")

    try:
        from .gemini_image import GeminiImageProvider
        ImageProviderFactory.register_provider("gemini", GeminiImageProvider)
    except ImportError as e:
        logger.warning(f"Gemini image provider not available: {e}")

    try:
        from .seedream import SeedreamProvider
        ImageProviderFactory.register_provider("seedream", SeedreamProvider)
    except ImportError as e:
        logger.warning(f"Seedream image provider not available: {e}")


_register_providers()

"""
Cloud Storage Provider Factory - Create appropriate provider instances
"""

import logging
from typing import Callable, Dict, List, Optional

from .base import BaseCloudStorageProvider

logger = logging.getLogger(__name__)


class CloudStorageProviderFactory:
    """Factory for creating cloud storage provider instances."""

    _providers: Dict[str, type] = {}

    @classmethod
    def register_provider(cls, provider_name: str, provider_class: type):
        if not issubclass(provider_class, BaseCloudStorageProvider):
            raise TypeError(f"{provider_class} must inherit from BaseCloudStorageProvider")
        cls._providers[provider_name.lower()] = provider_class
        logger.info(f"Registered cloud storage provider: {provider_name}")

    @classmethod
    def create_provider(
        cls,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ) -> Optional[BaseCloudStorageProvider]:
        provider_name = credentials.get("provider", "").lower()
        if not provider_name:
            logger.error("Credentials missing 'provider' field")
            return None

        provider_class = cls._providers.get(provider_name)
        if not provider_class:
            logger.error(f"Unsupported cloud storage provider: {provider_name}")
            return None

        try:
            return provider_class(credentials, on_token_refreshed)
        except Exception as e:
            logger.error(f"Failed to create {provider_name} provider: {e}", exc_info=True)
            return None

    @classmethod
    def get_supported_providers(cls) -> List[str]:
        return list(cls._providers.keys())


def _register_providers():
    """Auto-register all available cloud storage providers."""
    try:
        from .google_drive import GoogleDriveProvider
        CloudStorageProviderFactory.register_provider("google", GoogleDriveProvider)
    except ImportError as e:
        logger.warning(f"Google Drive provider not available: {e}")

    try:
        from .onedrive import OneDriveProvider
        CloudStorageProviderFactory.register_provider("onedrive", OneDriveProvider)
    except ImportError as e:
        logger.warning(f"OneDrive provider not available: {e}")

    try:
        from .dropbox_storage import DropboxProvider
        CloudStorageProviderFactory.register_provider("dropbox", DropboxProvider)
    except ImportError as e:
        logger.warning(f"Dropbox provider not available: {e}")

    try:
        from .supabase_storage import SupabaseStorageProvider
        CloudStorageProviderFactory.register_provider("supabase", SupabaseStorageProvider)
    except ImportError as e:
        logger.warning(f"Supabase storage provider not available: {e}")


_register_providers()

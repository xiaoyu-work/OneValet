"""
Image Provider Resolver - Find and select configured image providers.

Unlike todo/email resolvers that search OAuth accounts, this resolver
searches for API-key based image provider configurations.

Services stored in CredentialStore:
    - image_openai
    - image_azure
    - image_gemini
    - image_seedream
"""

import logging
from typing import Dict, List, Optional

from onevalet.constants import IMAGE_SERVICES
from onevalet.providers.email.resolver import AccountResolver

logger = logging.getLogger(__name__)

_IMAGE_SERVICES = IMAGE_SERVICES

# Map service names to provider names used by the factory
_SERVICE_TO_PROVIDER = {
    "image_openai": "openai",
    "image_azure": "azure",
    "image_gemini": "gemini",
    "image_seedream": "seedream",
}


class ImageProviderResolver:
    """
    Resolve image provider configurations from CredentialStore.

    Usage:
        # Get all configured providers
        providers = await ImageProviderResolver.resolve_all(tenant_id)

        # Get a specific provider
        provider = await ImageProviderResolver.resolve(tenant_id, "openai")

        # Get the default/best provider
        provider = await ImageProviderResolver.resolve_default(tenant_id)
    """

    @staticmethod
    async def resolve(
        tenant_id: str,
        provider_spec: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Resolve a single image provider configuration.

        Args:
            tenant_id: Tenant/user ID
            provider_spec: Provider name (e.g., "openai", "azure", "gemini", "seedream")
                          If None, returns the first available provider.

        Returns:
            Credentials dict with 'provider' field set, or None if not found.
        """
        resolver = AccountResolver()
        if not resolver.credential_store:
            logger.error("No credential store available")
            return None

        if provider_spec:
            # Look for specific provider
            service = f"image_{provider_spec.lower()}"
            if service not in _IMAGE_SERVICES:
                logger.warning(f"Unknown image service: {service}")
                return None

            creds = await resolver.credential_store.get(tenant_id, service, "primary")
            if creds:
                creds["provider"] = _SERVICE_TO_PROVIDER.get(service, provider_spec)
                creds["service"] = service
                return creds
            return None

        # No spec — return first available
        return await ImageProviderResolver.resolve_default(tenant_id)

    @staticmethod
    async def resolve_default(tenant_id: str) -> Optional[dict]:
        """
        Get the default image provider (first configured one).

        Returns:
            Credentials dict or None if no providers configured.
        """
        resolver = AccountResolver()
        if not resolver.credential_store:
            logger.error("No credential store available")
            return None

        for service in _IMAGE_SERVICES:
            creds = await resolver.credential_store.get(tenant_id, service, "primary")
            if creds and creds.get("api_key"):
                creds["provider"] = _SERVICE_TO_PROVIDER.get(service, "")
                creds["service"] = service
                logger.info(f"Default image provider: {service}")
                return creds

        return None

    @staticmethod
    async def resolve_all(tenant_id: str) -> List[dict]:
        """
        Get all configured image providers.

        Returns:
            List of credentials dicts, each with 'provider' field set.
        """
        resolver = AccountResolver()
        if not resolver.credential_store:
            logger.error("No credential store available")
            return []

        providers = []
        for service in _IMAGE_SERVICES:
            creds = await resolver.credential_store.get(tenant_id, service, "primary")
            if creds and creds.get("api_key"):
                creds["provider"] = _SERVICE_TO_PROVIDER.get(service, "")
                creds["service"] = service
                providers.append(creds)

        return providers

    @staticmethod
    async def get_provider_names(tenant_id: str) -> List[str]:
        """
        Get list of configured provider names for display.

        Returns:
            List of provider names like ["OpenAI", "Gemini"]
        """
        providers = await ImageProviderResolver.resolve_all(tenant_id)
        names = []
        for p in providers:
            name = p.get("provider", "").replace("_", " ").title()
            if name:
                names.append(name)
        return names

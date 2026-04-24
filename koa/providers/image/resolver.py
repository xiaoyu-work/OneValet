"""
Image Provider Resolver - Find and select configured image providers.

Resolution precedence (for each lookup):
    1. Per-tenant CredentialStore entries (BYOK) — preserves multi-tenant flexibility.
    2. Global operator-provided config (``image:`` section in ``config.yaml``) —
       registered at app init via ``set_global_config()``. Used for single-tenant
       self-hosted deployments where the operator brings one set of API keys for
       all users of the app.

Services stored in CredentialStore:
    - image_openai
    - image_azure
    - image_gemini
    - image_seedream
"""

import copy
import logging
from typing import Any, Dict, List, Optional

from koa.constants import IMAGE_SERVICES
from koa.providers.email.resolver import AccountResolver

logger = logging.getLogger(__name__)

_IMAGE_SERVICES = IMAGE_SERVICES

# Map service names to provider names used by the factory
_SERVICE_TO_PROVIDER = {
    "image_openai": "openai",
    "image_azure": "azure",
    "image_gemini": "gemini",
    "image_seedream": "seedream",
}

_PROVIDER_TO_SERVICE = {v: k for k, v in _SERVICE_TO_PROVIDER.items()}


class ImageProviderResolver:
    """
    Resolve image provider configurations.

    Precedence:
        tenant CredentialStore (BYOK) → global config (operator-provided).

    Usage:
        # Operator-provided config registered once at app startup:
        ImageProviderResolver.set_global_config({
            "provider": "azure",
            "api_key": "...",
            "endpoint": "https://x.openai.azure.com",
            "deployment": "gpt-image-2",
            "api_version": "2024-02-01",
        })

        # Per-request:
        providers = await ImageProviderResolver.resolve_all(tenant_id)
        provider  = await ImageProviderResolver.resolve(tenant_id, "azure")
        default   = await ImageProviderResolver.resolve_default(tenant_id)
    """

    # Operator-provided global fallback config. Set once at app init via
    # set_global_config(). Shape matches the credentials dict consumed by
    # ImageProviderFactory, plus a required "provider" field.
    _global_config: Optional[Dict[str, Any]] = None

    @classmethod
    def set_global_config(cls, config: Optional[Dict[str, Any]]) -> None:
        """Register (or clear) the operator-provided global image config.

        Called unconditionally from app init so that reloading the app with
        ``image:`` removed correctly clears stale state.
        """
        if not config:
            cls._global_config = None
            return

        provider = (config.get("provider") or "").lower()
        if provider not in _PROVIDER_TO_SERVICE:
            logger.warning(
                f"[ImageProviderResolver] ignoring global config — unknown provider "
                f"{provider!r} (expected one of {sorted(_PROVIDER_TO_SERVICE)})"
            )
            cls._global_config = None
            return
        if not config.get("api_key"):
            logger.warning("[ImageProviderResolver] ignoring global config — missing api_key")
            cls._global_config = None
            return

        # Store a normalized copy (never mutate caller's dict).
        normalized: Dict[str, Any] = {k: v for k, v in config.items() if v is not None}
        normalized["provider"] = provider
        normalized["service"] = _PROVIDER_TO_SERVICE[provider]
        cls._global_config = normalized
        logger.info(f"[ImageProviderResolver] global config registered: provider={provider}")

    @classmethod
    def _global_credentials(cls, provider_spec: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return a fresh copy of the global credentials dict if it matches provider_spec."""
        if not cls._global_config:
            return None
        if provider_spec and provider_spec.lower() != cls._global_config["provider"]:
            return None
        return copy.deepcopy(cls._global_config)

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
        credential_store = resolver.credential_store

        if provider_spec:
            service = f"image_{provider_spec.lower()}"
            if service not in _IMAGE_SERVICES:
                logger.warning(f"Unknown image service: {service}")
                return None

            # 1. Tenant credential store (BYOK)
            if credential_store:
                creds = await credential_store.get(tenant_id, service, "primary")
                if creds and creds.get("api_key"):
                    creds = dict(creds)
                    creds["provider"] = _SERVICE_TO_PROVIDER.get(service, provider_spec)
                    creds["service"] = service
                    return creds

            # 2. Global fallback
            return ImageProviderResolver._global_credentials(provider_spec)

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
        credential_store = resolver.credential_store

        if credential_store:
            for service in _IMAGE_SERVICES:
                creds = await credential_store.get(tenant_id, service, "primary")
                if creds and creds.get("api_key"):
                    creds = dict(creds)
                    creds["provider"] = _SERVICE_TO_PROVIDER.get(service, "")
                    creds["service"] = service
                    logger.info(f"Default image provider (tenant): {service}")
                    return creds

        # Global fallback
        global_creds = ImageProviderResolver._global_credentials()
        if global_creds:
            logger.info(f"Default image provider (global): {global_creds.get('service')}")
        return global_creds

    @staticmethod
    async def resolve_all(tenant_id: str) -> List[dict]:
        """
        Get all configured image providers.

        Merges tenant-configured providers with the global provider (if any
        and not already overridden by the same tenant service).

        Returns:
            List of credentials dicts, each with 'provider' field set.
        """
        resolver = AccountResolver()
        credential_store = resolver.credential_store

        providers: List[dict] = []
        seen_services: set = set()

        if credential_store:
            for service in _IMAGE_SERVICES:
                creds = await credential_store.get(tenant_id, service, "primary")
                if creds and creds.get("api_key"):
                    creds = dict(creds)
                    creds["provider"] = _SERVICE_TO_PROVIDER.get(service, "")
                    creds["service"] = service
                    providers.append(creds)
                    seen_services.add(service)

        global_creds = ImageProviderResolver._global_credentials()
        if global_creds and global_creds.get("service") not in seen_services:
            providers.append(global_creds)

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

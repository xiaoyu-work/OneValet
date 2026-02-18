"""
Cloud Storage Resolver - Find configured cloud storage providers.

Services in CredentialStore:
    - google_drive (shares tokens with gmail/google_calendar via Google OAuth)
    - onedrive (shares tokens with outlook/outlook_calendar via Microsoft OAuth)
    - dropbox (separate OAuth flow)
"""

import logging
from typing import Dict, List, Optional

from onevalet.constants import STORAGE_SERVICES
from onevalet.providers.email.resolver import AccountResolver

logger = logging.getLogger(__name__)

_STORAGE_SERVICES = STORAGE_SERVICES

_SERVICE_TO_PROVIDER = {
    "google_drive": "google",
    "onedrive": "onedrive",
    "dropbox": "dropbox",
}


class CloudStorageResolver:
    """
    Resolve cloud storage provider configurations.

    Usage:
        accounts = await CloudStorageResolver.resolve_all(tenant_id)
        account = await CloudStorageResolver.resolve(tenant_id, "dropbox")
    """

    @staticmethod
    async def resolve(
        tenant_id: str,
        provider_spec: Optional[str] = None,
    ) -> Optional[dict]:
        """Resolve a single cloud storage provider."""
        resolver = AccountResolver()
        if not resolver.credential_store:
            logger.error("No credential store available")
            return None

        if provider_spec:
            # Map provider name to service
            service = None
            for svc, prov in _SERVICE_TO_PROVIDER.items():
                if prov == provider_spec.lower() or svc == provider_spec.lower():
                    service = svc
                    break

            if not service:
                return None

            creds = await resolver.credential_store.get(tenant_id, service, "primary")
            if creds:
                creds["provider"] = _SERVICE_TO_PROVIDER.get(service, "")
                creds["service"] = service
                return creds
            return None

        # No spec — return first available
        return await CloudStorageResolver.resolve_default(tenant_id)

    @staticmethod
    async def resolve_default(tenant_id: str) -> Optional[dict]:
        """Get the default (first configured) cloud storage provider."""
        resolver = AccountResolver()
        if not resolver.credential_store:
            return None

        for service in _STORAGE_SERVICES:
            creds = await resolver.credential_store.get(tenant_id, service, "primary")
            if creds and creds.get("access_token"):
                creds["provider"] = _SERVICE_TO_PROVIDER.get(service, "")
                creds["service"] = service
                return creds

        return None

    @staticmethod
    async def resolve_all(tenant_id: str) -> List[dict]:
        """Get all configured cloud storage providers."""
        resolver = AccountResolver()
        if not resolver.credential_store:
            return []

        providers = []
        for service in _STORAGE_SERVICES:
            creds = await resolver.credential_store.get(tenant_id, service, "primary")
            if creds and creds.get("access_token"):
                creds["provider"] = _SERVICE_TO_PROVIDER.get(service, "")
                creds["service"] = service
                providers.append(creds)

        return providers

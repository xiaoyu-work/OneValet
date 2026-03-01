"""
Supabase Storage Provider - Supabase Storage API implementation.

Uses Supabase's S3-compatible storage service for file uploads and management.
Unlike OAuth-based providers (Google Drive, OneDrive), this uses a service role
key for authentication, making it simpler to set up.

Requires:
    - SUPABASE_URL: Your Supabase project URL
    - SUPABASE_SERVICE_ROLE_KEY: Service role key (not anon key)
    - A storage bucket created in Supabase Dashboard
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .base import BaseCloudStorageProvider

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "onevalet-files"
DEFAULT_SIGNED_URL_EXPIRY = 3600  # 1 hour


class SupabaseStorageProvider(BaseCloudStorageProvider):
    """
    Supabase Storage provider.

    Files are organized as: {bucket}/{tenant_id}/{folder_path}/{filename}
    This provides natural multi-tenant isolation at the storage level.

    Credentials dict expects:
        - provider: "supabase"
        - supabase_url: Project URL
        - supabase_key: Service role key
        - bucket: (optional) Bucket name, defaults to "onevalet-files"
        - tenant_id: (optional) Used as path prefix for tenant isolation
    """

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)
        self._supabase_url = credentials.get("supabase_url", "")
        self._supabase_key = credentials.get("supabase_key", "")
        self._bucket = credentials.get("bucket", DEFAULT_BUCKET)
        self._tenant_id = credentials.get("tenant_id", "")
        self._client = None

    def for_tenant(self, tenant_id: str) -> "SupabaseStorageProvider":
        """Return a lightweight copy scoped to a specific tenant."""
        import copy
        clone = copy.copy(self)
        clone._tenant_id = tenant_id
        return clone

    def _get_client(self):
        """Lazy-initialize the Supabase client."""
        if self._client is None:
            from supabase import create_client
            self._client = create_client(self._supabase_url, self._supabase_key)
        return self._client

    @property
    def _storage(self):
        """Get the storage bucket handle."""
        return self._get_client().storage.from_(self._bucket)

    def _full_path(self, path: str) -> str:
        """Prefix path with tenant_id for isolation."""
        if self._tenant_id:
            return f"{self._tenant_id}/{path.lstrip('/')}"
        return path.lstrip("/")

    async def upload_file(
        self,
        file_name: str,
        file_data: bytes,
        mime_type: str = "image/jpeg",
        folder_path: str = "",
    ) -> Dict[str, Any]:
        try:
            if folder_path:
                path = f"{folder_path.strip('/')}/{file_name}"
            else:
                path = file_name
            full_path = self._full_path(path)

            self._storage.upload(
                path=full_path,
                file=file_data,
                file_options={
                    "content-type": mime_type,
                    "upsert": "true",
                },
            )

            # Generate a signed URL (private buckets don't support public URLs)
            signed = self._storage.create_signed_url(
                full_path, DEFAULT_SIGNED_URL_EXPIRY
            )
            url = signed.get("signedURL", "") if isinstance(signed, dict) else str(signed)

            return {
                "success": True,
                "data": {
                    "id": full_path,
                    "url": url,
                    "name": file_name,
                },
            }
        except Exception as e:
            logger.error(f"Supabase upload failed: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    async def get_download_link(self, file_id: str) -> Dict[str, Any]:
        try:
            result = self._storage.create_signed_url(
                file_id, DEFAULT_SIGNED_URL_EXPIRY
            )
            url = result.get("signedURL", "") if isinstance(result, dict) else str(result)
            return {
                "success": True,
                "data": {"url": url, "expires": f"{DEFAULT_SIGNED_URL_EXPIRY}s"},
            }
        except Exception as e:
            logger.error(f"Supabase signed URL failed: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    async def search_files(
        self,
        query: str,
        file_type: Optional[str] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        try:
            # Supabase Storage list supports search parameter
            result = self._storage.list(
                path=self._tenant_id or "",
                options={"search": query, "limit": max_results},
            )
            files = [
                {
                    "id": f"{self._tenant_id}/{item['name']}" if self._tenant_id else item["name"],
                    "name": item.get("name", ""),
                    "type": item.get("metadata", {}).get("mimetype", ""),
                    "modified": item.get("updated_at", ""),
                    "size": item.get("metadata", {}).get("size", 0),
                    "path": item.get("name", ""),
                    "url": "",
                }
                for item in result
                if isinstance(item, dict)
            ]
            return {"success": True, "data": files[:max_results]}
        except Exception as e:
            logger.error(f"Supabase search failed: {e}", exc_info=True)
            return {"success": False, "data": [], "error": str(e)}

    async def list_recent_files(self, max_results: int = 10) -> Dict[str, Any]:
        try:
            result = self._storage.list(
                path=self._tenant_id or "",
                options={
                    "limit": max_results,
                    "sortBy": {"column": "updated_at", "order": "desc"},
                },
            )
            files = [
                {
                    "id": f"{self._tenant_id}/{item['name']}" if self._tenant_id else item["name"],
                    "name": item.get("name", ""),
                    "type": item.get("metadata", {}).get("mimetype", ""),
                    "modified": item.get("updated_at", ""),
                    "size": item.get("metadata", {}).get("size", 0),
                    "path": item.get("name", ""),
                    "url": "",
                }
                for item in result
                if isinstance(item, dict)
            ]
            return {"success": True, "data": files}
        except Exception as e:
            logger.error(f"Supabase list failed: {e}", exc_info=True)
            return {"success": False, "data": [], "error": str(e)}

    async def get_file_info(self, file_id: str) -> Dict[str, Any]:
        try:
            # List the parent folder and find the file
            parts = file_id.rsplit("/", 1)
            folder = parts[0] if len(parts) > 1 else ""
            filename = parts[-1]

            result = self._storage.list(
                path=folder,
                options={"search": filename, "limit": 1},
            )
            if result and isinstance(result[0], dict):
                item = result[0]
                return {
                    "success": True,
                    "data": {
                        "id": file_id,
                        "name": item.get("name", ""),
                        "type": item.get("metadata", {}).get("mimetype", ""),
                        "modified": item.get("updated_at", ""),
                        "size": item.get("metadata", {}).get("size", 0),
                        "path": file_id,
                        "url": "",  # use get_download_link() for access
                        "shared": False,
                    },
                }
            return {"success": False, "data": None, "error": "File not found"}
        except Exception as e:
            logger.error(f"Supabase file info failed: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    async def share_file(
        self,
        file_id: str,
        email: Optional[str] = None,
        link_type: str = "view",
    ) -> Dict[str, Any]:
        try:
            # Supabase doesn't support email-based sharing.
            # Return a signed URL instead.
            result = self._storage.create_signed_url(
                file_id, DEFAULT_SIGNED_URL_EXPIRY
            )
            url = result.get("signedURL", "") if isinstance(result, dict) else str(result)
            return {
                "success": True,
                "data": {"url": url, "type": "signed_url"},
            }
        except Exception as e:
            logger.error(f"Supabase share failed: {e}", exc_info=True)
            return {"success": False, "data": None, "error": str(e)}

    async def get_storage_usage(self) -> Dict[str, Any]:
        # Supabase doesn't expose a direct storage quota API.
        return {
            "success": True,
            "data": {"used": 0, "total": 0, "percent": 0.0},
        }

    async def refresh_access_token(self) -> Dict[str, Any]:
        # Supabase uses a service role key — no OAuth token refresh needed.
        return {"success": True, "access_token": self._supabase_key}

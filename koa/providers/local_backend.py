from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx

from koa.models import AgentToolContext


def _get_backend_url() -> str:
    """Resolve the koi-backend URL from config callbacks.notify_url or env."""
    # 1. Try KOIAI_BACKEND_URL env var (explicit override)
    url = os.getenv("KOIAI_BACKEND_URL", "")
    if url:
        return url.rstrip("/")

    # 2. Try callbacks.notify_url from loaded config (strip path to get base URL)
    try:
        from koa.server.app import _app

        if _app and _app.config:
            notify_url = (_app.config.get("callbacks") or {}).get("notify_url", "")
            if notify_url:
                parsed = urlparse(notify_url)
                return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        pass

    return ""


class LocalBackendClient:
    def __init__(self, koiai_url: str, service_key: str = ""):
        self._base_url = koiai_url.rstrip("/")
        self._headers = {"X-Service-Key": service_key} if service_key else {}

    @classmethod
    def from_context(cls, context: AgentToolContext) -> "LocalBackendClient":
        meta = context.metadata or {}
        url = _get_backend_url() or meta.get("koiai_url", "")
        service_key = os.getenv("KOIAI_SERVICE_KEY", "") or meta.get("service_key", "")
        return cls(url, service_key)

    async def get_routing_preference(self, tenant_id: str, surface: str) -> dict | None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self._base_url}/api/internal/routing-preferences/{surface}",
                params={"tenant_id": tenant_id},
                headers=self._headers,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json().get("preference")

    async def set_routing_preference(
        self,
        tenant_id: str,
        surface: str,
        provider: str,
        account: str | None = None,
    ) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/internal/routing-preferences",
                json={
                    "tenant_id": tenant_id,
                    "surface": surface,
                    "default_provider": provider,
                    "default_account": account,
                },
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()["preference"]

    async def list_local_events(
        self,
        tenant_id: str,
        time_min: str | None = None,
        time_max: str | None = None,
        query: str | None = None,
        max_results: int = 10,
    ) -> dict:
        params: dict = {"tenant_id": tenant_id, "max_results": max_results}
        if time_min is not None:
            params["time_min"] = time_min
        if time_max is not None:
            params["time_max"] = time_max
        if query is not None:
            params["query"] = query

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self._base_url}/api/internal/events",
                params=params,
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def create_local_event(self, tenant_id: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/internal/events",
                json={"tenant_id": tenant_id, **payload},
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def update_local_event(
        self,
        tenant_id: str,
        event_id: str,
        payload: dict,
    ) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.patch(
                f"{self._base_url}/api/internal/events/{event_id}",
                json={"tenant_id": tenant_id, **payload},
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def delete_local_event(self, tenant_id: str, event_id: str) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(
                f"{self._base_url}/api/internal/events/{event_id}",
                params={"tenant_id": tenant_id},
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def search_local_todos(
        self,
        tenant_id: str,
        query: str | None = None,
        list_id: str | None = None,
        completed: bool | None = None,
    ) -> dict:
        params = {"tenant_id": tenant_id}
        if query is not None:
            params["query"] = query
        if list_id is not None:
            params["list_id"] = list_id
        if completed is not None:
            params["completed"] = completed

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self._base_url}/api/internal/todos",
                params=params,
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def create_local_todo(self, tenant_id: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/internal/todos",
                json={"tenant_id": tenant_id, **payload},
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def update_local_todo(
        self,
        tenant_id: str,
        todo_id: str,
        payload: dict,
    ) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.patch(
                f"{self._base_url}/api/internal/todos/{todo_id}",
                json={"tenant_id": tenant_id, **payload},
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def delete_local_todo(self, tenant_id: str, todo_id: str) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(
                f"{self._base_url}/api/internal/todos/{todo_id}",
                params={"tenant_id": tenant_id},
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def create_important_date(self, tenant_id: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/internal/important-dates",
                json={"tenant_id": tenant_id, **payload},
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

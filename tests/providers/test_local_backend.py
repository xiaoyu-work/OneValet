from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koa.models import AgentToolContext
from koa.providers.local_backend import LocalBackendClient


class TestLocalBackendClient:
    def test_from_context_uses_metadata(self):
        client = LocalBackendClient.from_context(
            AgentToolContext(
                tenant_id="user-1",
                metadata={
                    "koiai_url": "https://koiai.example/",
                    "service_key": "svc-key",
                },
            )
        )

        assert client._base_url == "https://koiai.example"
        assert client._headers == {"X-Service-Key": "svc-key"}

    @pytest.mark.asyncio
    async def test_get_routing_preference_uses_internal_endpoint(self):
        response = MagicMock()
        response.json.return_value = {
            "preference": {
                "default_provider": "local",
                "default_account": None,
            }
        }
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.get.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.get_routing_preference("user-1", "calendar")

        async_client.get.assert_awaited_once_with(
            "https://koiai.example/api/internal/routing-preferences/calendar",
            params={"tenant_id": "user-1"},
            headers={"X-Service-Key": "svc-key"},
        )
        assert result == {
            "default_provider": "local",
            "default_account": None,
        }

    @pytest.mark.asyncio
    async def test_set_routing_preference_posts_expected_payload(self):
        response = MagicMock()
        response.json.return_value = {
            "preference": {
                "surface": "calendar",
                "default_provider": "google",
                "default_account": "primary",
            }
        }
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.post.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.set_routing_preference(
                "user-1",
                "calendar",
                "google",
                "primary",
            )

        async_client.post.assert_awaited_once_with(
            "https://koiai.example/api/internal/routing-preferences",
            json={
                "tenant_id": "user-1",
                "surface": "calendar",
                "default_provider": "google",
                "default_account": "primary",
            },
            headers={"X-Service-Key": "svc-key"},
        )
        assert result["default_provider"] == "google"

    @pytest.mark.asyncio
    async def test_list_local_events_uses_internal_endpoint_filters(self):
        response = MagicMock()
        response.json.return_value = {"events": [{"id": "event-1"}]}
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.get.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.list_local_events(
                "user-1",
                time_min="2026-04-12T00:00:00Z",
                time_max="2026-04-13T00:00:00Z",
                query="standup",
                max_results=5,
            )

        async_client.get.assert_awaited_once_with(
            "https://koiai.example/api/internal/events",
            params={
                "tenant_id": "user-1",
                "time_min": "2026-04-12T00:00:00Z",
                "time_max": "2026-04-13T00:00:00Z",
                "query": "standup",
                "max_results": 5,
            },
            headers={"X-Service-Key": "svc-key"},
        )
        assert result == {"events": [{"id": "event-1"}]}

    @pytest.mark.asyncio
    async def test_create_local_event_posts_expected_payload(self):
        response = MagicMock()
        response.json.return_value = {"event": {"id": "event-1"}}
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.post.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        payload = {
            "title": "Standup",
            "start_at": "2026-04-12T09:00:00Z",
            "end_at": "2026-04-12T09:30:00Z",
        }

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.create_local_event("user-1", payload)

        async_client.post.assert_awaited_once_with(
            "https://koiai.example/api/internal/events",
            json={"tenant_id": "user-1", **payload},
            headers={"X-Service-Key": "svc-key"},
        )
        assert result == {"event": {"id": "event-1"}}

    @pytest.mark.asyncio
    async def test_search_local_todos_uses_internal_endpoint_filters(self):
        response = MagicMock()
        response.json.return_value = {"todos": [{"id": "todo-1"}]}
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.get.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.search_local_todos("user-1", query="buy milk")

        async_client.get.assert_awaited_once_with(
            "https://koiai.example/api/internal/todos",
            params={"tenant_id": "user-1", "query": "buy milk"},
            headers={"X-Service-Key": "svc-key"},
        )
        assert result == {"todos": [{"id": "todo-1"}]}

    @pytest.mark.asyncio
    async def test_create_local_todo_posts_expected_payload(self):
        response = MagicMock()
        response.json.return_value = {"created": True, "todo": {"id": "todo-1"}}
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.post.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        payload = {"title": "Buy milk", "due_at": "2026-04-12T17:00:00Z"}

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.create_local_todo("user-1", payload)

        async_client.post.assert_awaited_once_with(
            "https://koiai.example/api/internal/todos",
            json={"tenant_id": "user-1", **payload},
            headers={"X-Service-Key": "svc-key"},
        )
        assert result == {"created": True, "todo": {"id": "todo-1"}}

    @pytest.mark.asyncio
    async def test_create_important_date_posts_expected_payload(self):
        response = MagicMock()
        response.json.return_value = {"created": True, "date": {"id": "important-1"}}
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.post.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        payload = {
            "title": "Mom's birthday",
            "date": "2026-05-01",
            "category": "birthday",
        }

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.create_important_date("user-1", payload)

        async_client.post.assert_awaited_once_with(
            "https://koiai.example/api/internal/important-dates",
            json={"tenant_id": "user-1", **payload},
            headers={"X-Service-Key": "svc-key"},
        )
        assert result == {"created": True, "date": {"id": "important-1"}}

    @pytest.mark.asyncio
    async def test_update_local_event_patches_expected_payload(self):
        response = MagicMock()
        response.json.return_value = {"updated": True, "event": {"id": "event-1"}}
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.patch.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        payload = {"title": "Weekly sync", "location": "Room B"}

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.update_local_event("user-1", "event-1", payload)

        async_client.patch.assert_awaited_once_with(
            "https://koiai.example/api/internal/events/event-1",
            json={"tenant_id": "user-1", **payload},
            headers={"X-Service-Key": "svc-key"},
        )
        assert result == {"updated": True, "event": {"id": "event-1"}}

    @pytest.mark.asyncio
    async def test_delete_local_event_calls_internal_endpoint(self):
        response = MagicMock()
        response.json.return_value = {"deleted": True}
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.delete.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.delete_local_event("user-1", "event-1")

        async_client.delete.assert_awaited_once_with(
            "https://koiai.example/api/internal/events/event-1",
            params={"tenant_id": "user-1"},
            headers={"X-Service-Key": "svc-key"},
        )
        assert result == {"deleted": True}

    @pytest.mark.asyncio
    async def test_search_local_todos_supports_list_and_completed_filters(self):
        response = MagicMock()
        response.json.return_value = {"todos": [{"id": "todo-1"}]}
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.get.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.search_local_todos(
                "user-1",
                query="buy milk",
                list_id="inbox",
                completed=False,
            )

        async_client.get.assert_awaited_once_with(
            "https://koiai.example/api/internal/todos",
            params={
                "tenant_id": "user-1",
                "query": "buy milk",
                "list_id": "inbox",
                "completed": False,
            },
            headers={"X-Service-Key": "svc-key"},
        )
        assert result == {"todos": [{"id": "todo-1"}]}

    @pytest.mark.asyncio
    async def test_update_local_todo_patches_expected_payload(self):
        response = MagicMock()
        response.json.return_value = {"updated": True, "todo": {"id": "todo-1"}}
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.patch.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        payload = {"is_completed": True}

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.update_local_todo("user-1", "todo-1", payload)

        async_client.patch.assert_awaited_once_with(
            "https://koiai.example/api/internal/todos/todo-1",
            json={"tenant_id": "user-1", **payload},
            headers={"X-Service-Key": "svc-key"},
        )
        assert result == {"updated": True, "todo": {"id": "todo-1"}}

    @pytest.mark.asyncio
    async def test_delete_local_todo_calls_internal_endpoint(self):
        response = MagicMock()
        response.json.return_value = {"deleted": True}
        response.raise_for_status = MagicMock()

        async_client = AsyncMock()
        async_client.delete.return_value = response
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = async_client

        with patch(
            "koa.providers.local_backend.httpx.AsyncClient",
            return_value=async_cm,
        ):
            client = LocalBackendClient("https://koiai.example", "svc-key")
            result = await client.delete_local_todo("user-1", "todo-1")

        async_client.delete.assert_awaited_once_with(
            "https://koiai.example/api/internal/todos/todo-1",
            params={"tenant_id": "user-1"},
            headers={"X-Service-Key": "svc-key"},
        )
        assert result == {"deleted": True}

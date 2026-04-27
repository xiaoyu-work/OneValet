import pytest

from koa.providers.todo.local import LocalTodoProvider


class DummyBackend:
    def __init__(self):
        self.calls = []

    async def search_local_todos(self, tenant_id, query=None, list_id=None, completed=None):
        self.calls.append(
            (
                "search_local_todos",
                tenant_id,
                {"query": query, "list_id": list_id, "completed": completed},
            )
        )
        return {
            "todos": [
                {
                    "id": "todo-1",
                    "title": "Buy milk",
                    "notes": "2%",
                    "due_date": "2026-04-12",
                    "priority": 3,
                    "is_completed": False,
                    "list_id": "inbox",
                }
            ]
        }

    async def create_local_todo(self, tenant_id, payload):
        self.calls.append(("create_local_todo", tenant_id, payload))
        return {
            "created": True,
            "todo": {
                "id": "todo-1",
                "title": payload["title"],
                "notes": payload.get("notes"),
                "due_date": payload.get("due_date"),
                "priority": payload.get("priority", 0),
                "is_completed": False,
                "list_id": payload.get("list_id"),
            },
        }

    async def update_local_todo(self, tenant_id, todo_id, payload):
        self.calls.append(("update_local_todo", tenant_id, todo_id, payload))
        return {
            "updated": True,
            "todo": {
                "id": todo_id,
                "title": "Buy milk",
                "notes": "2%",
                "due_date": "2026-04-12",
                "priority": 3,
                "is_completed": payload.get("is_completed", False),
                "list_id": "inbox",
            },
        }

    async def delete_local_todo(self, tenant_id, todo_id):
        self.calls.append(("delete_local_todo", tenant_id, todo_id))
        return {"deleted": True}


class TestLocalTodoProvider:
    @pytest.mark.asyncio
    async def test_list_tasks_uses_backend_filters_and_maps_shape(self):
        backend = DummyBackend()
        provider = LocalTodoProvider(tenant_id="user-1", backend_client=backend)

        result = await provider.list_tasks(list_id="inbox", completed=False)

        assert result["success"] is True
        assert backend.calls == [
            (
                "search_local_todos",
                "user-1",
                {"query": None, "list_id": "inbox", "completed": False},
            )
        ]
        task = result["data"][0]
        assert task["id"] == "todo-1"
        assert task["title"] == "Buy milk"
        assert task["due"] == "2026-04-12"
        assert task["priority"] == "high"
        assert task["completed"] is False
        assert task["description"] == "2%"
        assert task["list_id"] == "inbox"

    @pytest.mark.asyncio
    async def test_search_tasks_uses_backend_query(self):
        backend = DummyBackend()
        provider = LocalTodoProvider(tenant_id="user-1", backend_client=backend)

        result = await provider.search_tasks("milk", list_id="inbox")

        assert result["success"] is True
        assert backend.calls == [
            (
                "search_local_todos",
                "user-1",
                {"query": "milk", "list_id": "inbox", "completed": None},
            )
        ]
        assert result["data"][0]["title"] == "Buy milk"

    @pytest.mark.asyncio
    async def test_create_task_uses_backend_client(self):
        backend = DummyBackend()
        provider = LocalTodoProvider(tenant_id="user-1", backend_client=backend)

        result = await provider.create_task(
            title="Buy milk",
            due="2026-04-12",
            priority="high",
            description="2%",
            list_id="inbox",
        )

        assert result["success"] is True
        assert backend.calls[-1] == (
            "create_local_todo",
            "user-1",
            {
                "title": "Buy milk",
                "notes": "2%",
                "due_date": "2026-04-12",
                "priority": 3,
                "list_id": "inbox",
            },
        )
        assert result["data"]["title"] == "Buy milk"
        assert result["data"]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_create_task_without_priority_defaults_to_none(self):
        backend = DummyBackend()
        provider = LocalTodoProvider(tenant_id="user-1", backend_client=backend)

        result = await provider.create_task(title="Buy milk")

        assert result["success"] is True
        assert backend.calls[-1] == (
            "create_local_todo",
            "user-1",
            {
                "title": "Buy milk",
                "notes": None,
                "due_date": None,
                "priority": 0,
                "list_id": None,
            },
        )
        assert result["data"]["priority"] == "none"

    @pytest.mark.asyncio
    async def test_complete_task_uses_backend_client(self):
        backend = DummyBackend()
        provider = LocalTodoProvider(tenant_id="user-1", backend_client=backend)

        result = await provider.complete_task("todo-1", list_id="inbox")

        assert result["success"] is True
        assert backend.calls[-1] == (
            "update_local_todo",
            "user-1",
            "todo-1",
            {"is_completed": True},
        )

    @pytest.mark.asyncio
    async def test_delete_task_uses_backend_client(self):
        backend = DummyBackend()
        provider = LocalTodoProvider(tenant_id="user-1", backend_client=backend)

        result = await provider.delete_task("todo-1", list_id="inbox")

        assert result == {"success": True}
        assert backend.calls[-1] == ("delete_local_todo", "user-1", "todo-1")

    @pytest.mark.asyncio
    async def test_ensure_valid_token_is_always_true(self):
        provider = LocalTodoProvider(tenant_id="user-1", backend_client=DummyBackend())

        assert await provider.ensure_valid_token() is True


class FailingBackend:
    async def search_local_todos(self, tenant_id, query=None, list_id=None, completed=None):
        raise RuntimeError("backend down")

    async def create_local_todo(self, tenant_id, payload):
        raise RuntimeError("backend down")

    async def update_local_todo(self, tenant_id, todo_id, payload):
        raise RuntimeError("backend down")

    async def delete_local_todo(self, tenant_id, todo_id):
        raise RuntimeError("backend down")


class TestLocalTodoProviderErrorPaths:
    @pytest.mark.asyncio
    async def test_list_tasks_returns_failure_when_backend_raises(self):
        provider = LocalTodoProvider(tenant_id="user-1", backend_client=FailingBackend())

        result = await provider.list_tasks()

        assert result["success"] is False
        assert "backend down" in result["error"]

    @pytest.mark.asyncio
    async def test_create_task_returns_failure_when_backend_raises(self):
        provider = LocalTodoProvider(tenant_id="user-1", backend_client=FailingBackend())

        result = await provider.create_task(title="Buy milk")

        assert result["success"] is False
        assert "backend down" in result["error"]

    @pytest.mark.asyncio
    async def test_complete_task_returns_failure_when_backend_raises(self):
        provider = LocalTodoProvider(tenant_id="user-1", backend_client=FailingBackend())

        result = await provider.complete_task("todo-1")

        assert result["success"] is False
        assert "backend down" in result["error"]

    @pytest.mark.asyncio
    async def test_delete_task_returns_failure_when_backend_raises(self):
        provider = LocalTodoProvider(tenant_id="user-1", backend_client=FailingBackend())

        result = await provider.delete_task("todo-1")

        assert result["success"] is False
        assert "backend down" in result["error"]

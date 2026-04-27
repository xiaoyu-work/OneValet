from unittest.mock import AsyncMock, patch

import pytest

from koa.builtin_agents.shared.routing_preferences import (
    ResolvedSurfaceTarget,
    set_routing_preference,
)
from koa.builtin_agents.todo.agent import TodoAgent
from koa.builtin_agents.todo.tools import (
    _preview_create_task,
    _resolve_todo_provider,
    check_overdue_tasks,
    create_task,
    delete_task,
    query_tasks,
    update_task,
)
from koa.models import AgentToolContext, ToolOutput


class DummyTodoProvider:
    async def ensure_valid_token(self):
        return True

    async def list_tasks(self, **kwargs):
        return {
            "success": True,
            "data": [
                {
                    "id": "todo-1",
                    "title": "Buy milk",
                    "due": "2026-04-12",
                    "priority": "high",
                    "completed": False,
                    "description": "2%",
                    "list_id": "inbox",
                }
            ],
        }

    async def search_tasks(self, query, list_id=None):
        return await self.list_tasks()

    async def create_task(self, **kwargs):
        return {
            "success": True,
            "data": {
                "id": "todo-1",
                "title": kwargs["title"],
            },
        }

    async def complete_task(self, task_id, list_id=None):
        return {"success": True, "data": {"id": task_id}}

    async def delete_task(self, task_id, list_id=None):
        return {"success": True}


class OverdueTodoProvider(DummyTodoProvider):
    async def list_tasks(self, **kwargs):
        return {
            "success": True,
            "data": [
                {
                    "id": "todo-1",
                    "title": "File taxes",
                    "due": "2000-01-01",
                    "completed": False,
                }
            ],
        }


def _context() -> AgentToolContext:
    return AgentToolContext(
        tenant_id="user-1",
        metadata={"koiai_url": "https://koiai.example", "service_key": "svc-key"},
    )


class TestTodoPreviews:
    @pytest.mark.asyncio
    async def test_preview_create_task_returns_task_draft_marker(self):
        preview = await _preview_create_task(
            {"title": "File taxes", "due": "2026-04-18", "priority": "high"},
            None,
        )

        assert "<!-- inline_card:" in preview
        assert '"card_type": "task_draft"' in preview

    @pytest.mark.asyncio
    async def test_preview_set_reminder_returns_reminder_draft_marker(self):
        from koa.builtin_agents.todo.tools import _preview_set_reminder

        preview = await _preview_set_reminder(
            {
                "schedule_datetime": "2026-04-13T09:00:00",
                "reminder_message": "Call mom",
                "human_readable_time": "tomorrow at 9am",
            },
            None,
        )

        assert "<!-- inline_card:" in preview
        assert '"card_type": "reminder_draft"' in preview

    @pytest.mark.asyncio
    async def test_preview_important_date_returns_reminder_draft_marker(self):
        from koa.builtin_agents.todo.tools import _preview_important_date

        preview = await _preview_important_date(
            {
                "title": "Mom's birthday",
                "date": "2026-05-04",
                "category": "birthday",
            },
            None,
        )

        assert "<!-- inline_card:" in preview
        assert '"card_type": "reminder_draft"' in preview


class TestTodoToolSchema:
    def test_todo_tools_accept_explicit_target_arguments(self):
        for tool in (query_tasks, create_task, update_task, delete_task):
            assert "target_provider" in tool.parameters["properties"]
            assert "target_account" in tool.parameters["properties"]

    def test_create_task_description_matches_routed_behavior(self):
        assert create_task.description == "Create a new todo task in the resolved destination."

    def test_create_task_marks_account_as_compatibility_alias(self):
        description = create_task.parameters["properties"]["account"]["description"]
        assert "deprecated" in description.lower()
        assert "target_account" in description


class TestTodoToolRouting:
    @pytest.mark.asyncio
    async def test_query_tasks_uses_resolved_provider(self):
        provider = DummyTodoProvider()

        with patch(
            "koa.builtin_agents.todo.tools._resolve_todo_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await query_tasks.executor(
                {"search_query": None, "target_provider": "local"},
                _context(),
            )

        assert isinstance(result, ToolOutput)
        assert "Found 1 task(s):" in result.text

    @pytest.mark.asyncio
    async def test_query_tasks_routing_resolution_failure_returns_read_error(self):
        """Routing infrastructure failure inside query_tasks must emit read-oriented messaging."""
        with (
            patch(
                "koa.builtin_agents.todo.tools.resolve_surface_target",
                new=AsyncMock(side_effect=RuntimeError("backend down")),
            ),
            patch(
                "koa.builtin_agents.todo.tools.LocalBackendClient.from_context",
                return_value=object(),
            ),
        ):
            result = await query_tasks.executor(
                {"search_query": None},
                _context(),
            )

        assert "retrieve" in result.lower() or "read" in result.lower()
        assert "save it locally" not in result.lower()

    @pytest.mark.asyncio
    async def test_check_overdue_tasks_routing_resolution_failure_returns_read_error(self):
        """Routing infrastructure failure inside check_overdue_tasks must emit read-oriented messaging."""
        with (
            patch(
                "koa.builtin_agents.todo.tools.resolve_surface_target",
                new=AsyncMock(side_effect=RuntimeError("backend down")),
            ),
            patch(
                "koa.builtin_agents.todo.tools.LocalBackendClient.from_context",
                return_value=object(),
            ),
        ):
            result = await check_overdue_tasks.executor({}, _context())

        assert "retrieve" in result.lower() or "read" in result.lower()
        assert "save it locally" not in result.lower()

    @pytest.mark.asyncio
    async def test_create_task_routing_resolution_failure_returns_write_error(self):
        """Routing infrastructure failure inside create_task must emit write-oriented messaging."""
        with (
            patch(
                "koa.builtin_agents.todo.tools.resolve_surface_target",
                new=AsyncMock(side_effect=RuntimeError("backend down")),
            ),
            patch(
                "koa.builtin_agents.todo.tools.LocalBackendClient.from_context",
                return_value=object(),
            ),
        ):
            result = await create_task.executor({"title": "Buy milk"}, _context())

        assert "couldn't finish that todo action" in result.lower() or "try again" in result.lower()

    @pytest.mark.asyncio
    async def test_create_task_uses_resolved_provider(self):
        provider = DummyTodoProvider()

        resolver = AsyncMock(return_value=(provider, {"provider": "local"}, None))
        with patch(
            "koa.builtin_agents.todo.tools._resolve_todo_provider",
            new=resolver,
            create=True,
        ):
            result = await create_task.executor(
                {"title": "Buy milk", "target_provider": "local"},
                _context(),
            )

        assert "added 'buy milk'" in result.lower()
        assert resolver.await_args.kwargs["target_account"] is None

    @pytest.mark.asyncio
    async def test_create_task_forwards_explicit_account_override(self):
        provider = DummyTodoProvider()

        resolver = AsyncMock(return_value=(provider, {"provider": "google"}, None))
        with patch(
            "koa.builtin_agents.todo.tools._resolve_todo_provider",
            new=resolver,
            create=True,
        ):
            result = await create_task.executor(
                {"title": "Buy milk", "account": "work"},
                _context(),
            )

        assert "added 'buy milk'" in result.lower()
        assert resolver.await_args.kwargs["target_provider"] is None
        assert resolver.await_args.kwargs["target_account"] == "work"

    @pytest.mark.asyncio
    async def test_create_task_does_not_schedule_cron_reminder(self):
        """create_task must not auto-write to CronService as a side effect of task creation."""
        provider = DummyTodoProvider()
        cron_service = AsyncMock()

        ctx = AgentToolContext(
            tenant_id="user-1",
            metadata={"koiai_url": "https://koiai.example", "service_key": "svc-key"},
            context_hints={"cron_service": cron_service},
        )

        resolver = AsyncMock(return_value=(provider, {"provider": "local"}, None))
        with patch(
            "koa.builtin_agents.todo.tools._resolve_todo_provider",
            new=resolver,
            create=True,
        ):
            result = await create_task.executor(
                {"title": "Buy milk", "due": "2030-01-01"},
                ctx,
            )

        assert "added 'buy milk'" in result.lower()
        cron_service.add.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_task_search_failure_returns_read_error(self):
        """When search_tasks fails in update_task, the error must be read-oriented."""

        class FailingSearchProvider(DummyTodoProvider):
            async def search_tasks(self, query, list_id=None):
                return {"success": False}

        provider = FailingSearchProvider()
        with patch(
            "koa.builtin_agents.todo.tools._resolve_todo_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await update_task.executor(
                {"search_query": "buy milk"},
                _context(),
            )

        assert "read" in result.lower() or "retrieve" in result.lower()

    @pytest.mark.asyncio
    async def test_delete_task_search_failure_returns_read_error(self):
        """When search_tasks fails in delete_task, the error must be read-oriented."""

        class FailingSearchProvider(DummyTodoProvider):
            async def search_tasks(self, query, list_id=None):
                return {"success": False}

        provider = FailingSearchProvider()
        with patch(
            "koa.builtin_agents.todo.tools._resolve_todo_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await delete_task.executor(
                {"search_query": "buy milk"},
                _context(),
            )

        assert "read" in result.lower() or "retrieve" in result.lower()

    @pytest.mark.asyncio
    async def test_update_task_uses_resolved_provider(self):
        provider = DummyTodoProvider()

        with patch(
            "koa.builtin_agents.todo.tools._resolve_todo_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await update_task.executor(
                {"search_query": "buy milk", "target_provider": "local"},
                _context(),
            )

        assert "marked" in result.lower()
        assert "complete" in result.lower()

    @pytest.mark.asyncio
    async def test_delete_task_uses_resolved_provider(self):
        provider = DummyTodoProvider()

        with patch(
            "koa.builtin_agents.todo.tools._resolve_todo_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await delete_task.executor(
                {"search_query": "buy milk", "target_provider": "local"},
                _context(),
            )

        assert "deleted 1 task" in result.lower()

    @pytest.mark.asyncio
    async def test_resolve_todo_provider_uses_provider_specific_account_resolution(self):
        provider = DummyTodoProvider()

        with (
            patch(
                "koa.builtin_agents.todo.tools.LocalBackendClient.from_context",
                return_value=object(),
                create=True,
            ),
            patch(
                "koa.builtin_agents.todo.tools.resolve_surface_target",
                new=AsyncMock(
                    return_value=ResolvedSurfaceTarget(
                        surface="todo",
                        provider="google",
                        account="work",
                        source="saved",
                    )
                ),
            ),
            patch(
                "koa.providers.todo.factory.TodoProviderFactory.get_supported_providers",
                return_value=["google", "todoist", "microsoft"],
            ),
            patch(
                "koa.providers.todo.resolver.TodoAccountResolver.resolve_account_for_provider",
                new=AsyncMock(
                    return_value={
                        "provider": "google",
                        "account_name": "work",
                        "email": "user@example.com",
                    }
                ),
            ),
            patch(
                "koa.providers.todo.resolver.TodoAccountResolver.resolve_account",
                new=AsyncMock(
                    side_effect=AssertionError("generic resolve_account should not be used")
                ),
                create=True,
            ),
            patch(
                "koa.builtin_agents.todo.tools._get_provider",
                return_value=provider,
            ),
        ):
            resolved_provider, account, error = await _resolve_todo_provider(_context())

        assert error is None
        assert resolved_provider is provider
        assert account["provider"] == "google"
        assert account["account_name"] == "work"

    @pytest.mark.asyncio
    async def test_remember_important_date_uses_local_backend(self):
        from koa.builtin_agents.todo.tools import remember_important_date

        backend_client = AsyncMock()
        backend_client.create_important_date.return_value = {"created": True}

        with patch(
            "koa.builtin_agents.todo.tools.LocalBackendClient.from_context",
            return_value=backend_client,
            create=True,
        ):
            result = await remember_important_date.executor(
                {
                    "title": "Mom's birthday",
                    "date": "May 4",
                    "category": "birthday",
                },
                _context(),
            )

        backend_client.create_important_date.assert_awaited_once()
        args = backend_client.create_important_date.await_args.args
        assert args[0] == "user-1"
        assert args[1]["title"] == "Mom's birthday"
        assert args[1]["category"] == "birthday"
        assert args[1]["date"].endswith("-05-04")
        assert "saved mom's birthday" in result.lower()

    @pytest.mark.asyncio
    async def test_remember_important_date_rejects_invalid_date(self):
        from koa.builtin_agents.todo.tools import remember_important_date

        backend_client = AsyncMock()

        with patch(
            "koa.builtin_agents.todo.tools.LocalBackendClient.from_context",
            return_value=backend_client,
            create=True,
        ):
            result = await remember_important_date.executor(
                {
                    "title": "Mom's birthday",
                    "date": "definitely not a date",
                    "category": "birthday",
                },
                _context(),
            )

        backend_client.create_important_date.assert_not_awaited()
        assert "date must be a real date" in result.lower()

    @pytest.mark.asyncio
    async def test_check_overdue_tasks_uses_resolved_provider_for_local_users(self):
        provider = OverdueTodoProvider()

        with patch(
            "koa.builtin_agents.todo.tools._resolve_todo_provider",
            new=AsyncMock(return_value=(provider, {"provider": "local"}, None)),
            create=True,
        ):
            result = await check_overdue_tasks.executor({}, _context())

        assert "overdue" in result.lower()
        assert "file taxes" in result.lower()


class TestTodoAgent:
    def test_todo_agent_allows_local_routing_and_remembered_dates(self):
        assert "requires_service" not in TodoAgent._valet_metadata.extra
        assert set_routing_preference in TodoAgent.tools
        assert any(tool.name == "remember_important_date" for tool in TodoAgent.tools)
        assert "remember_important_date" in TodoAgent._SYSTEM_PROMPT_TEMPLATE
        assert "set_routing_preference" in TodoAgent._SYSTEM_PROMPT_TEMPLATE

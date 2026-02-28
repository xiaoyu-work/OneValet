"""Tests for onevalet.checkpoint.manager.CheckpointManager

Tests cover:
- save_checkpoint / get_checkpoint / get_agent_state
- list_checkpoints / list_user_checkpoints
- get_checkpoint_tree / get_latest_checkpoint
- compare_checkpoints
- delete_checkpoint / clear_agent_history / clear_user_history
- set_parent_checkpoint / get_parent_checkpoint
- Parent checkpoint chaining
"""

import pytest
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from onevalet.checkpoint.manager import CheckpointManager, CheckpointError
from onevalet.checkpoint.storage import MemoryStorage


@dataclass
class MockAgent:
    """Mock agent implementing the AgentProtocol"""
    agent_id: str = "agent_1"
    user_id: str = "user_1"
    status: Any = "collecting"
    collected_fields: Dict[str, Any] = field(default_factory=dict)
    execution_state: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def get_message_history(self) -> List[Any]:
        return []


@dataclass
class MockStatus:
    value: str = "collecting"


@pytest.fixture
def storage():
    return MemoryStorage()


@pytest.fixture
def manager(storage):
    return CheckpointManager(storage=storage)


class TestSaveCheckpoint:
    async def test_returns_checkpoint_id(self, manager):
        agent = MockAgent()
        cp_id = await manager.save_checkpoint(agent)
        assert cp_id.startswith("ckpt_")

    async def test_saves_agent_state(self, manager, storage):
        agent = MockAgent(
            collected_fields={"name": "Alice"},
            execution_state={"step": 2},
            context={"lang": "en"},
        )
        cp_id = await manager.save_checkpoint(agent)
        cp = await storage.get(cp_id)
        assert cp is not None
        assert cp.agent_id == "agent_1"
        assert cp.user_id == "user_1"
        assert cp.collected_fields == {"name": "Alice"}
        assert cp.execution_state == {"step": 2}
        assert cp.context == {"lang": "en"}

    async def test_saves_message_and_result(self, manager, storage):
        agent = MockAgent()
        msg = {"content": "hello"}
        result = {"response": "hi"}
        cp_id = await manager.save_checkpoint(agent, message=msg, result=result)
        cp = await storage.get(cp_id)
        assert cp.message == msg
        assert cp.result == result

    async def test_status_with_value_attr(self, manager, storage):
        agent = MockAgent(status=MockStatus(value="waiting"))
        cp_id = await manager.save_checkpoint(agent)
        cp = await storage.get(cp_id)
        assert cp.status == "waiting"

    async def test_status_as_string(self, manager, storage):
        agent = MockAgent(status="idle")
        cp_id = await manager.save_checkpoint(agent)
        cp = await storage.get(cp_id)
        assert cp.status == "idle"

    async def test_deep_copies_fields(self, manager, storage):
        fields = {"name": "Alice"}
        agent = MockAgent(collected_fields=fields)
        cp_id = await manager.save_checkpoint(agent)
        # Mutate original
        fields["name"] = "Bob"
        cp = await storage.get(cp_id)
        assert cp.collected_fields["name"] == "Alice"


class TestParentChaining:
    async def test_first_checkpoint_has_no_parent(self, manager, storage):
        agent = MockAgent()
        cp_id = await manager.save_checkpoint(agent)
        cp = await storage.get(cp_id)
        assert cp.parent_checkpoint_id is None

    async def test_second_checkpoint_chains_to_first(self, manager, storage):
        agent = MockAgent()
        cp1_id = await manager.save_checkpoint(agent)
        cp2_id = await manager.save_checkpoint(agent)
        cp2 = await storage.get(cp2_id)
        assert cp2.parent_checkpoint_id == cp1_id

    async def test_set_parent_manually(self, manager, storage):
        agent = MockAgent()
        manager.set_parent_checkpoint("agent_1", "ckpt_manual")
        cp_id = await manager.save_checkpoint(agent)
        cp = await storage.get(cp_id)
        assert cp.parent_checkpoint_id == "ckpt_manual"

    async def test_get_parent_checkpoint(self, manager):
        assert manager.get_parent_checkpoint("agent_1") is None
        manager.set_parent_checkpoint("agent_1", "ckpt_x")
        assert manager.get_parent_checkpoint("agent_1") == "ckpt_x"


class TestGetCheckpoint:
    async def test_get_existing(self, manager):
        agent = MockAgent()
        cp_id = await manager.save_checkpoint(agent)
        cp = await manager.get_checkpoint(cp_id)
        assert cp is not None
        assert cp.id == cp_id

    async def test_get_nonexistent(self, manager):
        result = await manager.get_checkpoint("nonexistent")
        assert result is None


class TestGetAgentState:
    async def test_returns_state_dict(self, manager):
        agent = MockAgent(
            collected_fields={"email": "a@b.com"},
            execution_state={"step": 1},
        )
        cp_id = await manager.save_checkpoint(agent)
        state = await manager.get_agent_state(cp_id)
        assert state is not None
        assert state["agent_id"] == "agent_1"
        assert state["agent_type"] == "MockAgent"
        assert state["user_id"] == "user_1"
        assert state["collected_fields"] == {"email": "a@b.com"}
        assert state["execution_state"] == {"step": 1}

    async def test_nonexistent_returns_none(self, manager):
        result = await manager.get_agent_state("nonexistent")
        assert result is None


class TestListCheckpoints:
    async def test_list_for_agent(self, manager):
        agent = MockAgent()
        await manager.save_checkpoint(agent)
        await manager.save_checkpoint(agent)
        result = await manager.list_checkpoints("agent_1")
        assert len(result) == 2

    async def test_list_for_user(self, manager):
        agent = MockAgent()
        await manager.save_checkpoint(agent)
        result = await manager.list_user_checkpoints("user_1")
        assert len(result) == 1


class TestGetCheckpointTree:
    async def test_returns_tree(self, manager):
        agent = MockAgent()
        await manager.save_checkpoint(agent)
        await manager.save_checkpoint(agent)
        tree = await manager.get_checkpoint_tree("agent_1")
        assert tree is not None
        assert len(tree.nodes) == 2

    async def test_empty_returns_none(self, manager):
        result = await manager.get_checkpoint_tree("agent_1")
        assert result is None


class TestGetLatestCheckpoint:
    async def test_returns_latest(self, manager):
        agent = MockAgent()
        cp1_id = await manager.save_checkpoint(agent)
        cp2_id = await manager.save_checkpoint(agent)
        latest = await manager.get_latest_checkpoint("agent_1")
        assert latest is not None
        assert latest.id == cp2_id

    async def test_empty_returns_none(self, manager):
        result = await manager.get_latest_checkpoint("agent_1")
        assert result is None


class TestCompareCheckpoints:
    async def test_compare_two_checkpoints(self, manager):
        agent = MockAgent(collected_fields={"name": "Alice"})
        cp1_id = await manager.save_checkpoint(agent)
        agent.collected_fields = {"name": "Bob", "age": 30}
        cp2_id = await manager.save_checkpoint(agent)
        diff = await manager.compare_checkpoints(cp1_id, cp2_id)
        assert diff is not None
        assert "name" in diff.fields_modified
        assert "age" in diff.fields_added

    async def test_compare_nonexistent_returns_none(self, manager):
        agent = MockAgent()
        cp_id = await manager.save_checkpoint(agent)
        assert await manager.compare_checkpoints(cp_id, "nonexistent") is None
        assert await manager.compare_checkpoints("nonexistent", cp_id) is None


class TestDeleteAndClear:
    async def test_delete_checkpoint(self, manager):
        agent = MockAgent()
        cp_id = await manager.save_checkpoint(agent)
        result = await manager.delete_checkpoint(cp_id)
        assert result is True
        assert await manager.get_checkpoint(cp_id) is None

    async def test_clear_agent_history(self, manager):
        agent = MockAgent()
        await manager.save_checkpoint(agent)
        await manager.save_checkpoint(agent)
        count = await manager.clear_agent_history("agent_1")
        assert count == 2
        assert await manager.list_checkpoints("agent_1") == []

    async def test_clear_agent_removes_parent_ref(self, manager):
        agent = MockAgent()
        await manager.save_checkpoint(agent)
        await manager.clear_agent_history("agent_1")
        assert manager.get_parent_checkpoint("agent_1") is None

    async def test_clear_user_history(self, manager):
        agent = MockAgent()
        await manager.save_checkpoint(agent)
        count = await manager.clear_user_history("user_1")
        assert count == 1


class TestDefaultStorage:
    async def test_defaults_to_memory_storage(self):
        manager = CheckpointManager()
        assert isinstance(manager.storage, MemoryStorage)

    async def test_auto_save_flag(self):
        manager = CheckpointManager(auto_save=False)
        assert manager.auto_save is False

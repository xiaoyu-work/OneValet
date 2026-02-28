"""Tests for onevalet.checkpoint.storage.MemoryStorage

Tests cover all CheckpointStorage interface methods:
- save / get / delete
- list_by_agent / list_by_user (with pagination)
- get_tree / get_latest
- clear_agent / clear_user / clear_all
- Max checkpoints per agent enforcement
"""

import pytest
from datetime import datetime, timedelta

from onevalet.checkpoint.models import Checkpoint, CheckpointTree
from onevalet.checkpoint.storage import MemoryStorage


def _make_checkpoint(
    id="ckpt_001",
    agent_id="agent_1",
    agent_type="TestAgent",
    user_id="user_1",
    status="collecting",
    timestamp=None,
    parent_checkpoint_id=None,
    collected_fields=None,
):
    return Checkpoint(
        id=id,
        agent_id=agent_id,
        agent_type=agent_type,
        user_id=user_id,
        status=status,
        timestamp=timestamp or datetime(2025, 1, 15, 10, 0, 0),
        parent_checkpoint_id=parent_checkpoint_id,
        collected_fields=collected_fields or {},
    )


@pytest.fixture
def storage():
    return MemoryStorage()


class TestMemoryStorageSaveAndGet:
    async def test_save_returns_id(self, storage):
        cp = _make_checkpoint()
        result = await storage.save(cp)
        assert result == "ckpt_001"

    async def test_get_returns_saved_checkpoint(self, storage):
        cp = _make_checkpoint()
        await storage.save(cp)
        retrieved = await storage.get("ckpt_001")
        assert retrieved is not None
        assert retrieved.id == "ckpt_001"
        assert retrieved.agent_id == "agent_1"

    async def test_get_nonexistent_returns_none(self, storage):
        result = await storage.get("nonexistent")
        assert result is None

    async def test_save_overwrites_same_id(self, storage):
        cp1 = _make_checkpoint(status="collecting")
        cp2 = _make_checkpoint(status="completed")
        await storage.save(cp1)
        await storage.save(cp2)
        retrieved = await storage.get("ckpt_001")
        assert retrieved.status == "completed"


class TestMemoryStorageDelete:
    async def test_delete_existing(self, storage):
        await storage.save(_make_checkpoint())
        result = await storage.delete("ckpt_001")
        assert result is True
        assert await storage.get("ckpt_001") is None

    async def test_delete_nonexistent(self, storage):
        result = await storage.delete("nonexistent")
        assert result is False


class TestMemoryStorageListByAgent:
    async def test_list_empty(self, storage):
        result = await storage.list_by_agent("agent_1")
        assert result == []

    async def test_list_returns_metadata(self, storage):
        await storage.save(_make_checkpoint(id="c1"))
        result = await storage.list_by_agent("agent_1")
        assert len(result) == 1
        assert result[0].id == "c1"

    async def test_list_sorted_newest_first(self, storage):
        await storage.save(_make_checkpoint(id="c1", timestamp=datetime(2025, 1, 1)))
        await storage.save(_make_checkpoint(id="c2", timestamp=datetime(2025, 1, 2)))
        await storage.save(_make_checkpoint(id="c3", timestamp=datetime(2025, 1, 3)))
        result = await storage.list_by_agent("agent_1")
        ids = [m.id for m in result]
        assert ids == ["c3", "c2", "c1"]

    async def test_list_pagination(self, storage):
        for i in range(5):
            await storage.save(_make_checkpoint(
                id=f"c{i}",
                timestamp=datetime(2025, 1, i + 1),
            ))
        result = await storage.list_by_agent("agent_1", limit=2, offset=1)
        assert len(result) == 2
        assert result[0].id == "c3"
        assert result[1].id == "c2"

    async def test_list_filters_by_agent(self, storage):
        await storage.save(_make_checkpoint(id="c1", agent_id="agent_1"))
        await storage.save(_make_checkpoint(id="c2", agent_id="agent_2"))
        result = await storage.list_by_agent("agent_1")
        assert len(result) == 1
        assert result[0].id == "c1"


class TestMemoryStorageListByUser:
    async def test_list_by_user(self, storage):
        await storage.save(_make_checkpoint(id="c1", user_id="user_1"))
        await storage.save(_make_checkpoint(id="c2", user_id="user_2"))
        result = await storage.list_by_user("user_1")
        assert len(result) == 1
        assert result[0].id == "c1"

    async def test_list_by_user_sorted(self, storage):
        await storage.save(_make_checkpoint(id="c1", timestamp=datetime(2025, 1, 1)))
        await storage.save(_make_checkpoint(id="c2", timestamp=datetime(2025, 1, 2)))
        result = await storage.list_by_user("user_1")
        assert result[0].id == "c2"


class TestMemoryStorageTree:
    async def test_get_tree_empty(self, storage):
        result = await storage.get_tree("agent_1")
        assert result is None

    async def test_get_tree_builds_structure(self, storage):
        await storage.save(_make_checkpoint(id="c1", timestamp=datetime(2025, 1, 1)))
        await storage.save(_make_checkpoint(
            id="c2", parent_checkpoint_id="c1", timestamp=datetime(2025, 1, 2),
        ))
        tree = await storage.get_tree("agent_1")
        assert tree is not None
        assert tree.root_id == "c1"
        assert "c1" in tree.nodes
        assert "c2" in tree.nodes
        assert "c2" in tree.children.get("c1", [])


class TestMemoryStorageLatest:
    async def test_get_latest_empty(self, storage):
        result = await storage.get_latest("agent_1")
        assert result is None

    async def test_get_latest_returns_newest(self, storage):
        await storage.save(_make_checkpoint(id="c1", timestamp=datetime(2025, 1, 1)))
        await storage.save(_make_checkpoint(id="c2", timestamp=datetime(2025, 1, 2)))
        result = await storage.get_latest("agent_1")
        assert result.id == "c2"


class TestMemoryStorageClear:
    async def test_clear_agent(self, storage):
        await storage.save(_make_checkpoint(id="c1"))
        await storage.save(_make_checkpoint(id="c2"))
        count = await storage.clear_agent("agent_1")
        assert count == 2
        assert await storage.list_by_agent("agent_1") == []

    async def test_clear_user(self, storage):
        await storage.save(_make_checkpoint(id="c1"))
        count = await storage.clear_user("user_1")
        assert count == 1
        assert await storage.list_by_user("user_1") == []

    async def test_clear_all(self, storage):
        await storage.save(_make_checkpoint(id="c1"))
        await storage.save(_make_checkpoint(id="c2", agent_id="agent_2", user_id="user_2"))
        storage.clear_all()
        assert await storage.get("c1") is None
        assert await storage.get("c2") is None


class TestMemoryStorageMaxCheckpoints:
    async def test_evicts_oldest_when_max_exceeded(self):
        storage = MemoryStorage(max_checkpoints_per_agent=3)
        for i in range(5):
            await storage.save(_make_checkpoint(
                id=f"c{i}",
                timestamp=datetime(2025, 1, i + 1),
            ))
        # c0 and c1 should have been evicted
        assert await storage.get("c0") is None
        assert await storage.get("c1") is None
        assert await storage.get("c2") is not None
        assert await storage.get("c3") is not None
        assert await storage.get("c4") is not None

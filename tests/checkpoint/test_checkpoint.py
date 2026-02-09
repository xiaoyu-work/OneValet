"""
Tests for FlowAgent Checkpoint System.

Tests:
- Checkpoint models and serialization
- Memory storage backend
- Checkpoint manager operations
- Time-travel and branching
"""

import pytest
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List

from flowagents.checkpoint import (
    # Models
    Checkpoint,
    CheckpointMetadata,
    CheckpointTree,
    # Storage
    CheckpointStorage,
    MemoryStorage,
    # Manager
    CheckpointManager,
    CheckpointError,
)
from flowagents.checkpoint.models import CheckpointDiff


# ============================================================================
# Test Models
# ============================================================================

class TestCheckpoint:
    """Tests for Checkpoint model"""

    def test_checkpoint_creation(self):
        """Test creating a checkpoint"""
        checkpoint = Checkpoint(
            id="ckpt_123",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="running",
            collected_fields={"name": "Alice", "email": "alice@example.com"},
            execution_state={"progress": 50},
            context={"timezone": "UTC"},
        )

        assert checkpoint.id == "ckpt_123"
        assert checkpoint.agent_type == "TestAgent"
        assert checkpoint.collected_fields["name"] == "Alice"
        assert checkpoint.execution_state["progress"] == 50

    def test_checkpoint_serialization(self):
        """Test checkpoint to/from dict"""
        checkpoint = Checkpoint(
            id="ckpt_123",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="completed",
            collected_fields={"field": "value"},
            parent_checkpoint_id="ckpt_122",
        )

        # To dict
        data = checkpoint.to_dict()
        assert data["id"] == "ckpt_123"
        assert data["status"] == "completed"
        assert data["parent_checkpoint_id"] == "ckpt_122"

        # From dict
        restored = Checkpoint.from_dict(data)
        assert restored.id == checkpoint.id
        assert restored.status == checkpoint.status
        assert restored.collected_fields == checkpoint.collected_fields

    def test_checkpoint_json(self):
        """Test checkpoint JSON serialization"""
        checkpoint = Checkpoint(
            id="ckpt_123",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="waiting",
        )

        json_str = checkpoint.to_json()
        restored = Checkpoint.from_json(json_str)

        assert restored.id == checkpoint.id
        assert restored.status == checkpoint.status

    def test_generate_id(self):
        """Test ID generation"""
        id1 = Checkpoint.generate_id()
        id2 = Checkpoint.generate_id()

        assert id1.startswith("ckpt_")
        assert id2.startswith("ckpt_")
        assert id1 != id2


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata"""

    def test_from_checkpoint(self):
        """Test creating metadata from checkpoint"""
        checkpoint = Checkpoint(
            id="ckpt_123",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="running",
            collected_fields={"a": 1, "b": 2, "c": 3},
            message={"content": "Hello, this is a test message"},
        )

        metadata = CheckpointMetadata.from_checkpoint(checkpoint)

        assert metadata.id == "ckpt_123"
        assert metadata.fields_count == 3
        assert metadata.message_preview == "Hello, this is a test message"


class TestCheckpointTree:
    """Tests for CheckpointTree"""

    def test_tree_building(self):
        """Test building a checkpoint tree"""
        tree = CheckpointTree(root_id="ckpt_1")

        # Add checkpoints
        ckpt1 = Checkpoint(
            id="ckpt_1",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="running",
        )
        ckpt2 = Checkpoint(
            id="ckpt_2",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="waiting",
            parent_checkpoint_id="ckpt_1",
        )
        ckpt3 = Checkpoint(
            id="ckpt_3",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="completed",
            parent_checkpoint_id="ckpt_2",
        )

        tree.add_checkpoint(ckpt1)
        tree.add_checkpoint(ckpt2)
        tree.add_checkpoint(ckpt3)

        assert len(tree.nodes) == 3
        assert tree.get_branches("ckpt_1") == ["ckpt_2"]
        assert tree.get_branches("ckpt_2") == ["ckpt_3"]

    def test_path_to_root(self):
        """Test getting path to root"""
        tree = CheckpointTree(root_id="ckpt_1")

        for i in range(1, 5):
            ckpt = Checkpoint(
                id=f"ckpt_{i}",
                agent_id="agent_1",
                agent_type="TestAgent",
                user_id="user_1",
                status="running",
                parent_checkpoint_id=f"ckpt_{i-1}" if i > 1 else None,
            )
            tree.add_checkpoint(ckpt)

        path = tree.get_path_to_root("ckpt_4")
        assert path == ["ckpt_4", "ckpt_3", "ckpt_2", "ckpt_1"]

    def test_branching(self):
        """Test branching in tree"""
        tree = CheckpointTree(root_id="ckpt_1")

        # Root
        tree.add_checkpoint(Checkpoint(
            id="ckpt_1",
            agent_id="a",
            agent_type="T",
            user_id="u",
            status="s"
        ))

        # Two branches from root
        tree.add_checkpoint(Checkpoint(
            id="ckpt_2a",
            agent_id="a",
            agent_type="T",
            user_id="u",
            status="s",
            parent_checkpoint_id="ckpt_1",
            branch_label="branch_a"
        ))
        tree.add_checkpoint(Checkpoint(
            id="ckpt_2b",
            agent_id="a",
            agent_type="T",
            user_id="u",
            status="s",
            parent_checkpoint_id="ckpt_1",
            branch_label="branch_b"
        ))

        branches = tree.get_branches("ckpt_1")
        assert len(branches) == 2
        assert "ckpt_2a" in branches
        assert "ckpt_2b" in branches

    def test_leaf_nodes(self):
        """Test getting leaf nodes"""
        tree = CheckpointTree(root_id="ckpt_1")

        tree.add_checkpoint(Checkpoint(
            id="ckpt_1", agent_id="a", agent_type="T", user_id="u", status="s"
        ))
        tree.add_checkpoint(Checkpoint(
            id="ckpt_2", agent_id="a", agent_type="T", user_id="u", status="s",
            parent_checkpoint_id="ckpt_1"
        ))
        tree.add_checkpoint(Checkpoint(
            id="ckpt_3", agent_id="a", agent_type="T", user_id="u", status="s",
            parent_checkpoint_id="ckpt_1"
        ))

        leaves = tree.get_leaf_nodes()
        assert len(leaves) == 2
        assert "ckpt_2" in leaves
        assert "ckpt_3" in leaves


class TestCheckpointDiff:
    """Tests for CheckpointDiff"""

    def test_compute_diff(self):
        """Test computing diff between checkpoints"""
        ckpt1 = Checkpoint(
            id="ckpt_1",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="waiting",
            collected_fields={"name": "Alice", "email": "alice@example.com"},
        )
        ckpt2 = Checkpoint(
            id="ckpt_2",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="running",
            collected_fields={"name": "Alice", "phone": "123-456"},
        )

        diff = CheckpointDiff.compute(ckpt1, ckpt2)

        assert diff.status_changed is True
        assert diff.old_status == "waiting"
        assert diff.new_status == "running"

        assert "phone" in diff.fields_added
        assert "email" in diff.fields_removed
        assert len(diff.fields_modified) == 0  # name unchanged

    def test_no_changes(self):
        """Test diff with identical checkpoints"""
        ckpt = Checkpoint(
            id="ckpt_1",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="running",
            collected_fields={"a": 1},
        )

        diff = CheckpointDiff.compute(ckpt, ckpt)
        assert diff.has_changes() is False


# ============================================================================
# Test Storage
# ============================================================================

class TestMemoryStorage:
    """Tests for MemoryStorage"""

    @pytest.mark.asyncio
    async def test_save_and_get(self):
        """Test saving and retrieving checkpoints"""
        storage = MemoryStorage()

        checkpoint = Checkpoint(
            id="ckpt_123",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="running",
        )

        await storage.save(checkpoint)
        retrieved = await storage.get("ckpt_123")

        assert retrieved is not None
        assert retrieved.id == "ckpt_123"

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting checkpoints"""
        storage = MemoryStorage()

        checkpoint = Checkpoint(
            id="ckpt_123",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="running",
        )

        await storage.save(checkpoint)
        assert await storage.delete("ckpt_123") is True
        assert await storage.get("ckpt_123") is None
        assert await storage.delete("ckpt_123") is False

    @pytest.mark.asyncio
    async def test_list_by_agent(self):
        """Test listing checkpoints by agent"""
        storage = MemoryStorage()

        for i in range(5):
            checkpoint = Checkpoint(
                id=f"ckpt_{i}",
                agent_id="agent_1",
                agent_type="TestAgent",
                user_id="user_1",
                status="running",
            )
            await storage.save(checkpoint)

        # Add checkpoint for different agent
        await storage.save(Checkpoint(
            id="ckpt_other",
            agent_id="agent_2",
            agent_type="TestAgent",
            user_id="user_1",
            status="running",
        ))

        checkpoints = await storage.list_by_agent("agent_1")
        assert len(checkpoints) == 5

        checkpoints = await storage.list_by_agent("agent_2")
        assert len(checkpoints) == 1

    @pytest.mark.asyncio
    async def test_list_by_user(self):
        """Test listing checkpoints by user"""
        storage = MemoryStorage()

        for i in range(3):
            await storage.save(Checkpoint(
                id=f"ckpt_u1_{i}",
                agent_id=f"agent_{i}",
                agent_type="TestAgent",
                user_id="user_1",
                status="running",
            ))

        await storage.save(Checkpoint(
            id="ckpt_u2",
            agent_id="agent_x",
            agent_type="TestAgent",
            user_id="user_2",
            status="running",
        ))

        checkpoints = await storage.list_by_user("user_1")
        assert len(checkpoints) == 3

        checkpoints = await storage.list_by_user("user_2")
        assert len(checkpoints) == 1

    @pytest.mark.asyncio
    async def test_get_latest(self):
        """Test getting latest checkpoint"""
        storage = MemoryStorage()

        for i in range(3):
            checkpoint = Checkpoint(
                id=f"ckpt_{i}",
                agent_id="agent_1",
                agent_type="TestAgent",
                user_id="user_1",
                status=f"status_{i}",
            )
            await storage.save(checkpoint)

        latest = await storage.get_latest("agent_1")
        assert latest is not None
        assert latest.id == "ckpt_2"

    @pytest.mark.asyncio
    async def test_get_tree(self):
        """Test getting checkpoint tree"""
        storage = MemoryStorage()

        await storage.save(Checkpoint(
            id="ckpt_1",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="running",
        ))
        await storage.save(Checkpoint(
            id="ckpt_2",
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            status="running",
            parent_checkpoint_id="ckpt_1",
        ))

        tree = await storage.get_tree("agent_1")
        assert tree is not None
        assert tree.root_id == "ckpt_1"
        assert len(tree.nodes) == 2

    @pytest.mark.asyncio
    async def test_clear_agent(self):
        """Test clearing all checkpoints for agent"""
        storage = MemoryStorage()

        for i in range(3):
            await storage.save(Checkpoint(
                id=f"ckpt_{i}",
                agent_id="agent_1",
                agent_type="TestAgent",
                user_id="user_1",
                status="running",
            ))

        count = await storage.clear_agent("agent_1")
        assert count == 3

        checkpoints = await storage.list_by_agent("agent_1")
        assert len(checkpoints) == 0

    @pytest.mark.asyncio
    async def test_max_checkpoints_limit(self):
        """Test max checkpoints per agent limit"""
        storage = MemoryStorage(max_checkpoints_per_agent=3)

        for i in range(5):
            await storage.save(Checkpoint(
                id=f"ckpt_{i}",
                agent_id="agent_1",
                agent_type="TestAgent",
                user_id="user_1",
                status="running",
            ))

        checkpoints = await storage.list_by_agent("agent_1")
        assert len(checkpoints) == 3

        # Should have newest 3
        ids = [c.id for c in checkpoints]
        assert "ckpt_4" in ids
        assert "ckpt_3" in ids
        assert "ckpt_2" in ids


# ============================================================================
# Test Manager
# ============================================================================

@dataclass
class MockAgent:
    """Mock agent for testing"""
    agent_id: str
    user_id: str
    status: str = "running"
    collected_fields: Dict[str, Any] = None
    execution_state: Dict[str, Any] = None
    context: Dict[str, Any] = None

    def __post_init__(self):
        self.collected_fields = self.collected_fields or {}
        self.execution_state = self.execution_state or {}
        self.context = self.context or {}

    def get_message_history(self) -> List:
        return []


class TestCheckpointManager:
    """Tests for CheckpointManager"""

    @pytest.mark.asyncio
    async def test_save_checkpoint(self):
        """Test saving checkpoint through manager"""
        manager = CheckpointManager()

        agent = MockAgent(
            agent_id="agent_1",
            user_id="user_1",
            status="running",
            collected_fields={"name": "Alice"},
        )

        checkpoint_id = await manager.save_checkpoint(
            agent,
            message={"content": "Hello"},
            result={"status": "success"}
        )

        assert checkpoint_id.startswith("ckpt_")

        # Verify saved
        checkpoint = await manager.get_checkpoint(checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.collected_fields["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_checkpoint_chaining(self):
        """Test checkpoints are linked to parents"""
        manager = CheckpointManager()

        agent = MockAgent(agent_id="agent_1", user_id="user_1")

        id1 = await manager.save_checkpoint(agent)
        id2 = await manager.save_checkpoint(agent)
        id3 = await manager.save_checkpoint(agent)

        ckpt3 = await manager.get_checkpoint(id3)
        assert ckpt3.parent_checkpoint_id == id2

        ckpt2 = await manager.get_checkpoint(id2)
        assert ckpt2.parent_checkpoint_id == id1

        ckpt1 = await manager.get_checkpoint(id1)
        assert ckpt1.parent_checkpoint_id is None

    @pytest.mark.asyncio
    async def test_get_agent_state(self):
        """Test getting agent state from checkpoint"""
        manager = CheckpointManager()

        agent = MockAgent(
            agent_id="agent_1",
            user_id="user_1",
            status="waiting",
            collected_fields={"name": "Bob"},
            execution_state={"progress": 75},
        )

        checkpoint_id = await manager.save_checkpoint(agent)
        state = await manager.get_agent_state(checkpoint_id)

        assert state is not None
        assert state["status"] == "waiting"
        assert state["collected_fields"]["name"] == "Bob"
        assert state["execution_state"]["progress"] == 75

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """Test listing checkpoints"""
        manager = CheckpointManager()

        agent = MockAgent(agent_id="agent_1", user_id="user_1")

        for _ in range(5):
            await manager.save_checkpoint(agent)

        checkpoints = await manager.list_checkpoints("agent_1")
        assert len(checkpoints) == 5

    @pytest.mark.asyncio
    async def test_compare_checkpoints(self):
        """Test comparing checkpoints"""
        manager = CheckpointManager()

        agent = MockAgent(
            agent_id="agent_1",
            user_id="user_1",
            status="waiting",
            collected_fields={"name": "Alice"},
        )

        id1 = await manager.save_checkpoint(agent)

        # Modify agent
        agent.status = "running"
        agent.collected_fields["email"] = "alice@example.com"

        id2 = await manager.save_checkpoint(agent)

        diff = await manager.compare_checkpoints(id1, id2)
        assert diff is not None
        assert diff.status_changed is True
        assert "email" in diff.fields_added

    @pytest.mark.asyncio
    async def test_clear_agent_history(self):
        """Test clearing agent history"""
        manager = CheckpointManager()

        agent = MockAgent(agent_id="agent_1", user_id="user_1")

        for _ in range(3):
            await manager.save_checkpoint(agent)

        count = await manager.clear_agent_history("agent_1")
        assert count == 3

        checkpoints = await manager.list_checkpoints("agent_1")
        assert len(checkpoints) == 0

    @pytest.mark.asyncio
    async def test_branching(self):
        """Test creating branches"""
        manager = CheckpointManager()

        agent = MockAgent(agent_id="agent_1", user_id="user_1")

        # Create main line
        id1 = await manager.save_checkpoint(agent)
        id2 = await manager.save_checkpoint(agent)

        # Create branch from id1
        manager.set_parent_checkpoint("agent_1", id1)
        id_branch = await manager.save_checkpoint(agent, branch_label="alternative")

        ckpt_branch = await manager.get_checkpoint(id_branch)
        assert ckpt_branch.parent_checkpoint_id == id1
        assert ckpt_branch.branch_label == "alternative"

        # Get tree
        tree = await manager.get_checkpoint_tree("agent_1")
        branches = tree.get_branches(id1)
        assert len(branches) == 2  # id2 and id_branch

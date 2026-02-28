"""Tests for onevalet.checkpoint.models

Tests cover:
- Checkpoint: serialization (to_dict, from_dict, to_json, from_json), generate_id
- CheckpointMetadata: from_checkpoint, to_dict, message_preview truncation
- CheckpointTree: add_checkpoint, get_path_to_root, get_branches, get_leaf_nodes, get_depth
- CheckpointDiff: compute, has_changes
"""

import json
import pytest
from datetime import datetime

from onevalet.checkpoint.models import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTree,
    CheckpointDiff,
)


def _make_checkpoint(
    id="ckpt_001",
    agent_id="agent_1",
    agent_type="TestAgent",
    user_id="user_1",
    status="collecting",
    collected_fields=None,
    execution_state=None,
    context=None,
    message=None,
    result=None,
    parent_checkpoint_id=None,
    branch_label=None,
    timestamp=None,
):
    return Checkpoint(
        id=id,
        agent_id=agent_id,
        agent_type=agent_type,
        user_id=user_id,
        status=status,
        collected_fields=collected_fields or {},
        execution_state=execution_state or {},
        context=context or {},
        message=message,
        result=result,
        parent_checkpoint_id=parent_checkpoint_id,
        branch_label=branch_label,
        timestamp=timestamp or datetime(2025, 1, 15, 10, 30, 0),
    )


class TestCheckpointGenerateId:
    def test_starts_with_ckpt_prefix(self):
        id = Checkpoint.generate_id()
        assert id.startswith("ckpt_")

    def test_unique_ids(self):
        ids = {Checkpoint.generate_id() for _ in range(100)}
        assert len(ids) == 100

    def test_has_12_hex_chars_after_prefix(self):
        id = Checkpoint.generate_id()
        hex_part = id[len("ckpt_"):]
        assert len(hex_part) == 12
        int(hex_part, 16)  # should not raise


class TestCheckpointSerialization:
    def test_to_dict_basic_fields(self):
        cp = _make_checkpoint()
        d = cp.to_dict()
        assert d["id"] == "ckpt_001"
        assert d["agent_id"] == "agent_1"
        assert d["agent_type"] == "TestAgent"
        assert d["user_id"] == "user_1"
        assert d["status"] == "collecting"

    def test_to_dict_timestamp_is_iso(self):
        cp = _make_checkpoint(timestamp=datetime(2025, 6, 15, 12, 0, 0))
        d = cp.to_dict()
        assert d["timestamp"] == "2025-06-15T12:00:00"

    def test_to_dict_collected_fields(self):
        cp = _make_checkpoint(collected_fields={"name": "Alice", "age": 30})
        d = cp.to_dict()
        assert d["collected_fields"] == {"name": "Alice", "age": 30}

    def test_to_dict_optional_fields_none(self):
        cp = _make_checkpoint()
        d = cp.to_dict()
        assert d["message"] is None
        assert d["result"] is None
        assert d["parent_checkpoint_id"] is None
        assert d["branch_label"] is None

    def test_roundtrip_dict(self):
        cp = _make_checkpoint(
            collected_fields={"email": "a@b.com"},
            execution_state={"step": 2},
            context={"session_id": "s1"},
            message={"content": "hello"},
            result={"response": "world"},
            parent_checkpoint_id="ckpt_000",
            branch_label="retry",
        )
        d = cp.to_dict()
        restored = Checkpoint.from_dict(d)
        assert restored.id == cp.id
        assert restored.agent_id == cp.agent_id
        assert restored.status == cp.status
        assert restored.collected_fields == cp.collected_fields
        assert restored.execution_state == cp.execution_state
        assert restored.context == cp.context
        assert restored.message == cp.message
        assert restored.result == cp.result
        assert restored.parent_checkpoint_id == cp.parent_checkpoint_id
        assert restored.branch_label == cp.branch_label
        assert restored.timestamp == cp.timestamp

    def test_roundtrip_json(self):
        cp = _make_checkpoint(collected_fields={"x": 1})
        json_str = cp.to_json()
        restored = Checkpoint.from_json(json_str)
        assert restored.id == cp.id
        assert restored.collected_fields == {"x": 1}

    def test_to_json_is_valid_json(self):
        cp = _make_checkpoint()
        parsed = json.loads(cp.to_json())
        assert parsed["id"] == "ckpt_001"

    def test_from_dict_defaults(self):
        minimal = {
            "id": "ckpt_x",
            "agent_id": "a1",
            "agent_type": "T",
            "user_id": "u1",
            "status": "idle",
            "timestamp": "2025-01-01T00:00:00",
        }
        cp = Checkpoint.from_dict(minimal)
        assert cp.collected_fields == {}
        assert cp.execution_state == {}
        assert cp.context == {}
        assert cp.message_history == []
        assert cp.parent_checkpoint_id is None
        assert cp.version == 1


class TestCheckpointMetadata:
    def test_from_checkpoint_basic(self):
        cp = _make_checkpoint()
        meta = CheckpointMetadata.from_checkpoint(cp)
        assert meta.id == cp.id
        assert meta.agent_id == cp.agent_id
        assert meta.agent_type == cp.agent_type
        assert meta.user_id == cp.user_id
        assert meta.status == cp.status
        assert meta.timestamp == cp.timestamp

    def test_from_checkpoint_fields_count(self):
        cp = _make_checkpoint(collected_fields={"a": 1, "b": 2, "c": 3})
        meta = CheckpointMetadata.from_checkpoint(cp)
        assert meta.fields_count == 3

    def test_from_checkpoint_no_message(self):
        cp = _make_checkpoint(message=None)
        meta = CheckpointMetadata.from_checkpoint(cp)
        assert meta.message_preview is None

    def test_from_checkpoint_short_message(self):
        cp = _make_checkpoint(message={"content": "Hello!"})
        meta = CheckpointMetadata.from_checkpoint(cp)
        assert meta.message_preview == "Hello!"

    def test_from_checkpoint_long_message_truncated(self):
        long_content = "x" * 200
        cp = _make_checkpoint(message={"content": long_content})
        meta = CheckpointMetadata.from_checkpoint(cp)
        assert len(meta.message_preview) == 100

    def test_from_checkpoint_non_string_content(self):
        cp = _make_checkpoint(message={"content": 42})
        meta = CheckpointMetadata.from_checkpoint(cp)
        assert meta.message_preview is None

    def test_from_checkpoint_parent_and_branch(self):
        cp = _make_checkpoint(parent_checkpoint_id="ckpt_000", branch_label="alt")
        meta = CheckpointMetadata.from_checkpoint(cp)
        assert meta.parent_checkpoint_id == "ckpt_000"
        assert meta.branch_label == "alt"

    def test_to_dict(self):
        cp = _make_checkpoint()
        meta = CheckpointMetadata.from_checkpoint(cp)
        d = meta.to_dict()
        assert d["id"] == "ckpt_001"
        assert "timestamp" in d
        assert "fields_count" in d


class TestCheckpointTree:
    def _build_linear_tree(self):
        """Build a simple 3-node linear chain: root -> mid -> leaf"""
        root = _make_checkpoint(id="c1", timestamp=datetime(2025, 1, 1, 0, 0))
        mid = _make_checkpoint(id="c2", parent_checkpoint_id="c1", timestamp=datetime(2025, 1, 1, 1, 0))
        leaf = _make_checkpoint(id="c3", parent_checkpoint_id="c2", timestamp=datetime(2025, 1, 1, 2, 0))
        tree = CheckpointTree(root_id="c1")
        tree.add_checkpoint(root)
        tree.add_checkpoint(mid)
        tree.add_checkpoint(leaf)
        return tree

    def test_add_checkpoint(self):
        tree = CheckpointTree(root_id="c1")
        cp = _make_checkpoint(id="c1")
        tree.add_checkpoint(cp)
        assert "c1" in tree.nodes

    def test_children_tracked(self):
        tree = self._build_linear_tree()
        assert "c2" in tree.children["c1"]
        assert "c3" in tree.children["c2"]

    def test_root_has_no_parent_in_children(self):
        tree = self._build_linear_tree()
        # Root has no parent, so no key for root's parent
        assert None not in tree.children

    def test_get_path_to_root(self):
        tree = self._build_linear_tree()
        path = tree.get_path_to_root("c3")
        assert path == ["c3", "c2", "c1"]

    def test_get_path_to_root_from_root(self):
        tree = self._build_linear_tree()
        path = tree.get_path_to_root("c1")
        assert path == ["c1"]

    def test_get_branches(self):
        tree = self._build_linear_tree()
        assert tree.get_branches("c1") == ["c2"]
        assert tree.get_branches("c3") == []

    def test_get_leaf_nodes(self):
        tree = self._build_linear_tree()
        leaves = tree.get_leaf_nodes()
        assert leaves == ["c3"]

    def test_get_leaf_nodes_with_branching(self):
        tree = self._build_linear_tree()
        branch = _make_checkpoint(id="c4", parent_checkpoint_id="c1", timestamp=datetime(2025, 1, 1, 3, 0))
        tree.add_checkpoint(branch)
        leaves = set(tree.get_leaf_nodes())
        assert leaves == {"c3", "c4"}

    def test_get_depth(self):
        tree = self._build_linear_tree()
        assert tree.get_depth("c1") == 0
        assert tree.get_depth("c2") == 1
        assert tree.get_depth("c3") == 2

    def test_to_dict(self):
        tree = self._build_linear_tree()
        d = tree.to_dict()
        assert d["root_id"] == "c1"
        assert "c1" in d["nodes"]
        assert "c2" in d["nodes"]
        assert "children" in d


class TestCheckpointDiff:
    def test_no_changes(self):
        cp1 = _make_checkpoint(id="c1", status="collecting", collected_fields={"a": 1})
        cp2 = _make_checkpoint(id="c2", status="collecting", collected_fields={"a": 1})
        diff = CheckpointDiff.compute(cp1, cp2)
        assert not diff.has_changes()

    def test_status_changed(self):
        cp1 = _make_checkpoint(id="c1", status="collecting")
        cp2 = _make_checkpoint(id="c2", status="completed")
        diff = CheckpointDiff.compute(cp1, cp2)
        assert diff.status_changed is True
        assert diff.old_status == "collecting"
        assert diff.new_status == "completed"

    def test_fields_added(self):
        cp1 = _make_checkpoint(id="c1", collected_fields={})
        cp2 = _make_checkpoint(id="c2", collected_fields={"name": "Alice"})
        diff = CheckpointDiff.compute(cp1, cp2)
        assert diff.fields_added == {"name": "Alice"}
        assert diff.has_changes()

    def test_fields_removed(self):
        cp1 = _make_checkpoint(id="c1", collected_fields={"name": "Alice"})
        cp2 = _make_checkpoint(id="c2", collected_fields={})
        diff = CheckpointDiff.compute(cp1, cp2)
        assert "name" in diff.fields_removed

    def test_fields_modified(self):
        cp1 = _make_checkpoint(id="c1", collected_fields={"name": "Alice"})
        cp2 = _make_checkpoint(id="c2", collected_fields={"name": "Bob"})
        diff = CheckpointDiff.compute(cp1, cp2)
        assert "name" in diff.fields_modified
        assert diff.fields_modified["name"] == {"old": "Alice", "new": "Bob"}

    def test_execution_state_changed(self):
        cp1 = _make_checkpoint(id="c1", execution_state={"step": 1})
        cp2 = _make_checkpoint(id="c2", execution_state={"step": 2})
        diff = CheckpointDiff.compute(cp1, cp2)
        assert diff.execution_state_changed is True

    def test_from_to_ids(self):
        cp1 = _make_checkpoint(id="c1")
        cp2 = _make_checkpoint(id="c2")
        diff = CheckpointDiff.compute(cp1, cp2)
        assert diff.from_checkpoint_id == "c1"
        assert diff.to_checkpoint_id == "c2"

"""
OneValet Checkpoint Models - Data structures for checkpoint system

This module defines:
- Checkpoint: Full state snapshot at a point in time
- CheckpointMetadata: Lightweight metadata for listing
- CheckpointTree: Tree structure for branching conversations
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
import json


@dataclass
class Checkpoint:
    """
    Complete state snapshot at a point in time.

    Captures:
    - Agent status and state layers
    - Message that triggered this checkpoint
    - Result from the state handler
    - Link to parent checkpoint (for branching)
    """
    # Identity
    id: str
    agent_id: str
    agent_type: str
    user_id: str

    # State snapshot
    status: str  # AgentStatus value
    collected_fields: Dict[str, Any] = field(default_factory=dict)
    execution_state: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    # Message that triggered this checkpoint
    message: Optional[Dict[str, Any]] = None  # Serialized Message

    # Result from state handler
    result: Optional[Dict[str, Any]] = None  # Serialized AgentResult

    # Message history (for replay)
    message_history: List[Dict[str, Any]] = field(default_factory=list)

    # Tree structure
    parent_checkpoint_id: Optional[str] = None  # Forms a tree
    branch_label: Optional[str] = None  # Optional label for branch

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1  # Schema version for forward compatibility

    @classmethod
    def generate_id(cls) -> str:
        """Generate a unique checkpoint ID"""
        return f"ckpt_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint to dictionary"""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "user_id": self.user_id,
            "status": self.status,
            "collected_fields": self.collected_fields,
            "execution_state": self.execution_state,
            "context": self.context,
            "message": self.message,
            "result": self.result,
            "message_history": self.message_history,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "branch_label": self.branch_label,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Deserialize checkpoint from dictionary"""
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            user_id=data["user_id"],
            status=data["status"],
            collected_fields=data.get("collected_fields", {}),
            execution_state=data.get("execution_state", {}),
            context=data.get("context", {}),
            message=data.get("message"),
            result=data.get("result"),
            message_history=data.get("message_history", []),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            branch_label=data.get("branch_label"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version", 1),
        )

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "Checkpoint":
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class CheckpointMetadata:
    """
    Lightweight metadata for checkpoint listing.

    Used for displaying checkpoint lists without loading full state.
    """
    id: str
    agent_id: str
    agent_type: str
    user_id: str
    status: str
    timestamp: datetime
    parent_checkpoint_id: Optional[str] = None
    branch_label: Optional[str] = None

    # Summary info
    fields_count: int = 0
    message_preview: Optional[str] = None  # First 100 chars of message

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint) -> "CheckpointMetadata":
        """Create metadata from full checkpoint"""
        message_preview = None
        if checkpoint.message:
            content = checkpoint.message.get("content", "")
            if isinstance(content, str):
                message_preview = content[:100] if len(content) > 100 else content

        return cls(
            id=checkpoint.id,
            agent_id=checkpoint.agent_id,
            agent_type=checkpoint.agent_type,
            user_id=checkpoint.user_id,
            status=checkpoint.status,
            timestamp=checkpoint.timestamp,
            parent_checkpoint_id=checkpoint.parent_checkpoint_id,
            branch_label=checkpoint.branch_label,
            fields_count=len(checkpoint.collected_fields),
            message_preview=message_preview,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "user_id": self.user_id,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "branch_label": self.branch_label,
            "fields_count": self.fields_count,
            "message_preview": self.message_preview,
        }


@dataclass
class CheckpointTree:
    """
    Tree structure for branching conversations.

    Represents the full tree of checkpoints for an agent,
    useful for visualizing conversation branches.
    """
    root_id: str
    nodes: Dict[str, CheckpointMetadata] = field(default_factory=dict)
    children: Dict[str, List[str]] = field(default_factory=dict)  # parent_id -> child_ids

    def add_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Add a checkpoint to the tree"""
        metadata = CheckpointMetadata.from_checkpoint(checkpoint)
        self.nodes[checkpoint.id] = metadata

        if checkpoint.parent_checkpoint_id:
            if checkpoint.parent_checkpoint_id not in self.children:
                self.children[checkpoint.parent_checkpoint_id] = []
            self.children[checkpoint.parent_checkpoint_id].append(checkpoint.id)

    def get_path_to_root(self, checkpoint_id: str) -> List[str]:
        """Get the path from a checkpoint to the root"""
        path = []
        current_id = checkpoint_id

        while current_id:
            path.append(current_id)
            metadata = self.nodes.get(current_id)
            if metadata:
                current_id = metadata.parent_checkpoint_id
            else:
                break

        return path

    def get_branches(self, checkpoint_id: str) -> List[str]:
        """Get all branches from a checkpoint"""
        return self.children.get(checkpoint_id, [])

    def get_leaf_nodes(self) -> List[str]:
        """Get all leaf nodes (checkpoints with no children)"""
        all_parents = set(self.children.keys())
        all_nodes = set(self.nodes.keys())
        return list(all_nodes - all_parents)

    def get_depth(self, checkpoint_id: str) -> int:
        """Get depth of a checkpoint in the tree"""
        return len(self.get_path_to_root(checkpoint_id)) - 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree to dictionary"""
        return {
            "root_id": self.root_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "children": self.children,
        }


@dataclass
class CheckpointDiff:
    """
    Difference between two checkpoints.

    Useful for understanding what changed between states.
    """
    from_checkpoint_id: str
    to_checkpoint_id: str

    # Status change
    status_changed: bool = False
    old_status: Optional[str] = None
    new_status: Optional[str] = None

    # Fields changes
    fields_added: Dict[str, Any] = field(default_factory=dict)
    fields_removed: List[str] = field(default_factory=list)
    fields_modified: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # key -> {old, new}

    # Execution state changes
    execution_state_changed: bool = False

    @classmethod
    def compute(
        cls,
        from_checkpoint: Checkpoint,
        to_checkpoint: Checkpoint
    ) -> "CheckpointDiff":
        """Compute diff between two checkpoints"""
        diff = cls(
            from_checkpoint_id=from_checkpoint.id,
            to_checkpoint_id=to_checkpoint.id,
        )

        # Status change
        if from_checkpoint.status != to_checkpoint.status:
            diff.status_changed = True
            diff.old_status = from_checkpoint.status
            diff.new_status = to_checkpoint.status

        # Fields changes
        from_fields = set(from_checkpoint.collected_fields.keys())
        to_fields = set(to_checkpoint.collected_fields.keys())

        # Added fields
        for key in to_fields - from_fields:
            diff.fields_added[key] = to_checkpoint.collected_fields[key]

        # Removed fields
        diff.fields_removed = list(from_fields - to_fields)

        # Modified fields
        for key in from_fields & to_fields:
            old_value = from_checkpoint.collected_fields[key]
            new_value = to_checkpoint.collected_fields[key]
            if old_value != new_value:
                diff.fields_modified[key] = {"old": old_value, "new": new_value}

        # Execution state changes
        diff.execution_state_changed = (
            from_checkpoint.execution_state != to_checkpoint.execution_state
        )

        return diff

    def has_changes(self) -> bool:
        """Check if there are any changes"""
        return (
            self.status_changed or
            bool(self.fields_added) or
            bool(self.fields_removed) or
            bool(self.fields_modified) or
            self.execution_state_changed
        )

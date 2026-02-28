"""
OneValet Checkpoint System - Time-travel debugging and state recovery

This module provides automatic checkpointing at every state transition
for debugging and replay capabilities.

Key Features:
- Auto-save: Checkpoint saved after each state transition
- Time-travel: Restore agent to any checkpoint
- Branching: Replay from checkpoint with different input
- Tree structure: Track conversation branches

Storage Backends:
- Memory (testing/development)
- SQLite (local development)
- PostgreSQL (production)

Example usage:
    from onevalet.checkpoint import CheckpointManager, MemoryStorage

    # Create manager with memory storage
    manager = CheckpointManager(storage=MemoryStorage())

    # Save checkpoint
    checkpoint_id = await manager.save_checkpoint(agent, message, result)

    # List checkpoints for an agent
    checkpoints = await manager.list_checkpoints(agent_id)

    # Restore agent to checkpoint
    agent = await manager.restore_agent(checkpoint_id)

    # Replay with different input
    result = await manager.replay_from(checkpoint_id, new_message)
"""

from .models import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTree,
)

from .storage import (
    CheckpointStorage,
    MemoryStorage,
    # SQLiteStorage,  # Available if sqlite installed
)

from .postgres_storage import PostgreSQLStorage

from .manager import (
    CheckpointManager,
    CheckpointError,
)

__all__ = [
    # Models
    "Checkpoint",
    "CheckpointMetadata",
    "CheckpointTree",
    # Storage
    "CheckpointStorage",
    "MemoryStorage",
    "PostgreSQLStorage",
    # Manager
    "CheckpointManager",
    "CheckpointError",
]

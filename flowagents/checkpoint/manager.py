"""
FlowAgent Checkpoint Manager - High-level checkpoint operations

This module provides the main CheckpointManager class for:
- Saving checkpoints after state transitions
- Restoring agents to previous states
- Replaying from checkpoints with different input
- Browsing and comparing checkpoints
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Protocol, TypeVar, Generic
import copy

from .models import Checkpoint, CheckpointMetadata, CheckpointTree, CheckpointDiff
from .storage import CheckpointStorage, MemoryStorage


class CheckpointError(Exception):
    """Raised when checkpoint operations fail"""
    pass


class AgentProtocol(Protocol):
    """Protocol for agent that can be checkpointed"""
    agent_id: str
    user_id: str
    status: Any  # AgentStatus
    collected_fields: Dict[str, Any]
    execution_state: Dict[str, Any]
    context: Dict[str, Any]

    def get_message_history(self) -> List[Any]: ...


class CheckpointManager:
    """
    Manages checkpoint operations for agents.

    Provides:
    - Automatic checkpoint saving after state transitions
    - Agent restoration from checkpoints
    - Replay from checkpoint with different input
    - Checkpoint browsing and comparison

    Example usage:
        manager = CheckpointManager(storage=MemoryStorage())

        # Save checkpoint
        checkpoint_id = await manager.save_checkpoint(agent, message, result)

        # List checkpoints
        checkpoints = await manager.list_checkpoints(agent_id)

        # Restore agent
        state = await manager.get_agent_state(checkpoint_id)

        # Compare checkpoints
        diff = await manager.compare_checkpoints(id1, id2)
    """

    def __init__(
        self,
        storage: Optional[CheckpointStorage] = None,
        auto_save: bool = True
    ):
        """
        Initialize checkpoint manager.

        Args:
            storage: Storage backend (defaults to MemoryStorage)
            auto_save: Whether to automatically save on each state transition
        """
        self.storage = storage or MemoryStorage()
        self.auto_save = auto_save
        self._last_checkpoint: Dict[str, str] = {}  # agent_id -> checkpoint_id

    async def save_checkpoint(
        self,
        agent: AgentProtocol,
        message: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        branch_label: Optional[str] = None
    ) -> str:
        """
        Save a checkpoint for an agent.

        Args:
            agent: Agent to checkpoint
            message: Message that triggered this state
            result: Result from state handler
            branch_label: Optional label for branching

        Returns:
            Checkpoint ID
        """
        # Get parent checkpoint (if any)
        parent_id = self._last_checkpoint.get(agent.agent_id)

        # Get message history
        try:
            message_history = [
                self._serialize_message(m)
                for m in agent.get_message_history()
            ]
        except Exception:
            message_history = []

        # Create checkpoint
        checkpoint = Checkpoint(
            id=Checkpoint.generate_id(),
            agent_id=agent.agent_id,
            agent_type=agent.__class__.__name__,
            user_id=agent.user_id,
            status=str(agent.status.value) if hasattr(agent.status, 'value') else str(agent.status),
            collected_fields=copy.deepcopy(agent.collected_fields),
            execution_state=copy.deepcopy(agent.execution_state),
            context=copy.deepcopy(agent.context),
            message=message,
            result=result,
            message_history=message_history,
            parent_checkpoint_id=parent_id,
            branch_label=branch_label,
            timestamp=datetime.now(),
        )

        # Save to storage
        await self.storage.save(checkpoint)

        # Update last checkpoint
        self._last_checkpoint[agent.agent_id] = checkpoint.id

        return checkpoint.id

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Get a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint or None
        """
        return await self.storage.get(checkpoint_id)

    async def get_agent_state(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent state from a checkpoint.

        Returns a dictionary that can be used to restore an agent.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Agent state dictionary or None
        """
        checkpoint = await self.storage.get(checkpoint_id)
        if not checkpoint:
            return None

        return {
            "agent_id": checkpoint.agent_id,
            "agent_type": checkpoint.agent_type,
            "user_id": checkpoint.user_id,
            "status": checkpoint.status,
            "collected_fields": checkpoint.collected_fields,
            "execution_state": checkpoint.execution_state,
            "context": checkpoint.context,
            "message_history": checkpoint.message_history,
        }

    async def restore_agent(
        self,
        checkpoint_id: str,
        agent_factory: Any
    ) -> Any:
        """
        Restore an agent from a checkpoint.

        Args:
            checkpoint_id: Checkpoint to restore from
            agent_factory: Factory for creating agent instances

        Returns:
            Restored agent instance
        """
        checkpoint = await self.storage.get(checkpoint_id)
        if not checkpoint:
            raise CheckpointError(f"Checkpoint not found: {checkpoint_id}")

        # Create new agent instance
        agent = await agent_factory.create_agent(
            agent_type=checkpoint.agent_type,
            user_id=checkpoint.user_id,
            context_hints=checkpoint.context
        )

        # Restore state
        agent.collected_fields = copy.deepcopy(checkpoint.collected_fields)
        agent.execution_state = copy.deepcopy(checkpoint.execution_state)

        # Restore status
        if hasattr(agent, 'status') and hasattr(agent.status, '__class__'):
            try:
                agent.status = agent.status.__class__(checkpoint.status)
            except Exception:
                pass

        # Update last checkpoint reference
        self._last_checkpoint[agent.agent_id] = checkpoint_id

        return agent

    async def replay_from(
        self,
        checkpoint_id: str,
        new_message: Dict[str, Any],
        agent_factory: Any,
        branch_label: Optional[str] = None
    ) -> Any:
        """
        Replay from a checkpoint with different input.

        This creates a new branch from the checkpoint.

        Args:
            checkpoint_id: Checkpoint to replay from
            new_message: New message to process
            agent_factory: Factory for creating agent instances
            branch_label: Label for the new branch

        Returns:
            Result from agent processing
        """
        # Restore agent to checkpoint state
        agent = await self.restore_agent(checkpoint_id, agent_factory)

        # Process new message
        # The agent will create its own checkpoint after processing
        from ..message import Message
        msg = Message(**new_message) if isinstance(new_message, dict) else new_message
        result = await agent.reply(msg)

        return result

    async def list_checkpoints(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[CheckpointMetadata]:
        """
        List checkpoints for an agent.

        Args:
            agent_id: Agent ID
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of checkpoint metadata
        """
        return await self.storage.list_by_agent(agent_id, limit, offset)

    async def list_user_checkpoints(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[CheckpointMetadata]:
        """
        List checkpoints for a user.

        Args:
            user_id: User ID
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of checkpoint metadata
        """
        return await self.storage.list_by_user(user_id, limit, offset)

    async def get_checkpoint_tree(self, agent_id: str) -> Optional[CheckpointTree]:
        """
        Get the full checkpoint tree for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            CheckpointTree or None
        """
        return await self.storage.get_tree(agent_id)

    async def get_latest_checkpoint(self, agent_id: str) -> Optional[Checkpoint]:
        """
        Get the most recent checkpoint for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Latest checkpoint or None
        """
        return await self.storage.get_latest(agent_id)

    async def compare_checkpoints(
        self,
        from_id: str,
        to_id: str
    ) -> Optional[CheckpointDiff]:
        """
        Compare two checkpoints.

        Args:
            from_id: First checkpoint ID
            to_id: Second checkpoint ID

        Returns:
            CheckpointDiff or None if either checkpoint not found
        """
        from_checkpoint = await self.storage.get(from_id)
        to_checkpoint = await self.storage.get(to_id)

        if not from_checkpoint or not to_checkpoint:
            return None

        return CheckpointDiff.compute(from_checkpoint, to_checkpoint)

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint to delete

        Returns:
            True if deleted
        """
        return await self.storage.delete(checkpoint_id)

    async def clear_agent_history(self, agent_id: str) -> int:
        """
        Delete all checkpoints for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Number of checkpoints deleted
        """
        if agent_id in self._last_checkpoint:
            del self._last_checkpoint[agent_id]
        return await self.storage.clear_agent(agent_id)

    async def clear_user_history(self, user_id: str) -> int:
        """
        Delete all checkpoints for a user.

        Args:
            user_id: User ID

        Returns:
            Number of checkpoints deleted
        """
        return await self.storage.clear_user(user_id)

    def _serialize_message(self, message: Any) -> Dict[str, Any]:
        """Serialize a message to dictionary"""
        if hasattr(message, 'to_dict'):
            return message.to_dict()
        elif hasattr(message, '__dict__'):
            return dict(message.__dict__)
        else:
            return {"content": str(message)}

    def set_parent_checkpoint(self, agent_id: str, checkpoint_id: str) -> None:
        """
        Manually set the parent checkpoint for next save.

        Used when creating branches.

        Args:
            agent_id: Agent ID
            checkpoint_id: Parent checkpoint ID
        """
        self._last_checkpoint[agent_id] = checkpoint_id

    def get_parent_checkpoint(self, agent_id: str) -> Optional[str]:
        """
        Get the current parent checkpoint for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Parent checkpoint ID or None
        """
        return self._last_checkpoint.get(agent_id)

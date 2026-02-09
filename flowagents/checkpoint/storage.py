"""
FlowAgent Checkpoint Storage - Backend storage implementations

This module provides storage backends for checkpoints:
- MemoryStorage: In-memory storage for testing
- SQLiteStorage: Local SQLite database
- RedisStorage: Redis with TTL support (planned)
- PostgreSQLStorage: PostgreSQL for production (planned)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Protocol
from collections import defaultdict
import json

from .models import Checkpoint, CheckpointMetadata, CheckpointTree


class CheckpointStorage(ABC):
    """
    Abstract base class for checkpoint storage backends.

    All storage backends must implement these methods.
    """

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> str:
        """
        Save a checkpoint.

        Args:
            checkpoint: Checkpoint to save

        Returns:
            Checkpoint ID
        """
        pass

    @abstractmethod
    async def get(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Get a checkpoint by ID.

        Args:
            checkpoint_id: ID of checkpoint to retrieve

        Returns:
            Checkpoint or None if not found
        """
        pass

    @abstractmethod
    async def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_by_agent(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[CheckpointMetadata]:
        """
        List checkpoints for an agent.

        Args:
            agent_id: Agent ID to filter by
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of checkpoint metadata
        """
        pass

    @abstractmethod
    async def list_by_user(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[CheckpointMetadata]:
        """
        List checkpoints for a user.

        Args:
            user_id: User ID to filter by
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of checkpoint metadata
        """
        pass

    @abstractmethod
    async def get_tree(self, agent_id: str) -> Optional[CheckpointTree]:
        """
        Get the full checkpoint tree for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            CheckpointTree or None if no checkpoints
        """
        pass

    @abstractmethod
    async def get_latest(self, agent_id: str) -> Optional[Checkpoint]:
        """
        Get the most recent checkpoint for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Latest checkpoint or None
        """
        pass

    @abstractmethod
    async def clear_agent(self, agent_id: str) -> int:
        """
        Delete all checkpoints for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Number of checkpoints deleted
        """
        pass

    @abstractmethod
    async def clear_user(self, user_id: str) -> int:
        """
        Delete all checkpoints for a user.

        Args:
            user_id: User ID

        Returns:
            Number of checkpoints deleted
        """
        pass


class MemoryStorage(CheckpointStorage):
    """
    In-memory checkpoint storage for testing and development.

    All data is lost when the process exits.
    """

    def __init__(self, max_checkpoints_per_agent: int = 1000):
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._by_agent: Dict[str, List[str]] = defaultdict(list)
        self._by_user: Dict[str, List[str]] = defaultdict(list)
        self._max_per_agent = max_checkpoints_per_agent

    async def save(self, checkpoint: Checkpoint) -> str:
        """Save checkpoint to memory"""
        self._checkpoints[checkpoint.id] = checkpoint
        self._by_agent[checkpoint.agent_id].append(checkpoint.id)
        self._by_user[checkpoint.user_id].append(checkpoint.id)

        # Enforce max checkpoints per agent
        agent_checkpoints = self._by_agent[checkpoint.agent_id]
        if len(agent_checkpoints) > self._max_per_agent:
            # Remove oldest
            oldest_id = agent_checkpoints.pop(0)
            if oldest_id in self._checkpoints:
                old_checkpoint = self._checkpoints.pop(oldest_id)
                self._by_user[old_checkpoint.user_id].remove(oldest_id)

        return checkpoint.id

    async def get(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint from memory"""
        return self._checkpoints.get(checkpoint_id)

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from memory"""
        if checkpoint_id not in self._checkpoints:
            return False

        checkpoint = self._checkpoints.pop(checkpoint_id)
        self._by_agent[checkpoint.agent_id].remove(checkpoint_id)
        self._by_user[checkpoint.user_id].remove(checkpoint_id)
        return True

    async def list_by_agent(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[CheckpointMetadata]:
        """List checkpoints for agent"""
        checkpoint_ids = self._by_agent.get(agent_id, [])

        # Sort by timestamp (newest first)
        checkpoints = [
            self._checkpoints[cid]
            for cid in checkpoint_ids
            if cid in self._checkpoints
        ]
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)

        # Apply pagination
        paginated = checkpoints[offset:offset + limit]

        return [CheckpointMetadata.from_checkpoint(c) for c in paginated]

    async def list_by_user(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[CheckpointMetadata]:
        """List checkpoints for user"""
        checkpoint_ids = self._by_user.get(user_id, [])

        checkpoints = [
            self._checkpoints[cid]
            for cid in checkpoint_ids
            if cid in self._checkpoints
        ]
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)

        paginated = checkpoints[offset:offset + limit]

        return [CheckpointMetadata.from_checkpoint(c) for c in paginated]

    async def get_tree(self, agent_id: str) -> Optional[CheckpointTree]:
        """Get checkpoint tree for agent"""
        checkpoint_ids = self._by_agent.get(agent_id, [])
        if not checkpoint_ids:
            return None

        checkpoints = [
            self._checkpoints[cid]
            for cid in checkpoint_ids
            if cid in self._checkpoints
        ]

        # Find root (oldest checkpoint)
        checkpoints.sort(key=lambda c: c.timestamp)
        root = checkpoints[0]

        tree = CheckpointTree(root_id=root.id)
        for checkpoint in checkpoints:
            tree.add_checkpoint(checkpoint)

        return tree

    async def get_latest(self, agent_id: str) -> Optional[Checkpoint]:
        """Get latest checkpoint for agent"""
        checkpoint_ids = self._by_agent.get(agent_id, [])
        if not checkpoint_ids:
            return None

        checkpoints = [
            self._checkpoints[cid]
            for cid in checkpoint_ids
            if cid in self._checkpoints
        ]

        if not checkpoints:
            return None

        # Use (timestamp, index) as sort key to handle same timestamp
        return max(enumerate(checkpoints), key=lambda x: (x[1].timestamp, x[0]))[1]

    async def clear_agent(self, agent_id: str) -> int:
        """Clear all checkpoints for agent"""
        checkpoint_ids = self._by_agent.get(agent_id, [])
        count = 0

        for cid in list(checkpoint_ids):
            if await self.delete(cid):
                count += 1

        return count

    async def clear_user(self, user_id: str) -> int:
        """Clear all checkpoints for user"""
        checkpoint_ids = self._by_user.get(user_id, [])
        count = 0

        for cid in list(checkpoint_ids):
            if await self.delete(cid):
                count += 1

        return count

    def clear_all(self) -> None:
        """Clear all checkpoints (for testing)"""
        self._checkpoints.clear()
        self._by_agent.clear()
        self._by_user.clear()


class SQLiteStorage(CheckpointStorage):
    """
    SQLite checkpoint storage for local development.

    Requires aiosqlite package.
    """

    def __init__(self, db_path: str = "checkpoints.db"):
        self.db_path = db_path
        self._initialized = False

    async def _ensure_initialized(self):
        """Initialize database tables if needed"""
        if self._initialized:
            return

        try:
            import aiosqlite
        except ImportError:
            raise ImportError(
                "aiosqlite is required for SQLite storage. "
                "Install it with: pip install aiosqlite"
            )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_id ON checkpoints(agent_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_id ON checkpoints(user_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON checkpoints(timestamp)"
            )
            await db.commit()

        self._initialized = True

    async def save(self, checkpoint: Checkpoint) -> str:
        """Save checkpoint to SQLite"""
        import aiosqlite

        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (id, agent_id, agent_type, user_id, status, data, parent_checkpoint_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.id,
                    checkpoint.agent_id,
                    checkpoint.agent_type,
                    checkpoint.user_id,
                    checkpoint.status,
                    checkpoint.to_json(),
                    checkpoint.parent_checkpoint_id,
                    checkpoint.timestamp.isoformat(),
                )
            )
            await db.commit()

        return checkpoint.id

    async def get(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint from SQLite"""
        import aiosqlite

        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT data FROM checkpoints WHERE id = ?",
                (checkpoint_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return Checkpoint.from_json(row[0])
                return None

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from SQLite"""
        import aiosqlite

        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM checkpoints WHERE id = ?",
                (checkpoint_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def list_by_agent(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[CheckpointMetadata]:
        """List checkpoints for agent from SQLite"""
        import aiosqlite

        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT data FROM checkpoints
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (agent_id, limit, offset)
            ) as cursor:
                rows = await cursor.fetchall()
                checkpoints = [Checkpoint.from_json(row[0]) for row in rows]
                return [CheckpointMetadata.from_checkpoint(c) for c in checkpoints]

    async def list_by_user(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[CheckpointMetadata]:
        """List checkpoints for user from SQLite"""
        import aiosqlite

        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT data FROM checkpoints
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, limit, offset)
            ) as cursor:
                rows = await cursor.fetchall()
                checkpoints = [Checkpoint.from_json(row[0]) for row in rows]
                return [CheckpointMetadata.from_checkpoint(c) for c in checkpoints]

    async def get_tree(self, agent_id: str) -> Optional[CheckpointTree]:
        """Get checkpoint tree from SQLite"""
        import aiosqlite

        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT data FROM checkpoints
                WHERE agent_id = ?
                ORDER BY timestamp ASC
                """,
                (agent_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                if not rows:
                    return None

                checkpoints = [Checkpoint.from_json(row[0]) for row in rows]
                tree = CheckpointTree(root_id=checkpoints[0].id)
                for checkpoint in checkpoints:
                    tree.add_checkpoint(checkpoint)
                return tree

    async def get_latest(self, agent_id: str) -> Optional[Checkpoint]:
        """Get latest checkpoint from SQLite"""
        import aiosqlite

        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT data FROM checkpoints
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (agent_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return Checkpoint.from_json(row[0])
                return None

    async def clear_agent(self, agent_id: str) -> int:
        """Clear all checkpoints for agent"""
        import aiosqlite

        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM checkpoints WHERE agent_id = ?",
                (agent_id,)
            )
            await db.commit()
            return cursor.rowcount

    async def clear_user(self, user_id: str) -> int:
        """Clear all checkpoints for user"""
        import aiosqlite

        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM checkpoints WHERE user_id = ?",
                (user_id,)
            )
            await db.commit()
            return cursor.rowcount

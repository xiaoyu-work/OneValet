"""
OneValet Agent Pool Manager - Manages agent instances per tenant

This module provides:
- AgentPoolManager: Manages agent lifecycle and storage
- Memory and Redis backends for agent persistence
- Session management with TTL-based cleanup

Tenant isolation:
- Each tenant (user, org, etc.) has isolated agent pools
- Use tenant_id="default" for single-tenant deployments
"""

import json
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

from .models import AgentPoolEntry, SessionConfig

if TYPE_CHECKING:
    from ..standard_agent import StandardAgent

logger = logging.getLogger(__name__)


class PoolBackend(ABC):
    """Abstract base class for pool storage backends"""

    @abstractmethod
    async def save_agent(self, tenant_id: str, entry: AgentPoolEntry) -> None:
        """Save agent entry to storage"""
        pass

    @abstractmethod
    async def get_agent(self, tenant_id: str, agent_id: str) -> Optional[AgentPoolEntry]:
        """Get agent entry from storage"""
        pass

    @abstractmethod
    async def list_agents(self, tenant_id: str) -> List[AgentPoolEntry]:
        """List all agents for a tenant"""
        pass

    @abstractmethod
    async def remove_agent(self, tenant_id: str, agent_id: str) -> None:
        """Remove agent from storage"""
        pass

    @abstractmethod
    async def clear_tenant(self, tenant_id: str) -> None:
        """Clear all agents for a tenant"""
        pass

    @abstractmethod
    async def get_active_tenants(self) -> List[str]:
        """Get list of tenants with active agents"""
        pass

    async def close(self) -> None:
        """Close backend connections. Override in subclasses that need cleanup."""
        pass


class MemoryPoolBackend(PoolBackend):
    """In-memory pool backend for development/testing"""

    def __init__(self):
        self._pools: Dict[str, Dict[str, AgentPoolEntry]] = {}
        self._active_tenants: set = set()

    async def save_agent(self, tenant_id: str, entry: AgentPoolEntry) -> None:
        if tenant_id not in self._pools:
            self._pools[tenant_id] = {}
        self._pools[tenant_id][entry.agent_id] = entry
        self._active_tenants.add(tenant_id)

    async def get_agent(self, tenant_id: str, agent_id: str) -> Optional[AgentPoolEntry]:
        if tenant_id in self._pools:
            return self._pools[tenant_id].get(agent_id)
        return None

    async def list_agents(self, tenant_id: str) -> List[AgentPoolEntry]:
        if tenant_id in self._pools:
            return list(self._pools[tenant_id].values())
        return []

    async def remove_agent(self, tenant_id: str, agent_id: str) -> None:
        if tenant_id in self._pools and agent_id in self._pools[tenant_id]:
            del self._pools[tenant_id][agent_id]
            if not self._pools[tenant_id]:
                del self._pools[tenant_id]
                self._active_tenants.discard(tenant_id)

    async def clear_tenant(self, tenant_id: str) -> None:
        if tenant_id in self._pools:
            del self._pools[tenant_id]
        self._active_tenants.discard(tenant_id)

    async def get_active_tenants(self) -> List[str]:
        return list(self._active_tenants)


class RedisPoolBackend(PoolBackend):
    """Redis pool backend for production with TTL support"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        active_ttl: int = 600,
        session_ttl: int = 86400
    ):
        self._redis_url = redis_url
        self._active_ttl = active_ttl
        self._session_ttl = session_ttl
        self._redis: Optional[Any] = None

    async def _get_redis(self):
        """Lazy initialize Redis connection"""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self._redis_url)
            except ImportError:
                raise ImportError("redis package required for Redis backend. Install with: pip install redis")
        return self._redis

    def _active_key(self, tenant_id: str) -> str:
        """Key for active pool (short TTL)"""
        return f"onevalet:active:{tenant_id}"

    def _session_key(self, tenant_id: str) -> str:
        """Key for session pool (long TTL)"""
        return f"onevalet:session:{tenant_id}"

    async def save_agent(self, tenant_id: str, entry: AgentPoolEntry) -> None:
        r = await self._get_redis()
        entry_json = json.dumps(entry.to_dict())

        # Save to active pool (short TTL)
        await r.hset(self._active_key(tenant_id), entry.agent_id, entry_json)
        await r.expire(self._active_key(tenant_id), self._active_ttl)

        # Save to session pool (long TTL)
        await r.hset(self._session_key(tenant_id), entry.agent_id, entry_json)
        await r.expire(self._session_key(tenant_id), self._session_ttl)

        # Track active tenants
        await r.sadd("onevalet:active_tenants", tenant_id)
        await r.expire("onevalet:active_tenants", self._session_ttl)

    async def get_agent(self, tenant_id: str, agent_id: str) -> Optional[AgentPoolEntry]:
        r = await self._get_redis()

        # Try active pool first
        entry_json = await r.hget(self._active_key(tenant_id), agent_id)

        # Fall back to session pool
        if not entry_json:
            entry_json = await r.hget(self._session_key(tenant_id), agent_id)

        if entry_json:
            return AgentPoolEntry.from_dict(json.loads(entry_json))
        return None

    async def list_agents(self, tenant_id: str) -> List[AgentPoolEntry]:
        r = await self._get_redis()

        # Get from active pool first
        entries = await r.hgetall(self._active_key(tenant_id))

        # If empty, try session pool
        if not entries:
            entries = await r.hgetall(self._session_key(tenant_id))

        return [
            AgentPoolEntry.from_dict(json.loads(v))
            for v in entries.values()
        ]

    async def remove_agent(self, tenant_id: str, agent_id: str) -> None:
        r = await self._get_redis()
        await r.hdel(self._active_key(tenant_id), agent_id)
        await r.hdel(self._session_key(tenant_id), agent_id)

        # Clean up tenant from active list if no more agents
        if not await r.hlen(self._session_key(tenant_id)):
            await r.srem("onevalet:active_tenants", tenant_id)

    async def clear_tenant(self, tenant_id: str) -> None:
        r = await self._get_redis()
        await r.delete(self._active_key(tenant_id))
        await r.delete(self._session_key(tenant_id))
        await r.srem("onevalet:active_tenants", tenant_id)

    async def get_active_tenants(self) -> List[str]:
        r = await self._get_redis()
        tenants = await r.smembers("onevalet:active_tenants")
        return [t.decode() if isinstance(t, bytes) else t for t in tenants]

    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()


class AgentPoolManager:
    """
    Manages agent instances per tenant.

    Features:
    - Multiple backend support (memory, redis)
    - Agent lifecycle management
    - Session persistence with TTL
    - Auto-backup and restore
    - Lazy restoration on demand

    Usage:
        pool = AgentPoolManager(config=SessionConfig(backend="memory"))

        # Add agent (tenant_id defaults to "default" for single-tenant)
        pool.add_agent(agent)  # uses agent.tenant_id

        # Get agent
        agent = pool.get_agent(tenant_id, agent_id)

        # List agents
        agents = pool.list_agents(tenant_id)

        # Remove agent
        pool.remove_agent(tenant_id, agent_id)
    """

    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        backend: Optional[PoolBackend] = None
    ):
        self.config = config or SessionConfig()

        if backend:
            self._backend = backend
        elif self.config.backend == "redis":
            self._backend = RedisPoolBackend(
                redis_url=self.config.redis_url or "redis://localhost:6379",
                active_ttl=self.config.active_ttl_seconds,
                session_ttl=self.config.session_ttl_seconds
            )
        else:
            self._backend = MemoryPoolBackend()

        # In-memory cache for fast access
        self._agents: Dict[str, Dict[str, "StandardAgent"]] = {}
        self._lock = asyncio.Lock()

        # Background task for auto-backup
        self._backup_task: Optional[asyncio.Task] = None

    async def add_agent(
        self,
        agent: "StandardAgent"
    ) -> None:
        """
        Add agent to the pool.

        Args:
            agent: StandardAgent instance (uses agent.tenant_id)
        """
        tenant_id = agent.tenant_id

        # Compute schema version from agent class
        from ..agents.decorator import get_schema_version
        schema_version = get_schema_version(type(agent))

        # Create pool entry from agent
        entry = AgentPoolEntry(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            tenant_id=tenant_id,
            status=agent.status.value,
            collected_fields=agent.collected_fields,
            execution_state=agent.execution_state,
            context=agent.context,
            schema_version=schema_version,
        )

        # Save to backend
        if self.config.enabled:
            await self._backend.save_agent(tenant_id, entry)

        # Cache in memory
        async with self._lock:
            if tenant_id not in self._agents:
                self._agents[tenant_id] = {}
            self._agents[tenant_id][agent.agent_id] = agent

        logger.debug(f"Added agent {agent.agent_id} for tenant {tenant_id}")

    async def get_agent(
        self,
        tenant_id: str,
        agent_id: str
    ) -> Optional["StandardAgent"]:
        """
        Get agent from pool by ID.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier

        Returns:
            StandardAgent instance or None if not found
        """
        # Check memory cache first
        if tenant_id in self._agents and agent_id in self._agents[tenant_id]:
            return self._agents[tenant_id][agent_id]

        # Agent not in memory - cannot restore without factory
        # This is handled by orchestrator which has the agent registry
        return None

    async def get_agent_entry(
        self,
        tenant_id: str,
        agent_id: str
    ) -> Optional[AgentPoolEntry]:
        """
        Get agent entry (metadata) from storage.

        This can be used to check if an agent exists in storage
        even if it's not in memory.
        """
        return await self._backend.get_agent(tenant_id, agent_id)

    async def list_agents(self, tenant_id: str = "default") -> List["StandardAgent"]:
        """
        List all active agents for a tenant.

        Args:
            tenant_id: Tenant identifier (default: "default")

        Returns:
            List of StandardAgent instances
        """
        async with self._lock:
            if tenant_id in self._agents:
                return list(self._agents[tenant_id].values())
            return []

    async def list_agent_entries(self, tenant_id: str = "default") -> List[AgentPoolEntry]:
        """
        List all agent entries from storage.

        This returns entries even if agents are not in memory.
        """
        return await self._backend.list_agents(tenant_id)

    async def update_agent(
        self,
        agent: "StandardAgent"
    ) -> None:
        """
        Update agent state in the pool.

        Args:
            agent: Updated StandardAgent instance (uses agent.tenant_id)
        """
        tenant_id = agent.tenant_id

        # Compute schema version from agent class
        from ..agents.decorator import get_schema_version
        schema_version = get_schema_version(type(agent))

        entry = AgentPoolEntry(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            tenant_id=tenant_id,
            status=agent.status.value,
            last_activity=datetime.now(),
            collected_fields=agent.collected_fields,
            execution_state=agent.execution_state,
            context=agent.context,
            schema_version=schema_version,
        )

        if self.config.enabled:
            await self._backend.save_agent(tenant_id, entry)

        # Update memory cache
        async with self._lock:
            if tenant_id not in self._agents:
                self._agents[tenant_id] = {}
            self._agents[tenant_id][agent.agent_id] = agent

    async def remove_agent(
        self,
        tenant_id: str,
        agent_id: str
    ) -> None:
        """
        Remove agent from pool.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
        """
        await self._backend.remove_agent(tenant_id, agent_id)

        async with self._lock:
            if tenant_id in self._agents and agent_id in self._agents[tenant_id]:
                del self._agents[tenant_id][agent_id]
                if not self._agents[tenant_id]:
                    del self._agents[tenant_id]

        logger.debug(f"Removed agent {agent_id} for tenant {tenant_id}")

    async def clear_tenant(self, tenant_id: str = "default") -> None:
        """Clear all agents for a tenant"""
        await self._backend.clear_tenant(tenant_id)
        async with self._lock:
            if tenant_id in self._agents:
                del self._agents[tenant_id]

    def has_agents_in_memory(self, tenant_id: str = "default") -> bool:
        """Check if tenant has agents loaded in memory"""
        return tenant_id in self._agents and len(self._agents[tenant_id]) > 0

    async def get_active_tenants(self) -> List[str]:
        """Get list of tenants with active agents"""
        return await self._backend.get_active_tenants()

    async def restore_tenant_session(
        self,
        tenant_id: str,
        agent_factory: Callable[[AgentPoolEntry], "StandardAgent"],
        agent_registry: Optional[Any] = None,
    ) -> int:
        """
        Restore all agents for a tenant from storage.

        Args:
            tenant_id: Tenant identifier
            agent_factory: Factory function to create agent from entry
            agent_registry: Optional registry to check schema versions against

        Returns:
            Number of agents restored
        """
        entries = await self._backend.list_agents(tenant_id)

        async with self._lock:
            if tenant_id not in self._agents:
                self._agents[tenant_id] = {}

        restored = 0
        for entry in entries:
            # Version guard: discard agents with stale schema versions
            if agent_registry is not None:
                current_version = agent_registry.get_schema_version(entry.agent_type)
                if current_version is not None and entry.schema_version != current_version:
                    logger.warning(
                        f"Discarded stale agent {entry.agent_id}: schema version mismatch "
                        f"(pool={entry.schema_version}, current={current_version})"
                    )
                    await self._backend.remove_agent(tenant_id, entry.agent_id)
                    continue

            try:
                agent = agent_factory(entry)
                async with self._lock:
                    self._agents[tenant_id][entry.agent_id] = agent
                restored += 1
            except Exception as e:
                logger.error(f"Failed to restore agent {entry.agent_id}: {e}")

        logger.info(f"Restored {restored} agents for tenant {tenant_id}")
        return restored

    async def restore_all_sessions(
        self,
        agent_factory: Callable[[AgentPoolEntry], "StandardAgent"],
        agent_registry: Optional[Any] = None,
    ) -> int:
        """
        Restore all active sessions from storage.

        Called on server startup when auto_restore_on_start is enabled.

        Args:
            agent_factory: Factory function to create agent from entry
            agent_registry: Optional registry to check schema versions against

        Returns:
            Total number of agents restored
        """
        tenants = await self.get_active_tenants()
        total = 0

        for tenant_id in tenants:
            restored = await self.restore_tenant_session(
                tenant_id, agent_factory, agent_registry=agent_registry
            )
            total += restored

        logger.info(f"Restored {total} agents for {len(tenants)} tenants")
        return total

    async def start_auto_backup(self) -> None:
        """Start background auto-backup task"""
        if self._backup_task is not None:
            return

        async def backup_loop():
            while True:
                await asyncio.sleep(self.config.auto_backup_interval_seconds)
                await self._backup_all()

        self._backup_task = asyncio.create_task(backup_loop())
        logger.info("Started auto-backup task")

    async def stop_auto_backup(self) -> None:
        """Stop background auto-backup task"""
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
            self._backup_task = None
            logger.info("Stopped auto-backup task")

    async def _backup_all(self) -> None:
        """Backup all in-memory agents to storage"""
        async with self._lock:
            snapshot = [
                agent
                for agents in self._agents.values()
                for agent in agents.values()
            ]
        for agent in snapshot:
            try:
                await self.update_agent(agent)
            except Exception as e:
                logger.error(f"Failed to backup agent {agent.agent_id}: {e}")

    async def close(self) -> None:
        """Clean up resources"""
        await self.stop_auto_backup()
        await self._backend.close()

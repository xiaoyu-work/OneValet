"""
OneValet Agent Pool Manager - Manages agent instances per tenant

This module provides:
- AgentPoolManager: Manages agent lifecycle and storage
- Memory and PostgreSQL backends for agent persistence
- Session management with TTL-based cleanup

Tenant isolation:
- Each tenant (user, org, etc.) has isolated agent pools
- Use tenant_id="default" for single-tenant deployments
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING

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


class AgentPoolManager:
    """
    Manages agent instances per tenant.

    Uses PostgreSQL when a database is provided, falls back to
    in-memory storage for testing.

    Features:
    - Agent lifecycle management
    - Session persistence with TTL
    - Auto-backup and restore
    - Lazy restoration on demand

    Usage:
        pool = AgentPoolManager(database=db)

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
        backend: Optional[PoolBackend] = None,
        database: Optional[Any] = None,
    ):
        self.config = config or SessionConfig()

        if backend:
            self._backend = backend
        elif database is not None:
            from .postgres_pool import PostgresPoolBackend
            self._backend = PostgresPoolBackend(
                db=database,
                session_ttl=self.config.session_ttl_seconds,
            )
        else:
            self._backend = MemoryPoolBackend()

        # In-memory cache for fast access
        self._agents: Dict[str, Dict[str, "StandardAgent"]] = {}
        self._lock = asyncio.Lock()

        # Background tasks
        self._backup_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

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

    async def cleanup_timed_out_agents(self, timeout_seconds: int = 300) -> List[str]:
        """
        Clean up agents stuck in WAITING states beyond the timeout.

        Args:
            timeout_seconds: Max seconds an agent can remain in WAITING state

        Returns:
            List of timed-out agent IDs
        """
        now = datetime.now()
        timed_out_ids: List[str] = []

        # Snapshot agents under the lock, then release before doing I/O
        async with self._lock:
            snapshot = [
                (tenant_id, agent_id, agent)
                for tenant_id, agents in self._agents.items()
                for agent_id, agent in agents.items()
            ]

        for tenant_id, agent_id, agent in snapshot:
            status_value = agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
            if status_value not in ("waiting_for_input", "waiting_for_approval"):
                continue

            elapsed = (now - agent.last_active).total_seconds()
            if elapsed > timeout_seconds:
                logger.warning(
                    f"Agent {agent_id} timed out after {elapsed:.0f}s in {status_value} "
                    f"(tenant={tenant_id}, timeout={timeout_seconds}s)"
                )
                try:
                    from ..result import AgentStatus
                    agent.transition_to(AgentStatus.ERROR)
                    agent.error_message = f"Timed out after {elapsed:.0f}s in {status_value}"
                except Exception:
                    pass
                await self.remove_agent(tenant_id, agent_id)
                timed_out_ids.append(agent_id)

        if timed_out_ids:
            logger.info(f"Cleaned up {len(timed_out_ids)} timed-out agents: {timed_out_ids}")

        return timed_out_ids

    async def get_waiting_agent_for_session(
        self,
        tenant_id: str,
        session_id: Optional[str] = None,
    ) -> Optional["StandardAgent"]:
        """
        Get the WAITING agent bound to a specific session.

        This provides deterministic session_id -> agent routing for
        agents in WAITING_FOR_INPUT or WAITING_FOR_APPROVAL states.

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier (stored in agent.context or agent.metadata)

        Returns:
            StandardAgent in a WAITING state for the session, or None
        """
        async with self._lock:
            agents = self._agents.get(tenant_id, {})
            for agent in agents.values():
                status_value = agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
                if status_value not in ("waiting_for_input", "waiting_for_approval"):
                    continue

                if session_id is None:
                    # No session filter - return first waiting agent
                    return agent

                # Check agent.context and agent.metadata for session_id match
                agent_session = (
                    getattr(agent, 'context', {}).get('session_id')
                    or getattr(agent, 'metadata', {}).get('session_id')
                )
                if agent_session == session_id:
                    return agent

        return None

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

    async def start_cleanup_loop(self) -> None:
        """Start background cleanup loop for timed-out WAITING agents."""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                await asyncio.sleep(60)  # check every minute
                try:
                    await self.cleanup_timed_out_agents(self.config.waiting_timeout_seconds)
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Started WAITING agent cleanup loop")

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

    async def stop_cleanup_loop(self) -> None:
        """Stop background cleanup loop."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped WAITING agent cleanup loop")

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
        await self.stop_cleanup_loop()
        await self._backend.close()

"""Agent lifecycle mixin — the pool of live StandardAgent instances.

An agent that pauses (waiting for input or approval) stays in the pool so the
next user message can be routed back to it. This module owns that pool: routing
to a waiting agent, constructing new agents, restoring them after a restart,
evicting stale ones, and the user-facing list/cancel/pause/resume operations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..message import Message
from ..result import AgentResult, AgentStatus
from ..standard_agent import StandardAgent
from .models import AgentCallback, AgentPoolEntry, callback_handler

logger = logging.getLogger(__name__)


class AgentLifecycleMixin:
    """Mixin owning the live-agent pool.

    Expects the following on ``self`` (provided by Orchestrator):
    - ``agent_pool``, ``_agent_registry``, ``config``, ``llm_client``
    - ``_audit``, ``_execution_policy``, ``checkpoint_manager``
    - ``message_hub``, ``trigger_engine``
    - ``STALE_AGENT_THRESHOLD_SECONDS``
    - ``_create_callback_invoker()`` (Orchestrator)
    """

    async def _check_pending_agents(
        self,
        tenant_id: str,
        message: str,
        context: Dict[str, Any],
    ) -> Optional[AgentResult]:
        """Check Pool for WAITING agents and route message to them.

        If there are agents in WAITING_FOR_INPUT or WAITING_FOR_APPROVAL state,
        route the user's message to the appropriate agent:
        - If metadata contains target_agent_id, route to that specific agent
        - Otherwise pick the most recently active waiting agent
        - Log a warning when multiple agents are waiting without explicit routing

        Returns:
            AgentResult if a pending agent handled the message, None otherwise.
        """
        agents = await self.agent_pool.list_agents(tenant_id)
        waiting_agents = [
            a
            for a in agents
            if a.status in (AgentStatus.WAITING_FOR_INPUT, AgentStatus.WAITING_FOR_APPROVAL)
        ]

        if not waiting_agents:
            return None

        # Pick the target agent
        metadata = context.get("metadata", {})
        target_agent_id = metadata.get("target_agent_id")

        if target_agent_id:
            agent = next((a for a in waiting_agents if a.agent_id == target_agent_id), None)
            if agent is None:
                logger.warning(f"target_agent_id={target_agent_id} not found among waiting agents")
                return None
        else:
            if len(waiting_agents) > 1:
                logger.warning(
                    f"Multiple waiting agents for tenant={tenant_id} without explicit routing: "
                    f"{[a.agent_id for a in waiting_agents]}. Picking most recently active."
                )
            agent = max(
                waiting_agents,
                key=lambda a: getattr(a, "last_active", 0) or 0,
            )

        reason = "explicit_target" if target_agent_id else "most_recent"
        self._audit.log_route_decision(
            tenant_id=tenant_id,
            target_agent_id=agent.agent_id,
            waiting_agents_count=len(waiting_agents),
            reason=reason,
        )

        try:
            msg = Message(
                name=metadata.get("sender_name", ""),
                content=message,
                role=metadata.get("sender_role", "user"),
                metadata=metadata,
            )
            result = await agent.reply(msg)
            agent.status = result.status

            # Update or remove from pool
            if agent.status in AgentStatus.terminal_states():
                await self.agent_pool.remove_agent(tenant_id, agent.agent_id)
            else:
                await self.agent_pool.update_agent(agent)

            return result
        except Exception as e:
            logger.error(f"Failed to route to pending agent {agent.agent_id}: {e}")
            return AgentResult(
                agent_type=agent.agent_type,
                status=AgentStatus.ERROR,
                error_message=str(e),
                agent_id=agent.agent_id,
            )

    async def create_agent(
        self,
        tenant_id: str,
        agent_type: str,
        context_hints: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[StandardAgent]:
        """
        Create a new agent instance.

        Override to customize agent creation:
        - Inject custom LLM client per tenant
        - Add tenant-specific tools
        - Set custom orchestrator callback

        Args:
            tenant_id: Tenant identifier
            agent_type: Type of agent to create
            context_hints: Hints extracted from message (pre-populates fields)
            context: Full context dict

        Returns:
            New agent instance or None if failed
        """
        if not self._agent_registry:
            logger.error("Cannot create agent: no registry available")
            return None

        try:
            # Enforce max agents per user
            active = await self.agent_pool.list_agents(tenant_id)
            if len(active) >= self.config.max_agents_per_user:
                logger.warning(
                    f"Max agents per user ({self.config.max_agents_per_user}) "
                    f"reached for tenant {tenant_id} "
                    f"(active: {[a.agent_type if hasattr(a, 'agent_type') else str(a) for a in active]})"
                )
                return None

            agent = self._agent_registry.create_agent(
                name=agent_type,
                tenant_id=tenant_id,
                checkpoint_manager=self.checkpoint_manager,
                message_hub=self.message_hub,
                orchestrator_callback=self._create_callback_invoker(tenant_id),
                context_hints=context_hints,
                execution_policy=self._execution_policy,
            )

            if not agent:
                available = self._agent_registry.get_all_agent_names()
                logger.error(
                    f"Agent type not found in registry: {agent_type}. "
                    f"Available agents ({len(available)}): {available}"
                )
                return None

            # Fallback: if agent has no LLM, use orchestrator's
            if not agent.llm_client:
                agent.llm_client = self.llm_client

            if context_hints and hasattr(agent, "set_recalled_memories"):
                recalled_memories = context_hints.get("recalled_memories") or []
                if recalled_memories:
                    agent.set_recalled_memories(recalled_memories)

            # Add to pool
            await self.agent_pool.add_agent(agent)

            logger.debug(f"Created agent {agent.agent_id} of type {agent_type}")
            return agent

        except Exception as e:
            logger.error(f"Failed to create agent {agent_type}: {e}", exc_info=True)
            return None

    @callback_handler("list_agents")
    async def _builtin_list_agents(self, callback: AgentCallback) -> List[Dict[str, Any]]:
        """
        Built-in callback: List all registered agents.

        Returns:
            List of agent info dicts with name, description, etc.
        """
        if not self._agent_registry:
            return []

        result = []
        for name, metadata in self._agent_registry.get_all_agent_metadata().items():
            result.append(
                {
                    "name": name,
                    "description": metadata.description,
                    "capabilities": getattr(metadata, "capabilities", []),
                }
            )
        return result

    @callback_handler("get_agent_config")
    async def _builtin_get_agent_config(self, callback: AgentCallback) -> Optional[Dict[str, Any]]:
        """
        Built-in callback: Get configuration for a specific agent.

        Args (in callback.data):
            agent_name: Name of the agent to look up
        """
        if not self._agent_registry:
            return None

        agent_name = callback.data.get("agent_name")
        if not agent_name:
            return None

        config = self._agent_registry.get_agent_config(agent_name)
        if not config:
            return None

        return {
            "name": config.name,
            "description": config.description,
            "capabilities": getattr(config, "capabilities", []),
            "inputs": [{"name": i.name, "type": i.type} for i in config.inputs],
            "outputs": [{"name": o.name, "type": o.type} for o in config.outputs],
        }

    async def _cleanup_stale_agents(self, tenant_id: str) -> None:
        """Remove completed and stale agents for a tenant at request start.

        This prevents state leakage between requests by:
        1. Immediately removing agents in terminal states (COMPLETED, ERROR,
           CANCELLED) — these are leftovers from previous requests.
        2. Removing non-terminal agents that have been idle beyond the
           stale agent threshold.

        Called at the beginning of ``_execute_message`` before any processing.
        """
        try:
            agents = await self.agent_pool.list_agents(tenant_id)
            if not agents:
                return

            now = datetime.now()
            removed_count = 0

            for agent in agents:
                should_remove = False
                reason = ""

                # 1. Remove agents in terminal states
                if agent.status in AgentStatus.terminal_states():
                    should_remove = True
                    reason = f"terminal state ({agent.status.value})"

                # 2. Remove non-terminal agents that are stale
                elif hasattr(agent, "last_active") and agent.last_active:
                    try:
                        elapsed = (now - agent.last_active).total_seconds()
                    except TypeError:
                        # last_active might be timezone-aware vs naive
                        elapsed = 0
                    if elapsed > self.STALE_AGENT_THRESHOLD_SECONDS:
                        should_remove = True
                        reason = (
                            f"stale ({elapsed:.0f}s idle, "
                            f"threshold={self.STALE_AGENT_THRESHOLD_SECONDS}s)"
                        )

                if should_remove:
                    logger.info(
                        f"[Pool cleanup] Removing agent {agent.agent_id} "
                        f"(type={agent.agent_type}, tenant={tenant_id}): {reason}"
                    )
                    await self.agent_pool.remove_agent(tenant_id, agent.agent_id)
                    removed_count += 1

            if removed_count:
                logger.info(
                    f"[Pool cleanup] Removed {removed_count} stale/completed "
                    f"agent(s) for tenant {tenant_id}"
                )
        except Exception as e:
            # Never block request processing due to cleanup failure
            logger.warning(f"[Pool cleanup] Failed for tenant {tenant_id}: {e}")

    async def _restore_sessions(self) -> None:
        """Restore all sessions from storage."""
        if not self._agent_registry:
            logger.warning("Cannot restore sessions: no registry available")
            return

        try:
            count = await self.agent_pool.restore_all_sessions(
                self._create_agent_from_entry,
                agent_registry=self._agent_registry,
            )
            logger.info(f"Restored {count} agent sessions")
        except Exception as e:
            logger.error(f"Failed to restore sessions: {e}")

    async def _restore_tenant_session(self, tenant_id: str) -> None:
        """Restore sessions for a specific tenant."""
        if not self._agent_registry:
            return

        try:
            await self.agent_pool.restore_tenant_session(
                tenant_id,
                self._create_agent_from_entry,
                agent_registry=self._agent_registry,
            )
        except Exception as e:
            logger.error(f"Failed to restore session for tenant {tenant_id}: {e}")

    def _create_agent_from_entry(self, entry: AgentPoolEntry) -> StandardAgent:
        """Create agent from pool entry for session restoration."""
        if not self._agent_registry:
            raise RuntimeError("Cannot restore agent: no registry available")

        agent = self._agent_registry.create_agent(
            name=entry.agent_type,
            tenant_id=entry.tenant_id,
            checkpoint_manager=self.checkpoint_manager,
            message_hub=self.message_hub,
            orchestrator_callback=self._create_callback_invoker(entry.tenant_id),
        )

        if not agent:
            raise RuntimeError(f"Agent type not found: {entry.agent_type}")

        # Restore state from entry
        agent.collected_fields = entry.collected_fields
        agent.execution_state = entry.execution_state
        agent.context = entry.context
        agent.status = AgentStatus(entry.status)
        agent.agent_id = entry.agent_id

        return agent

    async def list_pending_approvals(self, tenant_id: str) -> List[Dict[str, Any]]:
        """List all pending approvals for a tenant.

        Queries the agent pool for WAITING_FOR_APPROVAL agents and
        the trigger engine for PENDING_APPROVAL tasks.

        Returns:
            List of approval info dicts with agent_name, action_summary, source, etc.
        """
        results: List[Dict[str, Any]] = []

        # Pool: agents waiting for approval
        agents = await self.agent_pool.list_agents(tenant_id)
        for agent in agents:
            if agent.status == AgentStatus.WAITING_FOR_APPROVAL:
                results.append(
                    {
                        "agent_id": agent.agent_id,
                        "agent_type": agent.agent_type,
                        "agent_name": agent.agent_type,
                        "action_summary": getattr(agent, "raw_message", "")
                        or f"{agent.agent_type} awaiting approval",
                        "source": "user",
                        "created_at": getattr(agent, "created_at", None),
                    }
                )

        # TriggerEngine: tasks pending approval
        if self.trigger_engine:
            pending_tasks = await self.trigger_engine.list_pending_approvals(tenant_id)
            for task in pending_tasks:
                results.append(
                    {
                        "task_id": task.id,
                        "task_name": task.name,
                        "agent_name": task.name,
                        "action_summary": getattr(task, "description", "") or task.name,
                        "source": "trigger",
                        "trigger_type": task.trigger.type.value,
                    }
                )

        return results

    async def list_agents(self, tenant_id: str) -> List[Dict[str, Any]]:
        """List all active agents for a tenant."""
        agents = await self.agent_pool.list_agents(tenant_id)
        return [
            {
                "agent_id": a.agent_id,
                "agent_type": a.agent_type,
                "status": a.status.value,
            }
            for a in agents
        ]

    async def get_agent_status(self, tenant_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific agent."""
        agent = await self.agent_pool.get_agent(tenant_id, agent_id)
        if not agent:
            return None
        return agent.get_state_summary()

    async def cancel_agent(self, tenant_id: str, agent_id: str) -> bool:
        """Cancel an agent."""
        agent = await self.agent_pool.get_agent(tenant_id, agent_id)
        if agent:
            agent.status = AgentStatus.CANCELLED
            await self.agent_pool.remove_agent(tenant_id, agent_id)
            return True
        return False

    async def pause_agent(self, tenant_id: str, agent_id: str) -> Optional[AgentResult]:
        """Pause an agent."""
        agent = await self.agent_pool.get_agent(tenant_id, agent_id)
        if not agent:
            return None

        pauseable_states = {
            AgentStatus.RUNNING,
            AgentStatus.WAITING_FOR_INPUT,
            AgentStatus.WAITING_FOR_APPROVAL,
            AgentStatus.INITIALIZING,
        }

        if agent.status not in pauseable_states:
            logger.warning(f"Cannot pause agent {agent_id} in {agent.status} state")
            return None

        result = agent.pause()
        await self.agent_pool.update_agent(agent)
        return result

    async def resume_agent(
        self, tenant_id: str, agent_id: str, message: Optional[str] = None
    ) -> Optional[AgentResult]:
        """Resume a paused agent."""
        agent = await self.agent_pool.get_agent(tenant_id, agent_id)
        if not agent:
            return None

        if agent.status != AgentStatus.PAUSED:
            logger.warning(f"Cannot resume agent {agent_id}: not paused (status: {agent.status})")
            return None

        if message:
            metadata = {"tenant_id": tenant_id}
            msg = Message(
                name="",
                content=message,
                role="user",
                metadata=metadata,
            )
            result = await agent.reply(msg)
            agent.status = result.status
        else:
            result = await agent.resume()

        # Update or remove from pool
        if agent.status in AgentStatus.terminal_states():
            await self.agent_pool.remove_agent(tenant_id, agent.agent_id)
        else:
            await self.agent_pool.update_agent(agent)

        return result

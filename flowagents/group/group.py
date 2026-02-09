"""
FlowAgent Agent Group - Execute multiple agents with patterns and merging

This module provides:
- AgentGroup: Main class for agent group execution
- Support for sequential, parallel, and hierarchical patterns
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Protocol, Union

from .models import (
    ExecutionPattern,
    MergeStrategy,
    GroupResult,
    AgentExecutionResult,
    GroupConfig,
)
from .merge import StateMerger


class GroupExecutionError(Exception):
    """Raised when group execution fails"""
    pass


class AgentProtocol(Protocol):
    """Protocol for agents in a group"""
    agent_id: str

    async def reply(self, message: Any) -> Any: ...


class AgentFactoryProtocol(Protocol):
    """Protocol for creating agent instances"""

    async def create_agent(
        self,
        agent_type: str,
        user_id: str,
        context_hints: Optional[Dict[str, Any]] = None
    ) -> AgentProtocol: ...


class AgentGroup:
    """
    Execute multiple agents in parallel or sequential patterns with state merging.

    Supports three execution patterns:
    - sequential: Agent A → Agent B → Agent C (output passes through)
    - parallel: All agents execute simultaneously (states are merged)
    - hierarchical: Manager agent delegates to worker agents

    Example (parallel with merging):
        group = AgentGroup(
            agent_types=["GoogleSearch", "ArxivSearch", "WikiSearch"],
            pattern=ExecutionPattern.PARALLEL,
            merge_strategy={
                "results": MergeStrategy.ADD,
                "confidence": MergeStrategy.MAX,
            }
        )

        result = await group.execute(message, user_id="user_1", factory=agent_factory)

    Example (sequential):
        group = AgentGroup(
            agent_types=["Analyzer", "Summarizer", "Writer"],
            pattern=ExecutionPattern.SEQUENTIAL,
        )

        result = await group.execute(message, user_id="user_1", factory=agent_factory)
    """

    def __init__(
        self,
        agent_types: List[str],
        pattern: ExecutionPattern = ExecutionPattern.PARALLEL,
        merge_strategy: Optional[Dict[str, MergeStrategy]] = None,
        custom_merge_fns: Optional[Dict[str, Callable]] = None,
        max_concurrency: int = 10,
        timeout_seconds: int = 300,
        continue_on_error: bool = True,
        group_id: Optional[str] = None
    ):
        """
        Initialize agent group.

        Args:
            agent_types: List of agent type names to include
            pattern: Execution pattern (parallel, sequential, hierarchical)
            merge_strategy: Merge strategies for parallel execution
            custom_merge_fns: Custom merge functions
            max_concurrency: Max parallel agents
            timeout_seconds: Timeout for entire group execution
            continue_on_error: Continue if agent fails
            group_id: Optional group identifier
        """
        self.agent_types = agent_types
        self.pattern = pattern
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
        self.continue_on_error = continue_on_error
        self.group_id = group_id or f"group_{uuid.uuid4().hex[:8]}"

        # Initialize merger for parallel execution
        self.merger = StateMerger(
            merge_strategies=merge_strategy or {},
            custom_merge_fns=custom_merge_fns or {}
        )

    async def execute(
        self,
        message: Any,
        user_id: str,
        factory: AgentFactoryProtocol,
        shared_inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> GroupResult:
        """
        Execute the agent group.

        Args:
            message: Message to send to agents
            user_id: User ID for agent creation
            factory: Factory for creating agents
            shared_inputs: Inputs to share with all agents
            context: Additional context

        Returns:
            GroupResult with execution details
        """
        result = GroupResult(
            group_id=self.group_id,
            pattern=self.pattern,
            status="completed",
            started_at=datetime.now()
        )

        try:
            if self.pattern == ExecutionPattern.SEQUENTIAL:
                await self._execute_sequential(
                    message, user_id, factory, shared_inputs, context, result
                )
            elif self.pattern == ExecutionPattern.PARALLEL:
                await self._execute_parallel(
                    message, user_id, factory, shared_inputs, context, result
                )
            elif self.pattern == ExecutionPattern.HIERARCHICAL:
                await self._execute_hierarchical(
                    message, user_id, factory, shared_inputs, context, result
                )

            # Determine final status
            if result.failed_agents > 0:
                if result.completed_agents > 0:
                    result.status = "partial"
                else:
                    result.status = "failed"

        except Exception as e:
            result.status = "failed"
            result.errors.append(str(e))

        result.completed_at = datetime.now()
        return result

    async def _execute_sequential(
        self,
        message: Any,
        user_id: str,
        factory: AgentFactoryProtocol,
        shared_inputs: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
        result: GroupResult
    ) -> None:
        """Execute agents sequentially, passing output to next"""
        current_message = message
        accumulated_context = dict(context or {})

        if shared_inputs:
            accumulated_context.update(shared_inputs)

        for agent_type in self.agent_types:
            agent_result = await self._execute_single_agent(
                agent_type=agent_type,
                message=current_message,
                user_id=user_id,
                factory=factory,
                context=accumulated_context
            )

            result.add_result(agent_result)

            if agent_result.is_success:
                # Pass output to next agent
                if agent_result.output:
                    current_message = agent_result.output
                # Accumulate collected fields
                accumulated_context.update(agent_result.collected_fields)
            else:
                # Stop on error (unless continue_on_error)
                if not self.continue_on_error:
                    break

        # Final output is from last successful agent
        if result.agent_results:
            last_successful = None
            for r in reversed(result.agent_results):
                if r.is_success:
                    last_successful = r
                    break
            if last_successful:
                result.final_output = last_successful.output
                result.final_message = last_successful.raw_message
                result.merged_fields = accumulated_context

    async def _execute_parallel(
        self,
        message: Any,
        user_id: str,
        factory: AgentFactoryProtocol,
        shared_inputs: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
        result: GroupResult
    ) -> None:
        """Execute agents in parallel and merge results"""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def execute_with_semaphore(agent_type: str) -> AgentExecutionResult:
            async with semaphore:
                return await self._execute_single_agent(
                    agent_type=agent_type,
                    message=message,
                    user_id=user_id,
                    factory=factory,
                    context={**(context or {}), **(shared_inputs or {})}
                )

        # Execute all agents in parallel
        tasks = [execute_with_semaphore(agent_type) for agent_type in self.agent_types]
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        states_to_merge = []
        for i, res in enumerate(agent_results):
            if isinstance(res, Exception):
                # Task raised exception
                agent_result = AgentExecutionResult(
                    agent_id=f"{self.agent_types[i]}_{i}",
                    agent_type=self.agent_types[i],
                    status="failed",
                    error=str(res)
                )
            else:
                agent_result = res
                if agent_result.is_success:
                    states_to_merge.append(agent_result.collected_fields)

            result.add_result(agent_result)

        # Merge states
        result.merged_fields = self.merger.merge(states_to_merge)

        # Collect outputs
        outputs = []
        messages = []
        for r in result.agent_results:
            if r.is_success:
                if r.output is not None:
                    outputs.append(r.output)
                if r.raw_message:
                    messages.append(r.raw_message)

        result.final_output = outputs if outputs else None
        result.final_message = "\n\n".join(messages) if messages else None

    async def _execute_hierarchical(
        self,
        message: Any,
        user_id: str,
        factory: AgentFactoryProtocol,
        shared_inputs: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
        result: GroupResult
    ) -> None:
        """Execute hierarchical pattern (first agent is manager, rest are workers)"""
        if len(self.agent_types) < 2:
            raise GroupExecutionError(
                "Hierarchical pattern requires at least 2 agents (manager + workers)"
            )

        manager_type = self.agent_types[0]
        worker_types = self.agent_types[1:]

        # Execute manager first
        manager_result = await self._execute_single_agent(
            agent_type=manager_type,
            message=message,
            user_id=user_id,
            factory=factory,
            context={**(context or {}), **(shared_inputs or {})}
        )
        result.add_result(manager_result)

        if not manager_result.is_success:
            result.status = "failed"
            return

        # Manager decides which workers to invoke
        # For now, execute all workers with manager's output
        worker_message = manager_result.output or message

        # Execute workers in parallel
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def execute_worker(worker_type: str) -> AgentExecutionResult:
            async with semaphore:
                return await self._execute_single_agent(
                    agent_type=worker_type,
                    message=worker_message,
                    user_id=user_id,
                    factory=factory,
                    context={**(context or {}), **(shared_inputs or {}), **manager_result.collected_fields}
                )

        tasks = [execute_worker(wt) for wt in worker_types]
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)

        states_to_merge = [manager_result.collected_fields]

        for i, res in enumerate(worker_results):
            if isinstance(res, Exception):
                agent_result = AgentExecutionResult(
                    agent_id=f"{worker_types[i]}_{i}",
                    agent_type=worker_types[i],
                    status="failed",
                    error=str(res)
                )
            else:
                agent_result = res
                if agent_result.is_success:
                    states_to_merge.append(agent_result.collected_fields)

            result.add_result(agent_result)

        # Merge all states
        result.merged_fields = self.merger.merge(states_to_merge)

        # Final output from workers
        outputs = []
        for r in result.agent_results[1:]:  # Skip manager
            if r.is_success and r.output is not None:
                outputs.append(r.output)

        result.final_output = outputs if outputs else None

    async def _execute_single_agent(
        self,
        agent_type: str,
        message: Any,
        user_id: str,
        factory: AgentFactoryProtocol,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentExecutionResult:
        """Execute a single agent and return result"""
        agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
        started_at = datetime.now()

        try:
            # Create agent
            agent = await factory.create_agent(
                agent_type=agent_type,
                user_id=user_id,
                context_hints=context
            )

            # Execute agent
            reply = await agent.reply(message)

            return AgentExecutionResult(
                agent_id=agent_id,
                agent_type=agent_type,
                status="completed",
                collected_fields=getattr(agent, 'collected_fields', {}),
                execution_state=getattr(agent, 'execution_state', {}),
                raw_message=getattr(reply, 'raw_message', str(reply)),
                output=getattr(reply, 'data', reply),
                started_at=started_at,
                completed_at=datetime.now()
            )

        except Exception as e:
            return AgentExecutionResult(
                agent_id=agent_id,
                agent_type=agent_type,
                status="failed",
                error=str(e),
                error_type=type(e).__name__,
                started_at=started_at,
                completed_at=datetime.now()
            )

    def set_merge_strategy(
        self,
        field: str,
        strategy: MergeStrategy,
        custom_fn: Optional[Callable] = None
    ) -> None:
        """Set merge strategy for a field"""
        self.merger.set_strategy(field, strategy, custom_fn)

    def add_agent(self, agent_type: str) -> None:
        """Add an agent type to the group"""
        self.agent_types.append(agent_type)

    def remove_agent(self, agent_type: str) -> bool:
        """Remove an agent type from the group"""
        if agent_type in self.agent_types:
            self.agent_types.remove(agent_type)
            return True
        return False

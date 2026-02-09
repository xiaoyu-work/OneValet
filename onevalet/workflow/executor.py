"""
OneValet Workflow Executor - Execute multi-agent workflows

This module implements the workflow execution engine that supports:
- Sequential execution (run)
- Parallel execution (parallel)
- Aggregator pattern (then)
- Multi-stage execution (stages)

The executor integrates with the agent system to:
1. Create agent instances based on workflow definition
2. Pass resolved parameters to agents
3. Collect outputs and pass to subsequent agents
4. Handle errors and support pause/resume
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Awaitable, Protocol, TypeVar
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

from .models import (
    WorkflowType,
    WorkflowStatus,
    StageStatus,
    WorkflowTemplate,
    StageDefinition,
    UserProfile,
    UserWorkflowInstance,
    ResolvedInputs,
    WorkflowAgentResult,
    StageResult,
    WorkflowResult,
    WorkflowExecution,
    WorkflowContext,
)
from .resolver import ParameterResolver, AgentInputMatcher


class ExecutionError(Exception):
    """Raised when workflow execution fails"""
    pass


class AgentFactoryProtocol(Protocol):
    """Protocol for agent factory - creates agent instances"""

    async def create_agent(
        self,
        agent_type: str,
        user_id: str,
        context_hints: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create an agent instance of the given type"""
        ...

    def get_required_fields(self, agent_type: str) -> List[str]:
        """Get list of required field names for an agent type"""
        ...


class AgentExecutorProtocol(Protocol):
    """Protocol for executing a single agent"""

    async def execute(
        self,
        agent: Any,
        inputs: Dict[str, Any],
        context: WorkflowContext
    ) -> WorkflowAgentResult:
        """Execute an agent with given inputs and return result"""
        ...


class WorkflowExecutor:
    """
    Executes workflows based on their definition.

    The executor supports three execution patterns:
    - run: Sequential execution of agents
    - parallel: Parallel execution of agents
    - stages: Multi-stage execution with parallel/sequential stages

    Example usage:
        executor = WorkflowExecutor(
            agent_factory=my_agent_factory,
            parameter_resolver=ParameterResolver()
        )

        result = await executor.execute(
            workflow=workflow_template,
            user_profile=user_profile,
            instance=user_instance
        )
    """

    def __init__(
        self,
        agent_factory: AgentFactoryProtocol,
        parameter_resolver: Optional[ParameterResolver] = None,
        max_concurrency: int = 10,
        default_agent_timeout: int = 300,
        approval_callback: Optional[Callable[[str, str, str, Dict[str, Any]], Awaitable[bool]]] = None
    ):
        """
        Initialize the workflow executor.

        Args:
            agent_factory: Factory for creating agent instances
            parameter_resolver: Resolver for three-layer parameters
            max_concurrency: Maximum number of parallel agents
            default_agent_timeout: Default timeout for agent execution (seconds)
            approval_callback: Async callback for approval requests.
                Signature: (user_id, agent_type, approval_prompt, collected_data) -> bool
                Return True to approve, False to reject.
                If None, agents requiring approval will be auto-approved.
        """
        self.agent_factory = agent_factory
        self.parameter_resolver = parameter_resolver or ParameterResolver()
        self.input_matcher = AgentInputMatcher()
        self.max_concurrency = max_concurrency
        self.default_agent_timeout = default_agent_timeout
        self.approval_callback = approval_callback

    async def execute(
        self,
        workflow: WorkflowTemplate,
        user_id: str,
        user_profile: Optional[UserProfile] = None,
        instance: Optional[UserWorkflowInstance] = None,
        event_data: Optional[Dict[str, Any]] = None,
        trigger_type: str = "manual"
    ) -> WorkflowResult:
        """
        Execute a workflow for a user.

        Args:
            workflow: Workflow template to execute
            user_id: User executing the workflow
            user_profile: User profile for parameter resolution
            instance: User-specific workflow instance
            event_data: Event data for event-triggered workflows
            trigger_type: How the workflow was triggered

        Returns:
            WorkflowResult with execution results
        """
        # Create execution tracking object
        execution = WorkflowExecution(
            id=self._generate_execution_id(),
            workflow_id=workflow.id,
            user_id=user_id,
            status=WorkflowStatus.RUNNING,
            trigger_type=trigger_type,
            trigger_data=event_data or {},
            started_at=datetime.now()
        )

        try:
            # Resolve parameters
            resolved_inputs = self.parameter_resolver.resolve(
                template=workflow,
                user_profile=user_profile,
                instance=instance,
                event_data=event_data
            )
            execution.resolved_inputs = resolved_inputs

            # Build execution context
            context = WorkflowContext(
                workflow_id=workflow.id,
                execution_id=execution.id,
                user_id=user_id,
                resolved_inputs=resolved_inputs,
                user=user_profile.data if user_profile else {},
                event=event_data or {},
                system={
                    "today": resolved_inputs.get("today"),
                    "timezone": resolved_inputs.get("timezone"),
                }
            )

            # Execute based on workflow type
            exec_type = workflow.get_execution_type()

            if exec_type == "run":
                await self._execute_sequential(
                    agents=workflow.run or [],
                    execution=execution,
                    context=context,
                    workflow=workflow
                )
            elif exec_type == "parallel":
                await self._execute_parallel(
                    agents=workflow.parallel or [],
                    aggregator=workflow.then,
                    execution=execution,
                    context=context,
                    workflow=workflow
                )
            elif exec_type == "stages":
                await self._execute_stages(
                    stages=workflow.stages or [],
                    execution=execution,
                    context=context,
                    workflow=workflow
                )

            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()

        return execution.to_result()

    async def _execute_sequential(
        self,
        agents: List[str],
        execution: WorkflowExecution,
        context: WorkflowContext,
        workflow: Optional[WorkflowTemplate] = None
    ) -> None:
        """Execute agents sequentially (run keyword)"""
        for i, agent_type in enumerate(agents):
            execution.current_agent_index = i

            result = await self._execute_single_agent(
                agent_type=agent_type,
                execution=execution,
                context=context,
                workflow=workflow
            )

            execution.add_agent_result(result)

            # Update context with output for next agent
            if result.output is not None:
                context.outputs[agent_type] = result.output

            # Stop on failure
            if result.status == "failed":
                raise ExecutionError(
                    f"Agent '{agent_type}' failed: {result.error}"
                )

    async def _execute_parallel(
        self,
        agents: List[str],
        aggregator: Optional[str],
        execution: WorkflowExecution,
        context: WorkflowContext,
        workflow: Optional[WorkflowTemplate] = None
    ) -> None:
        """Execute agents in parallel (parallel keyword)"""
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def execute_with_semaphore(agent_type: str) -> WorkflowAgentResult:
            async with semaphore:
                return await self._execute_single_agent(
                    agent_type=agent_type,
                    execution=execution,
                    context=context,
                    workflow=workflow
                )

        # Execute all agents in parallel
        tasks = [execute_with_semaphore(agent_type) for agent_type in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to failed result
                result = WorkflowAgentResult(
                    agent_id=f"{agents[i]}_{i}",
                    agent_type=agents[i],
                    status="failed",
                    error=str(result)
                )

            execution.add_agent_result(result)

            # Update context with output
            if result.output is not None:
                context.outputs[result.agent_type] = result.output

        # Execute aggregator if specified
        if aggregator:
            # Pass all parallel outputs to aggregator
            aggregator_result = await self._execute_single_agent(
                agent_type=aggregator,
                execution=execution,
                context=context,
                workflow=workflow,
                is_aggregator=True
            )
            execution.add_agent_result(aggregator_result)

    async def _execute_stages(
        self,
        stages: List[StageDefinition],
        execution: WorkflowExecution,
        context: WorkflowContext,
        workflow: Optional[WorkflowTemplate] = None
    ) -> None:
        """Execute multi-stage workflow (stages keyword)"""
        for stage_index, stage in enumerate(stages):
            execution.current_stage_index = stage_index

            # Check stage condition if specified
            if stage.condition:
                if not self._evaluate_condition(stage.condition, context):
                    # Skip this stage
                    stage_result = StageResult(
                        stage_index=stage_index,
                        status=StageStatus.SKIPPED
                    )
                    execution.add_stage_result(stage_result)
                    continue

            # Execute stage
            stage_result = await self._execute_stage(
                stage=stage,
                stage_index=stage_index,
                execution=execution,
                context=context,
                workflow=workflow
            )

            execution.add_stage_result(stage_result)

            # Stop on failure
            if stage_result.status == StageStatus.FAILED:
                raise ExecutionError(
                    f"Stage {stage_index} failed: {stage_result.error}"
                )

    async def _execute_stage(
        self,
        stage: StageDefinition,
        stage_index: int,
        execution: WorkflowExecution,
        context: WorkflowContext,
        workflow: Optional[WorkflowTemplate] = None
    ) -> StageResult:
        """Execute a single stage"""
        stage_result = StageResult(
            stage_index=stage_index,
            status=StageStatus.RUNNING,
            started_at=datetime.now()
        )

        try:
            if stage.parallel:
                # Parallel execution within stage
                semaphore = asyncio.Semaphore(self.max_concurrency)

                async def execute_with_semaphore(agent_type: str) -> WorkflowAgentResult:
                    async with semaphore:
                        return await self._execute_single_agent(
                            agent_type=agent_type,
                            execution=execution,
                            context=context,
                            workflow=workflow
                        )

                tasks = [execute_with_semaphore(agent_type) for agent_type in stage.parallel]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        result = WorkflowAgentResult(
                            agent_id=f"{stage.parallel[i]}_{i}",
                            agent_type=stage.parallel[i],
                            status="failed",
                            error=str(result)
                        )
                    stage_result.agent_results.append(result)

                    # Update context
                    if result.output is not None:
                        context.outputs[result.agent_type] = result.output

            elif stage.run:
                # Sequential execution within stage
                for agent_type in stage.run:
                    result = await self._execute_single_agent(
                        agent_type=agent_type,
                        execution=execution,
                        context=context,
                        workflow=workflow
                    )
                    stage_result.agent_results.append(result)

                    # Update context
                    if result.output is not None:
                        context.outputs[agent_type] = result.output

                    if result.status == "failed":
                        raise ExecutionError(f"Agent '{agent_type}' failed: {result.error}")

            # Execute aggregator if specified
            if stage.then:
                aggregator_result = await self._execute_single_agent(
                    agent_type=stage.then,
                    execution=execution,
                    context=context,
                    workflow=workflow,
                    is_aggregator=True
                )
                stage_result.aggregator_result = aggregator_result
                stage_result.output = aggregator_result.output

                # Update context
                if aggregator_result.output is not None:
                    context.outputs[stage.then] = aggregator_result.output
            else:
                # Use last agent's output as stage output
                if stage_result.agent_results:
                    stage_result.output = stage_result.agent_results[-1].output

            stage_result.status = StageStatus.COMPLETED
            stage_result.completed_at = datetime.now()

        except Exception as e:
            stage_result.status = StageStatus.FAILED
            stage_result.error = str(e)
            stage_result.completed_at = datetime.now()

        return stage_result

    async def _execute_single_agent(
        self,
        agent_type: str,
        execution: WorkflowExecution,
        context: WorkflowContext,
        workflow: Optional[WorkflowTemplate] = None,
        is_aggregator: bool = False
    ) -> WorkflowAgentResult:
        """Execute a single agent"""
        started_at = datetime.now()
        agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"

        try:
            # Get required fields for this agent
            required_fields = self.agent_factory.get_required_fields(agent_type)

            # Match inputs for this agent
            if is_aggregator:
                # Aggregators receive all previous outputs
                agent_inputs = {
                    **execution.resolved_inputs.values,
                    "outputs": context.outputs
                }
            else:
                # Regular agents receive matched inputs
                agent_inputs = self.input_matcher.get_all_matching_inputs(
                    execution.resolved_inputs,
                    required_fields
                )

            # Create agent instance
            agent = await self.agent_factory.create_agent(
                agent_type=agent_type,
                user_id=execution.user_id,
                context_hints=agent_inputs
            )

            # Execute agent with full reply cycle
            result = await self._run_agent(
                agent=agent,
                inputs=agent_inputs,
                context=context,
                workflow=workflow,
                user_id=execution.user_id
            )

            return WorkflowAgentResult(
                agent_id=agent_id,
                agent_type=agent_type,
                status="completed",
                output=result.get("output"),
                raw_message=result.get("message"),
                started_at=started_at,
                completed_at=datetime.now()
            )

        except Exception as e:
            return WorkflowAgentResult(
                agent_id=agent_id,
                agent_type=agent_type,
                status="failed",
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now()
            )

    async def _run_agent(
        self,
        agent: Any,
        inputs: Dict[str, Any],
        context: WorkflowContext,
        workflow: Optional[WorkflowTemplate] = None,
        user_id: str = ""
    ) -> Dict[str, Any]:
        """
        Run an agent to completion with full reply cycle.

        Handles:
        1. Initial message with inputs
        2. Field collection (inputs should already be provided via context_hints)
        3. Approval workflow (auto-approve based on workflow config or callback)
        4. Final execution

        Args:
            agent: The agent instance to run
            inputs: Pre-resolved inputs for the agent
            context: Workflow context
            workflow: Workflow template (for auto-approve settings)
            user_id: User ID for approval callback

        Returns:
            Dict with 'output' and 'message' keys
        """
        # Check if agent has a direct execute method (for simple agents)
        if hasattr(agent, "execute_with_inputs"):
            result = await agent.execute_with_inputs(inputs)
            return {
                "output": result.data if hasattr(result, "data") else result,
                "message": result.raw_message if hasattr(result, "raw_message") else str(result)
            }

        # For StandardAgent, use reply() cycle
        if hasattr(agent, "reply"):
            return await self._run_agent_reply_cycle(
                agent=agent,
                inputs=inputs,
                workflow=workflow,
                user_id=user_id
            )

        # Fallback: call _execute directly if available
        if hasattr(agent, "_execute"):
            result = await agent._execute(inputs)
            return {
                "output": result,
                "message": str(result)
            }

        raise ExecutionError(
            f"Agent '{agent.__class__.__name__}' does not support direct execution. "
            "Implement 'execute_with_inputs', 'reply', or '_execute' method."
        )

    async def _run_agent_reply_cycle(
        self,
        agent: Any,
        inputs: Dict[str, Any],
        workflow: Optional[WorkflowTemplate] = None,
        user_id: str = ""
    ) -> Dict[str, Any]:
        """
        Run a StandardAgent through its complete reply cycle.

        Args:
            agent: StandardAgent instance
            inputs: Pre-resolved inputs (already set as context_hints)
            workflow: Workflow template for auto-approve settings
            user_id: User ID for approval callback

        Returns:
            Dict with 'output' and 'message' keys
        """
        from ..message import Message

        # Create initial message summarizing the task
        current_msg: Optional[Message] = Message(
            name="workflow",
            content=f"Execute with inputs: {inputs}",
            role="user"
        )

        max_iterations = 10  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Call agent reply with current message
            result = await agent.reply(current_msg)
            current_msg = None  # Clear for next iteration unless set below

            # Check terminal states
            if result.status.value == "completed":
                return {
                    "output": result.data,
                    "message": result.raw_message
                }

            if result.status.value == "cancelled":
                raise ExecutionError(f"Agent cancelled: {result.raw_message}")

            if result.status.value == "error":
                raise ExecutionError(f"Agent error: {result.error_message or result.raw_message}")

            # Handle approval
            if result.status.value == "waiting_for_approval":
                approved = await self._handle_approval(
                    agent_type=agent.__class__.__name__,
                    approval_prompt=result.raw_message,
                    collected_data=result.data,
                    workflow=workflow,
                    user_id=user_id
                )

                if approved:
                    # Set approval message for next iteration
                    current_msg = Message(
                        name="workflow",
                        content="yes",
                        role="user"
                    )
                else:
                    # Reject
                    reject_msg = Message(
                        name="workflow",
                        content="no",
                        role="user"
                    )
                    result = await agent.reply(reject_msg)
                    raise ExecutionError(f"Agent rejected by user: {result.raw_message}")

            # Handle waiting for input (should not happen in workflow context)
            elif result.status.value == "waiting_for_input":
                # In workflow context, all inputs should be pre-provided
                # If still waiting, it's a configuration error
                missing = result.missing_fields or []
                raise ExecutionError(
                    f"Agent requires missing inputs: {missing}. "
                    "Ensure workflow inputs provide all required fields."
                )

        raise ExecutionError(f"Agent execution exceeded max iterations ({max_iterations})")

    async def _handle_approval(
        self,
        agent_type: str,
        approval_prompt: str,
        collected_data: Dict[str, Any],
        workflow: Optional[WorkflowTemplate],
        user_id: str
    ) -> bool:
        """
        Handle approval request for an agent.

        Approval logic:
        1. If workflow.auto_approve_all is True, auto-approve
        2. If agent_type is in workflow.auto_approve list, auto-approve
        3. If approval_callback is set, call it and return result
        4. Otherwise, auto-approve (default for workflow execution)

        Args:
            agent_type: Type of agent requesting approval
            approval_prompt: The approval prompt from the agent
            collected_data: Data collected by the agent
            workflow: Workflow template with approval settings
            user_id: User ID for callback

        Returns:
            True to approve, False to reject
        """
        # Check workflow-level auto-approve settings
        if workflow:
            # Auto-approve all agents in this workflow
            if workflow.auto_approve_all:
                logger.debug(f"Auto-approving {agent_type} (workflow.auto_approve_all=True)")
                return True

            # Auto-approve specific agent types
            if agent_type in workflow.auto_approve:
                logger.debug(f"Auto-approving {agent_type} (in workflow.auto_approve list)")
                return True

        # Use callback if available
        if self.approval_callback:
            logger.debug(f"Requesting approval for {agent_type} via callback")
            return await self.approval_callback(
                user_id,
                agent_type,
                approval_prompt,
                collected_data
            )

        # Default: auto-approve in workflow context
        # (workflows are typically automated, user already initiated the workflow)
        logger.debug(f"Auto-approving {agent_type} (no callback, default workflow behavior)")
        return True

    def _evaluate_condition(self, condition: str, context: WorkflowContext) -> bool:
        """
        Evaluate a Python expression condition.

        Supports safe evaluation with access to:
        - outputs: Dict of outputs from previous agents/stages
        - inputs: Dict of resolved workflow inputs
        - user: Dict of user profile data
        - event: Dict of event data (for event-triggered workflows)

        Also has access to safe built-in functions:
        len, str, int, float, bool, list, dict, any, all

        Examples:
            condition: "outputs['check_user']['is_vip'] == True"
            condition: "outputs['spam_check']['score'] > 0.8"
            condition: "inputs['send_notification'] and outputs['process']['success']"
            condition: "user['tier'] == 'premium'"
            condition: "len(outputs['results']) > 0"

        Args:
            condition: Python expression string
            context: WorkflowContext with outputs, resolved_inputs, user, and event

        Returns:
            Boolean result of the condition evaluation
        """
        if not condition or not condition.strip():
            return True

        # Build safe namespace
        # Get inputs from resolved_inputs.values
        inputs_dict = {}
        if context.resolved_inputs:
            if hasattr(context.resolved_inputs, 'values'):
                inputs_dict = dict(context.resolved_inputs.values)
            elif isinstance(context.resolved_inputs, dict):
                inputs_dict = dict(context.resolved_inputs)

        allowed_names = {
            "outputs": dict(context.outputs) if context.outputs else {},
            "inputs": inputs_dict,
            "user": dict(context.user) if context.user else {},
            "event": dict(context.event) if context.event else {},
            "True": True,
            "False": False,
            "None": None,
            # Safe built-in functions
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "any": any,
            "all": all,
        }

        try:
            # Evaluate with restricted builtins for security
            result = eval(condition, {"__builtins__": {}}, allowed_names)
            return bool(result)
        except Exception as e:
            logger.warning(f"Condition evaluation failed: '{condition}' - {e}")
            return False

    def _generate_execution_id(self) -> str:
        """Generate a unique execution ID"""
        return f"wfexec_{uuid.uuid4().hex}"


class SimpleAgentFactory:
    """
    Simple agent factory implementation for testing.

    In production, this would be replaced by the actual
    AgentRegistry or ConfigLoader.
    """

    def __init__(self, agent_classes: Optional[Dict[str, type]] = None):
        self._agent_classes = agent_classes or {}
        self._field_requirements: Dict[str, List[str]] = {}

    def register(
        self,
        agent_type: str,
        agent_class: type,
        required_fields: Optional[List[str]] = None
    ) -> None:
        """Register an agent class"""
        self._agent_classes[agent_type] = agent_class
        self._field_requirements[agent_type] = required_fields or []

    async def create_agent(
        self,
        agent_type: str,
        user_id: str,
        context_hints: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create an agent instance"""
        if agent_type not in self._agent_classes:
            raise ExecutionError(f"Unknown agent type: {agent_type}")

        agent_class = self._agent_classes[agent_type]
        return agent_class(user_id=user_id, context_hints=context_hints or {})

    def get_required_fields(self, agent_type: str) -> List[str]:
        """Get required fields for an agent type"""
        return self._field_requirements.get(agent_type, [])

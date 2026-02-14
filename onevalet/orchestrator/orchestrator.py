"""
OneValet Orchestrator - Central coordinator using ReAct loop

This module provides an extensible Orchestrator using the Template Method pattern
combined with a ReAct (Reasoning + Acting) loop for tool/agent execution.

Extension Points (override in subclass):
    - prepare_context(): Add memories, user info, custom metadata
    - should_process(): Guardrails, rate limits, tier access control
    - reject_message(): Custom rejection handling
    - create_agent(): Custom agent instantiation
    - post_process(): Save to memory, notifications, response wrapping

Hook-based Extension (no subclass needed):
    - guardrails_checker: Safety filter with check_input / check_output methods
    - rate_limiter: Async callable (tenant_id, context) -> {"allowed": bool}
    - post_process_hooks: List of async callables (result, context) -> result
      for profile detection, usage recording, personality wrapping, etc.

ReAct Loop:
    The orchestrator uses a ReAct loop that:
    1. Sends messages + tool schemas to the LLM
    2. If LLM returns tool_calls, executes them concurrently
    3. Appends results and repeats until LLM produces a final answer
    4. Handles Agent-Tools (agents-as-tools) with approval flow

Example (subclass):
    class MyOrchestrator(Orchestrator):
        async def should_process(self, message, context):
            if not await self.safety_checker.check(message):
                return False
            return True

        async def post_process(self, result, context):
            await self.memory.save(result)
            return result

Example (hooks, no subclass):
    orchestrator = Orchestrator(
        momex=momex,
        llm_client=llm,
        guardrails_checker=my_guardrails,
        rate_limiter=my_rate_limiter,
        post_process_hooks=[profile_detection_hook, usage_recording_hook],
    )
"""

import json
import asyncio
import dataclasses
import logging
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, AsyncIterator, Callable, TYPE_CHECKING

from ..message import Message
from ..result import AgentResult, AgentStatus
from ..streaming.models import StreamMode, AgentEvent, EventType

from .models import (
    OrchestratorConfig,
    AgentPoolEntry,
    AgentCallback,
    CALLBACK_HANDLER_ATTR,
    callback_handler,
)
from .pool import AgentPoolManager
from .react_config import ReactLoopConfig, ToolCallRecord, TokenUsage
from .context_manager import ContextManager
from .agent_tool import execute_agent_tool, AgentToolResult
from .approval import collect_batch_approvals
from .prompts import DEFAULT_SYSTEM_PROMPT
from .audit_logger import AuditLogger
from .tool_policy import ToolPolicyFilter

if TYPE_CHECKING:
    from ..checkpoint import CheckpointManager
    from ..msghub import MessageHub
    from ..protocols import LLMClientProtocol
    from ..memory.momex import MomexMemory

from ..standard_agent import StandardAgent
from ..config import AgentRegistry
from ..tools.models import ToolExecutionContext, ToolDefinition, ToolCategory
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central coordinator for all agents with ReAct loop architecture.

    Uses Template Method pattern - override extension points to customize:

    1. prepare_context() - Build context before processing
    2. should_process() - Gate for message processing
    3. reject_message() - Handle rejected messages
    4. create_agent() - Custom agent instantiation
    5. post_process() - Post-processing before response

    ReAct Loop:
        The _react_loop_events() method implements the Reasoning + Acting pattern:
        - LLM reasons about user request and decides which tools to call
        - Tools (regular + agent-tools) are executed concurrently
        - Results are fed back to the LLM for the next reasoning step
        - Loop continues until LLM produces a final answer or max_turns reached

    Callback Handlers:
        Use @callback_handler decorator to register handlers that agents can invoke:

        class MyOrchestrator(Orchestrator):
            @callback_handler("get_cache")
            async def get_cache(self, callback: AgentCallback) -> Any:
                return self.cache.get(callback.data["key"])

    Basic Usage:
        orchestrator = Orchestrator(
            llm_client=llm_client,
            agent_registry=registry,
            system_prompt="You are a helpful assistant.",
        )
        await orchestrator.initialize()
        response = await orchestrator.handle_message(tenant_id, message)
    """

    # Class-level handler map: callback_name -> method_name
    # Populated by __init_subclass__, with built-in handlers pre-registered
    _callback_handler_map: Dict[str, str] = {
        "list_agents": "_builtin_list_agents",
        "get_agent_config": "_builtin_get_agent_config",
    }

    # Reserved callback names that cannot be overridden by subclasses
    _builtin_callback_names: set = {"list_agents", "get_agent_config"}
    _ROUTER_POLICY_SYSTEM_PROMPT = (
        "You are a tool routing policy engine for an orchestrator. "
        "Return strict JSON only with this schema: "
        "{\"intent\": string, \"must_use_tools\": boolean, "
        "\"selected_tools\": string[], \"force_first_tool\": string|null, "
        "\"reason_code\": string}. "
        "Rules: choose from provided tool names only; use must_use_tools=true "
        "for requests that require external, account, or real-time data/actions. "
        "Prefer composite planner agents for multi-step tasks when available. "
        "Use must_use_tools=false ONLY for clear non-tool scenarios: "
        "casual_chat, creative_writing, language_translation, text_rewrite, pure_math."
    )

    def __init_subclass__(cls, **kwargs):
        """Collect @callback_handler decorated methods when subclass is defined."""
        super().__init_subclass__(**kwargs)

        # Start with parent's handlers
        handler_map: Dict[str, str] = {}
        for base in cls.__mro__[1:]:  # Skip cls itself
            if hasattr(base, '_callback_handler_map'):
                handler_map.update(base._callback_handler_map)

        # Add handlers defined in this class (cls.__dict__ only has this class's attrs)
        for method_name, method in cls.__dict__.items():
            if callable(method):
                callback_name = getattr(method, CALLBACK_HANDLER_ATTR, None)
                if callback_name is not None:
                    # Check for reserved builtin names
                    if callback_name in Orchestrator._builtin_callback_names:
                        raise ValueError(
                            f"Cannot override built-in callback '{callback_name}' in {cls.__name__}. "
                            f"Reserved callbacks: {Orchestrator._builtin_callback_names}"
                        )
                    handler_map[callback_name] = method_name

        cls._callback_handler_map = handler_map

    def __init__(
        self,
        momex: "MomexMemory",
        config: Optional[OrchestratorConfig] = None,
        llm_client: Optional["LLMClientProtocol"] = None,
        agent_registry: Optional[AgentRegistry] = None,
        system_prompt: str = "",
        react_config: Optional[ReactLoopConfig] = None,
        credential_store: Optional[Any] = None,
        database: Optional[Any] = None,
        trigger_engine: Optional[Any] = None,
        checkpoint_manager: Optional["CheckpointManager"] = None,
        message_hub: Optional["MessageHub"] = None,
        guardrails_checker: Optional[Any] = None,
        rate_limiter: Optional[Callable] = None,
        post_process_hooks: Optional[List[Callable]] = None,
        tool_policy_filter: Optional[ToolPolicyFilter] = None,
    ):
        """
        Initialize Orchestrator.

        Args:
            momex: Momex memory Ã¢â‚¬â€ conversation history + long-term knowledge
            config: Full orchestrator configuration
            llm_client: LLM client for the ReAct loop
            agent_registry: Pre-configured agent registry
            system_prompt: Optional user-defined persona / custom instructions appended
                after the built-in OneValet system prompt
            react_config: ReAct loop configuration (max_turns, timeouts, etc.)
            credential_store: CredentialStore for tool execution context
            trigger_engine: TriggerEngine for proactive trigger tasks
            checkpoint_manager: Checkpoint manager for state persistence
            message_hub: Message hub for multi-agent communication
            guardrails_checker: Optional safety checker with async ``check_input(msg)``
                and ``check_output(msg, tenant_id)`` methods.  ``check_input``
                returns ``{"blocked": bool, "reason": str}``.  ``check_output``
                returns ``{"modified": bool, "output": str}``.
            rate_limiter: Optional async callable ``(tenant_id, context) -> dict``
                that returns ``{"allowed": bool, ...}``.  Extra keys are stored
                in ``context["rate_limit_info"]`` for ``reject_message``.
            post_process_hooks: Optional list of async callables
                ``(result: AgentResult, context: dict) -> AgentResult`` invoked
                after the base post_process logic (momex save).  Hooks run in
                order; each receives the result returned by the previous hook.
                Useful for profile detection, usage recording, response wrapping,
                or sending notifications without subclassing the orchestrator.
        """
        # Configuration
        self.config = config or OrchestratorConfig()

        # Core dependencies
        self.momex = momex
        self.llm_client = llm_client
        self.checkpoint_manager = checkpoint_manager
        self.message_hub = message_hub
        self.credential_store = credential_store
        self.database = database
        self.trigger_engine = trigger_engine
        self.system_prompt = system_prompt

        # ReAct loop configuration
        self._react_config = react_config or ReactLoopConfig()
        self._context_manager = ContextManager(self._react_config)

        # Agent registry
        self._agent_registry: Optional[AgentRegistry] = agent_registry
        self._registry_initialized = agent_registry is not None

        # Agent pool manager
        self.agent_pool = AgentPoolManager(config=self.config.session)

        # Extension hooks
        self.guardrails_checker = guardrails_checker
        self.rate_limiter = rate_limiter
        self._post_process_hooks: List[Callable] = list(post_process_hooks or [])
        self._tool_policy_filter = tool_policy_filter

        # Audit logging
        self._audit = AuditLogger()

        # State
        self._initialized = False

    @property
    def agent_registry(self) -> Optional[AgentRegistry]:
        """Get the agent registry"""
        return self._agent_registry

    def add_post_process_hook(self, hook: Callable) -> None:
        """Register an additional post-process hook at runtime.

        Args:
            hook: Async callable ``(result, context) -> AgentResult``
        """
        self._post_process_hooks.append(hook)

    # ==========================================================================
    # LIFECYCLE METHODS
    # ==========================================================================

    async def initialize(self) -> None:
        """
        Initialize the orchestrator.

        Override to add custom initialization logic.
        """
        if self._initialized:
            return

        # Initialize agent registry if not provided
        if not self._registry_initialized and self._agent_registry is None:
            logger.warning("No agent registry provided. Agent-Tools will not be available.")

        # Validate LLM client is available
        if not self.llm_client:
            raise RuntimeError(
                "LLM client is required. Pass llm_client to Orchestrator()."
            )

        # Restore sessions if configured
        if self.config.session.enabled and self.config.session.auto_restore_on_start:
            await self._restore_sessions()

        # Start auto-backup if configured
        if self.config.session.enabled and self.config.session.auto_backup_interval_seconds > 0:
            await self.agent_pool.start_auto_backup()

        # Start trigger engine if configured
        if self.trigger_engine:
            await self.trigger_engine.start()

        # Register recall_memory tool (needs momex instance)
        self._register_memory_tool()

        self._initialized = True
        logger.info("Orchestrator initialized")

    async def shutdown(self) -> None:
        """Shutdown the orchestrator gracefully."""
        if self.trigger_engine:
            await self.trigger_engine.stop()
        await self.agent_pool.close()
        if self._agent_registry:
            await self._agent_registry.shutdown()
        self._initialized = False
        logger.info("Orchestrator shutdown")

    # ==========================================================================
    # MAIN ENTRY POINT
    # ==========================================================================

    async def handle_message(
        self,
        tenant_id: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Main entry point - handle user message via ReAct loop.

        Flow:
        1. prepare_context() - build context
        2. should_process() - gate check
        3. _check_pending_agents() - check for WAITING agents in pool
        4. _build_llm_messages() - system prompt + history + user message
        5. _build_tool_schemas() - merge regular Tools + Agent-Tools
        6. _react_loop_events() - ReAct reasoning loop
        7. post_process() - final processing

        Args:
            tenant_id: Tenant/user identifier
            message: User message text
            metadata: Optional message metadata

        Returns:
            AgentResult with response
        """
        if not self._initialized:
            await self.initialize()

        # Step 1: Prepare context
        context = await self.prepare_context(tenant_id, message, metadata)

        # Step 2: Check if should process
        if not await self.should_process(message, context):
            return await self.reject_message(message, context)

        # Step 3: Check pending agents (WAITING_FOR_INPUT / WAITING_FOR_APPROVAL)
        agent_result = await self._check_pending_agents(tenant_id, message, context)
        if agent_result is not None:
            # Agent still waiting -> return prompt directly, don't enter ReAct
            if agent_result.status in (AgentStatus.WAITING_FOR_INPUT, AgentStatus.WAITING_FOR_APPROVAL):
                return await self.post_process(agent_result, context)
            # Agent completed -> feed result into ReAct loop as context
            context["pending_agent_result"] = agent_result

        # Step 4: Build tool schemas
        tool_schemas = self._build_tool_schemas()
        policy = await self._plan_tool_policy(message, tool_schemas)
        selected_tool_schemas = policy["tool_schemas"]
        logger.info(
            f"[ToolPolicy] intent={policy['intent']} must_use_tools={policy['must_use_tools']} "
            f"selected={len(selected_tool_schemas)}/{len(tool_schemas)} first_turn_choice={policy['first_turn_tool_choice']}"
        )
        self._audit.log_policy_decision(
            intent=policy["intent"],
            must_use_tools=policy["must_use_tools"],
            selected_tools=[self._tool_name_from_schema(s) or "" for s in selected_tool_schemas],
            reason_code=str(policy.get("first_turn_tool_choice", "auto")),
            tenant_id=tenant_id,
        )

        # Step 5: Build LLM messages
        messages = self._build_llm_messages(context, message)

        # Step 6: Run ReAct loop (consume events silently)
        exec_data: Dict[str, Any] = {}
        async for event in self._react_loop_events(
            messages,
            selected_tool_schemas,
            tenant_id,
            first_turn_tool_choice=policy["first_turn_tool_choice"],
            retry_with_required_on_empty=policy["retry_with_required_on_empty"],
        ):
            if event.type == EventType.EXECUTION_END:
                exec_data = event.data

        # Step 7: Map loop results -> AgentResult
        final_response = exec_data.get("final_response", "")
        pending_approvals = exec_data.get("pending_approvals", [])
        result_status = exec_data.get("result_status")

        if pending_approvals:
            status = AgentStatus.WAITING_FOR_APPROVAL
        elif result_status == "WAITING_FOR_INPUT":
            status = AgentStatus.WAITING_FOR_INPUT
        else:
            status = AgentStatus.COMPLETED

        result = AgentResult(
            agent_type=self.__class__.__name__,
            status=status,
            raw_message=final_response,
            metadata={
                "react_turns": exec_data.get("turns", 0),
                "token_usage": exec_data.get("token_usage", {}),
                "duration_ms": exec_data.get("duration_ms", 0),
                "tool_calls_count": exec_data.get("tool_calls_count", 0),
                "tool_policy": {
                    "intent": policy["intent"],
                    "must_use_tools": policy["must_use_tools"],
                    "first_turn_tool_choice": policy["first_turn_tool_choice"],
                    "selected_tool_count": len(selected_tool_schemas),
                    "total_tool_count": len(tool_schemas),
                },
            },
        )

        if pending_approvals:
            result.metadata["pending_approvals"] = [
                {
                    "agent_name": a.agent_name,
                    "action_summary": a.action_summary,
                    "details": a.details,
                    "options": a.options,
                }
                for a in pending_approvals
            ]

        # Step 8: Post-process
        return await self.post_process(result, context)

    # ==========================================================================
    # STREAMING ENTRY POINT
    # ==========================================================================

    async def stream_message(
        self,
        tenant_id: str,
        message: str,
        mode: StreamMode = StreamMode.EVENTS,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[AgentEvent]:
        """
        Stream agent execution events via ReAct loop.

        Same flow as handle_message but yielding streaming events at each stage.

        Args:
            tenant_id: Tenant identifier
            message: User message text
            mode: Stream mode
            metadata: Optional message metadata

        Yields:
            AgentEvent objects
        """
        if not self._initialized:
            await self.initialize()

        # Prepare context
        context = await self.prepare_context(tenant_id, message, metadata)

        # Check if should process
        if not await self.should_process(message, context):
            result = await self.reject_message(message, context)
            yield AgentEvent(
                type=EventType.MESSAGE_CHUNK,
                data={"chunk": result.raw_message or ""},
            )
            return

        # Check pending agents
        agent_result = await self._check_pending_agents(tenant_id, message, context)
        if agent_result is not None:
            # Agent still waiting -> return prompt directly, don't enter ReAct
            if agent_result.status in (AgentStatus.WAITING_FOR_INPUT, AgentStatus.WAITING_FOR_APPROVAL):
                agent_result = await self.post_process(agent_result, context)
                yield AgentEvent(
                    type=EventType.MESSAGE_START,
                    data={"agent_type": agent_result.agent_type},
                )
                yield AgentEvent(
                    type=EventType.MESSAGE_CHUNK,
                    data={"chunk": agent_result.raw_message or ""},
                )
                yield AgentEvent(
                    type=EventType.MESSAGE_END,
                    data={},
                )
                return
            # Agent completed -> feed result into ReAct loop as context
            context["pending_agent_result"] = agent_result

        # Build tool schemas
        tool_schemas = self._build_tool_schemas()
        policy = await self._plan_tool_policy(message, tool_schemas)
        selected_tool_schemas = policy["tool_schemas"]
        logger.info(
            f"[ToolPolicy] intent={policy['intent']} must_use_tools={policy['must_use_tools']} "
            f"selected={len(selected_tool_schemas)}/{len(tool_schemas)} first_turn_choice={policy['first_turn_tool_choice']}"
        )
        self._audit.log_policy_decision(
            intent=policy["intent"],
            must_use_tools=policy["must_use_tools"],
            selected_tools=[self._tool_name_from_schema(s) or "" for s in selected_tool_schemas],
            reason_code=str(policy.get("first_turn_tool_choice", "auto")),
            tenant_id=tenant_id,
        )
        messages = self._build_llm_messages(context, message)

        # Delegate to shared ReAct loop
        final_response = ""
        exec_data: Dict[str, Any] = {}
        async for event in self._react_loop_events(
            messages, selected_tool_schemas, tenant_id,
            first_turn_tool_choice=policy["first_turn_tool_choice"],
            retry_with_required_on_empty=policy["retry_with_required_on_empty"],
        ):
            if event.type == EventType.EXECUTION_END:
                exec_data = event.data
                final_response = exec_data.get("final_response", "")
            yield event

        # Post-process: momex save, guardrails output, hooks
        pending_approvals = exec_data.get("pending_approvals", [])
        result_status = exec_data.get("result_status")
        if pending_approvals:
            status = AgentStatus.WAITING_FOR_APPROVAL
        elif result_status == "WAITING_FOR_INPUT":
            status = AgentStatus.WAITING_FOR_INPUT
        else:
            status = AgentStatus.COMPLETED
        result = AgentResult(
            agent_type=self.__class__.__name__,
            status=status,
            raw_message=final_response,
        )
        await self.post_process(result, context)

    # ==========================================================================
    # REACT LOOP
    # ==========================================================================

    async def _react_loop_events(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: List[Dict[str, Any]],
        tenant_id: str,
        first_turn_tool_choice: Any = "auto",
        retry_with_required_on_empty: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Unified ReAct loop implementation yielding streaming events.

        Both stream_message() and handle_message() delegate to this single
        implementation, eliminating the previous code duplication between
        the inline stream_message loop and react_loop().

        The final EXECUTION_END event carries all metadata (final_response,
        pending_approvals, token_usage, tool_calls records, etc.) so callers
        can build AgentResult or persist to memory as needed.
        """
        start_time = time.monotonic()
        turn = 0
        all_tool_records: List[ToolCallRecord] = []
        total_usage = TokenUsage()
        pending_approvals = []
        final_response = ""
        result_status = None

        logger.info(f"[ReAct] tenant={tenant_id}")

        yield AgentEvent(
            type=EventType.EXECUTION_START,
            data={"tenant_id": tenant_id},
        )

        for turn in range(1, self._react_config.max_turns + 1):
            # Context guard
            messages = self._context_manager.trim_if_needed(messages)

            # LLM call
            try:
                tool_choice = first_turn_tool_choice if turn == 1 else "auto"
                response = await self._llm_call_with_retry(
                    messages, tool_schemas, tool_choice=tool_choice,
                )
            except Exception as e:
                yield AgentEvent(
                    type=EventType.ERROR,
                    data={"error": str(e), "error_type": type(e).__name__},
                )
                return

            # Accumulate token usage
            usage = getattr(response, "usage", None)
            if usage:
                total_usage.input_tokens += getattr(usage, "prompt_tokens", 0)
                total_usage.output_tokens += getattr(usage, "completion_tokens", 0)

            tool_calls = response.tool_calls

            # No tool calls: retry once if policy says so, else final answer
            if not tool_calls:
                if (
                    turn == 1
                    and retry_with_required_on_empty
                    and tool_schemas
                    and first_turn_tool_choice != "required"
                ):
                    response = await self._llm_call_with_retry(
                        messages, tool_schemas, tool_choice="required",
                    )
                    usage_retry = getattr(response, "usage", None)
                    if usage_retry:
                        total_usage.input_tokens += getattr(usage_retry, "prompt_tokens", 0)
                        total_usage.output_tokens += getattr(usage_retry, "completion_tokens", 0)
                    tool_calls = response.tool_calls

                if not tool_calls:
                    final_text = response.content or ""
                    final_response = final_text
                    logger.info(f"[ReAct] turn={turn} final_answer ({len(final_text)} chars)")
                    self._audit.log_react_turn(
                        turn=turn, tool_calls=[], final_answer=True,
                        tenant_id=tenant_id,
                    )
                    yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                    yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_text})
                    yield AgentEvent(type=EventType.MESSAGE_END, data={})
                    break

            if tool_calls:
                # Append assistant message with tool_calls
                messages.append(self._assistant_message_from_response(response))

                tool_names = [tc.name for tc in tool_calls]
                logger.info(f"[ReAct] turn={turn} calling: {', '.join(tool_names)}")

                # Yield tool call start events
                for tc in tool_calls:
                    yield AgentEvent(
                        type=EventType.TOOL_CALL_START,
                        data={"tool_name": tc.name, "call_id": tc.id},
                    )

                # Execute all tool calls concurrently
                tc_batch_start = time.monotonic()
                results = await asyncio.gather(
                    *[self._execute_with_timeout(tc, tenant_id) for tc in tool_calls],
                    return_exceptions=True,
                )
                tc_batch_duration = int((time.monotonic() - tc_batch_start) * 1000)

                # Token attribution for this turn
                turn_tokens = None
                if usage:
                    turn_tokens = TokenUsage(
                        input_tokens=getattr(usage, "prompt_tokens", 0),
                        output_tokens=getattr(usage, "completion_tokens", 0),
                    )

                loop_broken = False
                loop_broken_text = None

                for tc, result in zip(tool_calls, results):
                    tc_name = tc.name
                    is_agent = self._is_agent_tool(tc_name)
                    kind = "agent" if is_agent else "tool"

                    try:
                        args_summary = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args_summary = {}
                    args_summary = {k: str(v)[:100] for k, v in args_summary.items()}

                    if isinstance(result, BaseException):
                        logger.warning(f"[ReAct]   {kind}={tc_name} ERROR: {result}")
                        error_text = f"Error executing {tc_name}: {result}"
                        messages.append(self._build_tool_result_message(tc.id, error_text, is_error=True))
                        all_tool_records.append(ToolCallRecord(
                            name=tc_name, args_summary=args_summary,
                            duration_ms=tc_batch_duration, success=False,
                            result_chars=len(error_text), token_attribution=turn_tokens,
                        ))
                        yield AgentEvent(
                            type=EventType.TOOL_RESULT,
                            data={
                                "tool_name": tc_name, "call_id": tc.id,
                                "kind": kind, "success": False,
                                "error": str(result),
                                "result_preview": error_text[:240],
                            },
                        )
                        self._audit.log_tool_execution(
                            tool_name=tc_name, args_summary=args_summary,
                            success=False, duration_ms=tc_batch_duration,
                            error=str(result), tenant_id=tenant_id,
                        )

                    elif isinstance(result, AgentToolResult) and not result.completed:
                        logger.info(f"[ReAct]   {kind}={tc_name} WAITING")
                        if result.agent:
                            await self.agent_pool.add_agent(result.agent)
                        if result.approval_request:
                            pending_approvals.append(result.approval_request)
                        waiting_text = result.result_text or "Agent is waiting for input."
                        messages.append(self._build_tool_result_message(tc.id, waiting_text))
                        waiting_status = (
                            "WAITING_FOR_APPROVAL" if result.approval_request
                            else "WAITING_FOR_INPUT"
                        )
                        all_tool_records.append(ToolCallRecord(
                            name=tc_name, args_summary=args_summary,
                            duration_ms=tc_batch_duration, success=True,
                            result_status=waiting_status,
                            result_chars=len(waiting_text), token_attribution=turn_tokens,
                        ))
                        tool_trace = []
                        if isinstance(result.metadata, dict):
                            tool_trace = result.metadata.get("tool_trace") or []
                        yield AgentEvent(
                            type=EventType.TOOL_RESULT,
                            data={
                                "tool_name": tc_name, "call_id": tc.id,
                                "kind": "agent", "success": True,
                                "waiting": True, "status": waiting_status,
                                "result_preview": waiting_text[:240],
                                "tool_trace": tool_trace,
                            },
                        )
                        yield AgentEvent(
                            type=EventType.STATE_CHANGE,
                            data={"agent_type": tc_name, "status": waiting_status},
                        )
                        self._audit.log_tool_execution(
                            tool_name=tc_name, args_summary=args_summary,
                            success=True, duration_ms=tc_batch_duration,
                            tenant_id=tenant_id,
                        )
                        loop_broken = True
                        loop_broken_text = waiting_text

                    else:
                        if isinstance(result, AgentToolResult):
                            result_text = result.result_text
                            r_meta = result.metadata if isinstance(result.metadata, dict) else {}
                            tool_trace = r_meta.get("tool_trace") or []
                        else:
                            result_text = str(result) if result is not None else ""
                            tool_trace = []
                        result_chars_original = len(result_text)
                        result_text = self._context_manager.truncate_tool_result(result_text)
                        logger.info(f"[ReAct]   {kind}={tc_name} OK ({len(result_text)} chars)")
                        messages.append(self._build_tool_result_message(tc.id, result_text))
                        all_tool_records.append(ToolCallRecord(
                            name=tc_name, args_summary=args_summary,
                            duration_ms=tc_batch_duration, success=True,
                            result_status="COMPLETED" if isinstance(result, AgentToolResult) else None,
                            result_chars=result_chars_original, token_attribution=turn_tokens,
                        ))
                        yield AgentEvent(
                            type=EventType.TOOL_RESULT,
                            data={
                                "tool_name": tc_name, "call_id": tc.id,
                                "kind": kind, "success": True,
                                "result_preview": result_text[:240],
                                "tool_trace": tool_trace,
                            },
                        )
                        self._audit.log_tool_execution(
                            tool_name=tc_name, args_summary=args_summary,
                            success=True, duration_ms=tc_batch_duration,
                            tenant_id=tenant_id,
                        )

                # Audit: log turn summary
                self._audit.log_react_turn(
                    turn=turn,
                    tool_calls=tool_names,
                    final_answer=False,
                    tenant_id=tenant_id,
                )

                if loop_broken:
                    final_response = loop_broken_text or ""
                    result_status = "WAITING_FOR_APPROVAL" if pending_approvals else "WAITING_FOR_INPUT"
                    if pending_approvals:
                        pending_approvals = collect_batch_approvals(pending_approvals)
                    if loop_broken_text:
                        yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                        yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": loop_broken_text})
                        yield AgentEvent(type=EventType.MESSAGE_END, data={})
                    break

                # Agent passthrough: single completed agent-tool skips LLM re-summary
                if (
                    len(tool_calls) == 1
                    and self._is_agent_tool(tool_calls[0].name)
                    and isinstance(results[0], AgentToolResult)
                    and results[0].completed
                ):
                    agent_text = results[0].result_text
                    logger.info(
                        f"[ReAct] turn={turn} agent_passthrough "
                        f"({len(agent_text)} chars from {tool_calls[0].name})"
                    )
                    final_response = agent_text
                    yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                    yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": agent_text})
                    yield AgentEvent(type=EventType.MESSAGE_END, data={})
                    break

        else:
            # max_turns reached: ask LLM for summary without tools
            messages.append({
                "role": "user",
                "content": (
                    "You have used all available turns. Please provide your best "
                    "final answer based on the information gathered so far."
                ),
            })
            try:
                response = await self._llm_call_with_retry(messages, tool_schemas=None)
                final_text = response.content or ""
                usage = getattr(response, "usage", None)
                if usage:
                    total_usage.input_tokens += getattr(usage, "prompt_tokens", 0)
                    total_usage.output_tokens += getattr(usage, "completion_tokens", 0)
            except Exception:
                final_text = "I was unable to complete the request within the allowed turns."

            final_response = final_text
            yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
            yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_text})
            yield AgentEvent(type=EventType.MESSAGE_END, data={})

        duration_ms = int((time.monotonic() - start_time) * 1000)

        yield AgentEvent(
            type=EventType.EXECUTION_END,
            data={
                "duration_ms": duration_ms,
                "turns": turn,
                "tool_calls_count": len(all_tool_records),
                "final_response": final_response,
                "result_status": result_status,
                "pending_approvals": pending_approvals,
                "token_usage": {
                    "input_tokens": total_usage.input_tokens,
                    "output_tokens": total_usage.output_tokens,
                },
                "tool_calls": [dataclasses.asdict(r) for r in all_tool_records],
            },
        )

    # ==========================================================================
    # REACT LOOP HELPERS
    # ==========================================================================

    async def _llm_call_with_retry(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any] = None,
    ) -> Any:
        """LLM call with error recovery strategy per design doc section 3.3.

        - RateLimitError -> exponential backoff
        - ContextOverflowError -> three-step recovery (trim -> truncate_all -> force_trim)
        - AuthError -> raise immediately
        - TimeoutError -> retry once

        Args:
            tool_choice: Override for tool_choice param ("auto", "required", "none").
                         If None, the LLM client uses its default ("auto").
        """
        last_error = None
        for attempt in range(self._react_config.llm_max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {"messages": messages}
                if tool_schemas:
                    kwargs["tools"] = tool_schemas
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice
                    logger.info(f"[LLM] Sending {len(tool_schemas)} tools, tool_choice={tool_choice or 'auto'}, sample: {json.dumps(tool_schemas[0], ensure_ascii=False)[:200]}")
                else:
                    logger.info("[LLM] Sending request with NO tools")
                response = await self.llm_client.chat_completion(**kwargs)
                # Debug: log what came back
                tc = getattr(response, 'tool_calls', None)
                sr = getattr(response, 'stop_reason', None)
                content_len = len(getattr(response, 'content', '') or '')
                logger.info(f"[LLM] Response: stop_reason={sr}, tool_calls={len(tc) if tc else 0}, content_len={content_len}")
                return response

            except Exception as e:
                last_error = e
                error_name = type(e).__name__.lower()

                # Auth errors: raise immediately
                if "auth" in error_name or "authentication" in error_name or "permission" in error_name:
                    raise

                # Rate limit: exponential backoff
                if "ratelimit" in error_name or "rate_limit" in error_name or "429" in str(e):
                    delay = self._react_config.llm_retry_base_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue

                # Context overflow: three-step recovery
                if "context" in error_name or "overflow" in error_name or "token" in error_name or "length" in str(e).lower():
                    if attempt == 0:
                        logger.warning("Context overflow, trimming history")
                        messages = self._context_manager.trim_if_needed(messages)
                    elif attempt == 1:
                        logger.warning("Context overflow persists, truncating all tool results")
                        messages = self._context_manager.truncate_all_tool_results(messages)
                    else:
                        logger.warning("Context overflow persists, force trimming")
                        messages = self._context_manager.force_trim(messages)
                    continue

                # Timeout: retry once
                if "timeout" in error_name:
                    if attempt == 0:
                        logger.warning("LLM timeout, retrying once")
                        continue
                    raise

                # Unknown error: retry with backoff
                if attempt < self._react_config.llm_max_retries:
                    delay = self._react_config.llm_retry_base_delay * (2 ** attempt)
                    logger.warning(f"LLM call failed ({e}), retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue

                raise

        raise last_error  # type: ignore[misc]

    async def _execute_with_timeout(self, tool_call: Any, tenant_id: str) -> Any:
        """Execute a single tool/agent-tool with timeout."""
        tool_name = tool_call.name
        is_agent = self._is_agent_tool(tool_name)
        timeout = (
            self._react_config.agent_tool_execution_timeout
            if is_agent
            else self._react_config.tool_execution_timeout
        )

        try:
            return await asyncio.wait_for(
                self._execute_single(tool_call, tenant_id),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            kind = "Agent-Tool" if is_agent else "Tool"
            raise TimeoutError(f"{kind} '{tool_name}' timed out after {timeout}s")

    async def _execute_single(self, tool_call: Any, tenant_id: str) -> Any:
        """Dispatch to agent-tool or regular tool execution."""
        tool_name = tool_call.name
        try:
            args = tool_call.arguments if isinstance(tool_call.arguments, dict) else json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            return (
                f"Error: Failed to parse arguments for tool '{tool_name}': {e}. "
                "Please retry with valid JSON arguments."
            )

        if self._is_agent_tool(tool_name):
            # Agent-Tool execution
            task_instruction = args.pop("task_instruction", "")
            return await execute_agent_tool(
                self,
                agent_type=tool_name,
                tenant_id=tenant_id,
                tool_call_args=args,
                task_instruction=task_instruction,
            )
        else:
            # Regular tool execution
            if not self._agent_registry:
                return f"Error: No tool registry available to execute '{tool_name}'"

            tool_def = self._agent_registry.tool_registry.get_tool(tool_name)
            if not tool_def:
                return f"Error: Tool '{tool_name}' not found"

            context = ToolExecutionContext(
                user_id=tenant_id,
                credentials=self.credential_store,
                metadata=self._build_tool_metadata(),
            )
            return await tool_def.executor(args, context)

    def _is_agent_tool(self, tool_name: str) -> bool:
        """Check if tool_name corresponds to a registered agent."""
        if not self._agent_registry:
            return False
        return self._agent_registry.get_agent_class(tool_name) is not None

    def _build_tool_metadata(self) -> dict:
        """Build metadata dict for regular tool execution context."""
        meta = {}
        if self.database:
            from onevalet.builtin_agents.digest.important_dates_repo import ImportantDatesRepository
            meta["important_dates_store"] = ImportantDatesRepository(self.database)
        return meta

    def _build_tool_result_message(
        self,
        tool_call_id: str,
        content: str,
        is_error: bool = False,
    ) -> Dict[str, Any]:
        """Build a tool result message for the LLM messages list."""
        if is_error:
            content = f"[ERROR] {content}"
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    @staticmethod
    def _assistant_message_from_response(response: Any) -> Dict[str, Any]:
        """Convert LLMResponse to dict for the messages list."""
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": getattr(response, "content", None),
        }
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return msg

    # ==========================================================================
    # MESSAGE BUILDING
    # ==========================================================================

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
            a for a in agents
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

    def _build_llm_messages(
        self,
        context: Dict[str, Any],
        user_message: str,
    ) -> List[Dict[str, Any]]:
        """Build the initial LLM message list.

        Contains:
        - System prompt + recalled memories
        - Conversation history (from Momex short-term memory)
        - Current user message
        """
        messages: List[Dict[str, Any]] = []

        # System prompt: built-in base + optional user customization
        system_parts = [DEFAULT_SYSTEM_PROMPT]
        if self.system_prompt:
            system_parts.append(self.system_prompt)

        # Framework context
        framework_parts = []
        now = datetime.now(timezone.utc)
        framework_parts.append(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if framework_parts:
            system_parts.append("\n[Context]\n" + "\n".join(framework_parts))

        # NOTE: The system prompt lists agent/tool names as a routing table
        # so the LLM knows WHICH tool to call for each domain. The detailed
        # schemas are still passed only via the native `tools` parameter.

        if system_parts:
            messages.append({
                "role": "system",
                "content": "\n\n".join(system_parts),
            })

        # Conversation history (from Momex short-term memory)
        history = context.get("conversation_history", [])
        if history:
            logger.info(f"[ReAct] history: {len(history)} messages, roles: {[m.get('role') for m in history[:6]]}...")
            messages.extend(history)
        else:
            logger.info("[ReAct] history: 0 messages (clean session)")

        # Current user message
        messages.append({
            "role": "user",
            "content": user_message,
        })

        # If a pending agent just completed, include its result
        pending_result = context.get("pending_agent_result")
        if pending_result:
            result_text = pending_result.raw_message or f"Agent {pending_result.agent_type} completed."
            messages.append({
                "role": "user",
                "content": f"[Previous agent result: {pending_result.agent_type}]\n{result_text}\n\nBased on this result, determine if any follow-up actions are needed.",
            })

        return messages

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """Build combined tool schemas: regular tools + agent-tools."""
        schemas: List[Dict[str, Any]] = []

        if not self._agent_registry:
            return schemas

        # Regular tools
        all_tools = self._agent_registry.tool_registry.get_all_tools()
        for tool in all_tools:
            schemas.append(tool.to_openai_schema())

        # Agent-tools
        agent_tool_schemas = self._agent_registry.get_all_agent_tool_schemas()
        schemas.extend(agent_tool_schemas)

        # Apply tool policy filter if configured
        if self._tool_policy_filter:
            schemas = self._tool_policy_filter.filter_tools(schemas)

        logger.info(f"[Tools] {len(schemas)} total available")

        return schemas

    @staticmethod
    def _tool_name_from_schema(schema: Dict[str, Any]) -> Optional[str]:
        """Extract function name from an OpenAI tool schema."""
        if not isinstance(schema, dict):
            return None
        function_part = schema.get("function")
        if not isinstance(function_part, dict):
            return None
        name = function_part.get("name")
        return name if isinstance(name, str) else None

    def _filter_tool_schemas(
        self,
        tool_schemas: List[Dict[str, Any]],
        preferred_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Filter tool schemas by a preferred name list while preserving order."""
        if not preferred_names:
            return tool_schemas
        name_set = set(preferred_names)
        filtered = [
            schema for schema in tool_schemas
            if self._tool_name_from_schema(schema) in name_set
        ]
        return filtered

    def _score_tool_relevance(self, user_message: str, schema: Dict[str, Any]) -> float:
        """Generic lexical relevance score for fallback tool narrowing."""
        name = (self._tool_name_from_schema(schema) or "").lower()
        description = str(schema.get("function", {}).get("description", "") or "").lower()
        user_text = (user_message or "").lower()

        user_tokens = set(re.findall(r"[a-z0-9_]+", user_text))
        tool_tokens = set(re.findall(r"[a-z0-9_]+", f"{name} {description}"))
        overlap = user_tokens.intersection(tool_tokens)

        score = float(len(overlap))
        if name and name in user_text:
            score += 2.0
        if self._is_agent_tool(name):
            score += 0.5
        return score

    def _choose_fallback_tools(
        self,
        user_message: str,
        tool_schemas: List[Dict[str, Any]],
        max_tools: int = 8,
    ) -> List[Dict[str, Any]]:
        """Choose a bounded fallback subset when router output is missing/ambiguous."""
        if not tool_schemas:
            return []

        scored: List[tuple[float, Dict[str, Any]]] = []
        for schema in tool_schemas:
            scored.append((self._score_tool_relevance(user_message, schema), schema))
        scored.sort(key=lambda item: item[0], reverse=True)

        picked = [schema for score, schema in scored if score > 0][:max_tools]
        if not picked:
            picked = tool_schemas[:max_tools]

        if not any(self._is_agent_tool(self._tool_name_from_schema(s) or "") for s in picked):
            for schema in tool_schemas:
                name = self._tool_name_from_schema(schema) or ""
                if self._is_agent_tool(name):
                    if schema not in picked:
                        if len(picked) >= max_tools:
                            picked = picked[:-1]
                        picked.append(schema)
                    break

        return picked

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Extract first JSON object from model output."""
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    async def _plan_tool_policy(
        self,
        user_message: str,
        tool_schemas: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Plan tool policy dynamically with a router LLM call (no hardcoded intent map)."""
        tool_names = [
            self._tool_name_from_schema(schema)
            for schema in tool_schemas
            if self._tool_name_from_schema(schema)
        ]
        if not tool_names:
            return {
                "intent": "general",
                "must_use_tools": False,
                "tool_schemas": tool_schemas,
                "first_turn_tool_choice": "auto",
                "retry_with_required_on_empty": False,
            }

        default_policy = {
            "intent": "general",
            "must_use_tools": False,
            "tool_schemas": tool_schemas,
            "first_turn_tool_choice": "auto",
            "retry_with_required_on_empty": False,
        }

        tool_lines = []
        for schema in tool_schemas:
            name = self._tool_name_from_schema(schema)
            if not name:
                continue
            desc = schema.get("function", {}).get("description", "")
            tool_lines.append(f"- {name}: {desc}")

        router_messages = [
            {"role": "system", "content": self._ROUTER_POLICY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User message:\n{user_message}\n\n"
                    f"Available tools:\n" + "\n".join(tool_lines) + "\n\n"
                    "Return JSON only."
                ),
            },
        ]

        try:
            router_response = await self.llm_client.chat_completion(messages=router_messages)
            router_json = self._extract_json_object(router_response.content or "")
            if not router_json:
                return default_policy
        except Exception as e:
            logger.warning(f"[ToolPolicy] router failed, falling back to auto: {e}")
            return default_policy

        selected_names_raw = router_json.get("selected_tools", [])
        selected_names = [
            str(name)
            for name in selected_names_raw
            if isinstance(name, str) and name in tool_names
        ]

        must_use_tools = bool(router_json.get("must_use_tools"))
        reason_code = str(router_json.get("reason_code", "") or "").strip().lower()
        no_tool_reasons = {
            "casual_chat",
            "creative_writing",
            "language_translation",
            "text_rewrite",
            "pure_math",
        }
        if not must_use_tools and reason_code not in no_tool_reasons:
            must_use_tools = True

        force_first_tool = router_json.get("force_first_tool")
        first_turn_tool_choice: Any = "auto"

        if isinstance(force_first_tool, str) and force_first_tool in tool_names:
            selected_schemas = self._filter_tool_schemas(tool_schemas, [force_first_tool])
            first_turn_tool_choice = {
                "type": "function",
                "function": {"name": force_first_tool},
            }
            must_use_tools = True
        elif selected_names:
            selected_schemas = self._filter_tool_schemas(tool_schemas, selected_names)
            if must_use_tools:
                first_turn_tool_choice = "required"
        elif must_use_tools:
            selected_schemas = self._choose_fallback_tools(user_message, tool_schemas, max_tools=8)
            first_turn_tool_choice = "required"
        else:
            selected_schemas = tool_schemas

        intent = str(router_json.get("intent", "domain"))
        return {
            "intent": intent,
            "must_use_tools": must_use_tools,
            "tool_schemas": selected_schemas,
            "first_turn_tool_choice": first_turn_tool_choice,
            "retry_with_required_on_empty": must_use_tools,
        }

    def _register_memory_tool(self) -> None:
        """Register recall_memory as a tool so the LLM can query long-term memory on demand."""
        momex = self.momex

        async def recall_memory_executor(args: dict, context: ToolExecutionContext = None) -> str:
            query = args.get("query", "")
            if not query:
                return "Error: query is required"
            tenant_id = context.user_id if context else "default"
            results = await momex.search(tenant_id=tenant_id, query=query, limit=5)
            if not results:
                return "No relevant memories found."
            lines = []
            for r in results:
                text = r.get("memory", r.get("text", str(r)))
                lines.append(f"- {text}")
            return "\n".join(lines)

        registry = ToolRegistry.get_instance()
        registry.register(ToolDefinition(
            name="recall_memory",
            description=(
                "Search the user's long-term memory for past preferences, facts, or context. "
                "Use when the user refers to something from a previous conversation, or when "
                "personalization would improve the response (e.g. dietary preferences, travel habits)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the user's memory",
                    }
                },
                "required": ["query"],
            },
            executor=recall_memory_executor,
            category=ToolCategory.USER,
        ))
        logger.info("recall_memory tool registered")

    # ==========================================================================
    # EXTENSION POINTS - Override these in subclasses
    # ==========================================================================

    async def prepare_context(
        self,
        tenant_id: str,
        message: str,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Prepare context for processing.

        Automatically loads conversation history and recalls relevant
        long-term memories from Momex.

        Override to add:
        - User preferences/tier info
        - Custom metadata

        Args:
            tenant_id: Tenant identifier
            message: User message
            metadata: Request metadata

        Returns:
            Context dict passed to all subsequent methods
        """
        # Lazy restore if needed
        if (self.config.session.lazy_restore and
            not self.agent_pool.has_agents_in_memory(tenant_id)):
            await self._restore_tenant_session(tenant_id)

        # Get active agents
        active_agents = await self.agent_pool.list_agents(tenant_id)

        meta = metadata or {}
        session_id = meta.get("session_id", tenant_id)

        context: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "message": message,
            "metadata": meta,
            "active_agents": active_agents,
        }

        # Load conversation history (short-term memory, in-memory cache)
        history = self.momex.get_history(
            tenant_id=tenant_id,
            session_id=session_id,
        )
        if history:
            context["conversation_history"] = history

        return context

    async def should_process(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Check if message should be processed.

        Built-in checks (when configured via __init__):
        - guardrails_checker: safety/content filter
        - rate_limiter: per-tenant rate limiting

        Override to add:
        - Tier access control
        - Feature flags
        - Input validation

        Args:
            message: User message
            context: Context from prepare_context()

        Returns:
            True to continue processing, False to reject
        """
        # Guardrails check
        if self.guardrails_checker:
            try:
                safety_result = await self.guardrails_checker.check_input(message)
                if safety_result.get("blocked"):
                    context["rejection_reason"] = "blocked"
                    context["rejection_detail"] = safety_result.get("reason", "")
                    logger.warning(f"Input blocked by guardrails: {safety_result.get('reason')}")
                    return False
            except Exception as e:
                logger.error(f"Guardrails check failed: {e}")

        # Rate limiter check
        if self.rate_limiter:
            try:
                tenant_id = context["tenant_id"]
                limit_result = await self.rate_limiter(tenant_id, context)
                if not limit_result.get("allowed", True):
                    context["rejection_reason"] = "rate_limited"
                    context["rate_limit_info"] = limit_result
                    logger.warning(f"Rate limited: tenant={tenant_id}")
                    return False
            except Exception as e:
                logger.error(f"Rate limiter check failed: {e}")

        return True

    async def reject_message(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Handle rejected messages (when should_process returns False).

        Override to provide custom rejection response.

        Args:
            message: Original message
            context: Context from prepare_context()

        Returns:
            AgentResult - subclasses define the response
        """
        return AgentResult(
            agent_type=self.__class__.__name__,
            status=AgentStatus.COMPLETED,
        )

    async def create_agent(
        self,
        tenant_id: str,
        agent_type: str,
        context_hints: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
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
                    f"reached for tenant {tenant_id}"
                )
                return None

            agent = self._agent_registry.create_agent(
                name=agent_type,
                tenant_id=tenant_id,
                checkpoint_manager=self.checkpoint_manager,
                message_hub=self.message_hub,
                orchestrator_callback=self._create_callback_invoker(tenant_id),
                context_hints=context_hints,
            )

            if not agent:
                logger.error(f"Agent type not found: {agent_type}")
                return None

            # Fallback: if agent has no LLM, use orchestrator's
            if not agent.llm_client:
                agent.llm_client = self.llm_client

            # Add to pool
            await self.agent_pool.add_agent(agent)

            logger.debug(f"Created agent {agent.agent_id} of type {agent_type}")
            return agent

        except Exception as e:
            logger.error(f"Failed to create agent {agent_type}: {e}")
            return None

    async def post_process(
        self,
        result: AgentResult,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Post-process result before returning to user.

        Automatically saves conversation history and extracts long-term
        knowledge via Momex.  Then runs guardrails output check and
        any registered post_process_hooks.

        Override to add:
        - Send notifications (SMS, push, email)
        - Wrap with personality/style
        - Add analytics/logging
        - Record API usage

        Or use post_process_hooks (passed at __init__) to avoid subclassing:
        - Profile detection as background task
        - Usage recording
        - Response wrapping / personality layer

        Args:
            result: Agent result
            context: Context dict

        Returns:
            Modified result
        """
        tenant_id = context["tenant_id"]
        session_id = context.get("session_id", tenant_id)
        user_message = context.get("message", "")

        # Build conversation messages for storage
        messages = []
        if user_message:
            messages.append({"role": "user", "content": user_message})
        if result.raw_message:
            messages.append({"role": "assistant", "content": result.raw_message})

        if messages:
            # Save conversation history (short-term, sync)
            for msg in messages:
                self.momex.save_message(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    content=msg["content"],
                    role=msg["role"],
                )

            # Long-term knowledge extraction (async)
            await self.momex.add(
                tenant_id=tenant_id,
                messages=messages,
                infer=True,
            )

        # Guardrails output check
        if self.guardrails_checker and result.raw_message:
            try:
                safety_result = await self.guardrails_checker.check_output(
                    result.raw_message, tenant_id,
                )
                if safety_result.get("modified"):
                    result.raw_message = safety_result.get("output", result.raw_message)
            except Exception as e:
                logger.error(f"Guardrails output check failed: {e}")

        # Run registered post-process hooks
        for hook in self._post_process_hooks:
            try:
                result = await hook(result, context)
            except Exception as e:
                logger.error(f"Post-process hook {hook.__name__} failed: {e}")

        return result

    # ==========================================================================
    # CALLBACK SYSTEM
    # ==========================================================================

    def _create_callback_invoker(self, tenant_id: str) -> Callable:
        """
        Create the callback function for an agent.

        Args:
            tenant_id: Tenant ID to bind to callbacks from this agent

        Returns:
            Async function that agents call to invoke registered handlers
        """
        async def invoke_callback(
            name: str,
            data: Optional[Dict[str, Any]] = None
        ) -> Any:
            callback = AgentCallback(
                event=name,
                tenant_id=tenant_id,
                data=data or {}
            )
            return await self.handle_callback(callback)

        return invoke_callback

    async def handle_callback(self, callback: AgentCallback) -> Any:
        """
        Handle a callback from an agent.

        Looks up the registered handler by callback.event name and executes it.
        Override this method to add custom pre/post processing or fallback logic.
        """
        method_name = self._callback_handler_map.get(callback.event)
        if method_name is None:
            logger.warning(f"No callback handler registered for '{callback.event}'")
            return None

        handler = getattr(self, method_name, None)
        if handler is None:
            logger.error(f"Callback handler method '{method_name}' not found")
            return None

        try:
            return await handler(callback)
        except Exception as e:
            logger.error(f"Callback handler '{callback.event}' failed: {e}")
            return None

    def list_callbacks(self) -> List[str]:
        """List all registered callback handler names."""
        return list(self._callback_handler_map.keys())

    # ==========================================================================
    # BUILT-IN CALLBACK HANDLERS
    # ==========================================================================

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
            result.append({
                "name": name,
                "description": metadata.description,
                "capabilities": getattr(metadata, "capabilities", []),
            })
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

    # ==========================================================================
    # SESSION RESTORATION
    # ==========================================================================

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

    # ==========================================================================
    # AGENT MANAGEMENT API
    # ==========================================================================

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
                results.append({
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "agent_name": agent.agent_type,
                    "action_summary": getattr(agent, 'raw_message', '') or f"{agent.agent_type} awaiting approval",
                    "source": "user",
                    "created_at": getattr(agent, 'created_at', None),
                })

        # TriggerEngine: tasks pending approval
        if self.trigger_engine:
            pending_tasks = await self.trigger_engine.list_pending_approvals(tenant_id)
            for task in pending_tasks:
                results.append({
                    "task_id": task.id,
                    "task_name": task.name,
                    "agent_name": task.name,
                    "action_summary": getattr(task, 'description', '') or task.name,
                    "source": "trigger",
                    "trigger_type": task.trigger.type.value,
                })

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

    async def get_agent_status(
        self,
        tenant_id: str,
        agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific agent."""
        agent = await self.agent_pool.get_agent(tenant_id, agent_id)
        if not agent:
            return None
        return agent.get_state_summary()

    async def cancel_agent(
        self,
        tenant_id: str,
        agent_id: str
    ) -> bool:
        """Cancel an agent."""
        agent = await self.agent_pool.get_agent(tenant_id, agent_id)
        if agent:
            agent.status = AgentStatus.CANCELLED
            await self.agent_pool.remove_agent(tenant_id, agent_id)
            return True
        return False

    async def pause_agent(
        self,
        tenant_id: str,
        agent_id: str
    ) -> Optional[AgentResult]:
        """Pause an agent."""
        agent = await self.agent_pool.get_agent(tenant_id, agent_id)
        if not agent:
            return None

        pauseable_states = {
            AgentStatus.RUNNING,
            AgentStatus.WAITING_FOR_INPUT,
            AgentStatus.WAITING_FOR_APPROVAL,
            AgentStatus.INITIALIZING
        }

        if agent.status not in pauseable_states:
            logger.warning(f"Cannot pause agent {agent_id} in {agent.status} state")
            return None

        result = agent.pause()
        await self.agent_pool.update_agent(agent)
        return result

    async def resume_agent(
        self,
        tenant_id: str,
        agent_id: str,
        message: Optional[str] = None
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










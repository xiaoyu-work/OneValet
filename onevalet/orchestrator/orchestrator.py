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
from .react_config import (
    ReactLoopConfig, ToolCallRecord, TokenUsage,
    COMPLETE_TASK_TOOL_NAME, COMPLETE_TASK_SCHEMA, CompleteTaskResult,
)
from ..constants import GENERATE_PLAN_TOOL_NAME, GENERATE_PLAN_SCHEMA
from .context_manager import ContextManager
from .agent_tool import execute_agent_tool, AgentToolResult
from .approval import collect_batch_approvals
from .prompts import build_system_prompt, DEFAULT_SYSTEM_PROMPT
from .audit_logger import AuditLogger
from .tool_policy import ToolPolicyFilter
from .transcript_repair import repair_transcript

if TYPE_CHECKING:
    from ..checkpoint import CheckpointManager
    from ..llm.router import ModelRouter
    from ..msghub import MessageHub
    from ..protocols import LLMClientProtocol
    from ..memory.momex import MomexMemory

from ..models import AgentTool, AgentToolContext
from ..standard_agent import StandardAgent
from ..config import AgentRegistry

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
        model_router: Optional["ModelRouter"] = None,
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
            model_router: Optional ``ModelRouter`` instance for complexity-based
                model routing.  When provided, the first turn of each ReAct loop
                classifies the request and selects a provider from the
                ``LLMRegistry``.  Subsequent turns reuse the same provider.
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
        self.agent_pool = AgentPoolManager(
            config=self.config.session,
            database=database,
        )

        # Extension hooks
        self.guardrails_checker = guardrails_checker
        self.rate_limiter = rate_limiter
        self._post_process_hooks: List[Callable] = list(post_process_hooks or [])
        self._tool_policy_filter = tool_policy_filter
        self._model_router = model_router

        # Audit logging
        self._audit = AuditLogger()

        # State
        self._initialized = False
        self._pending_plan: Optional[Dict[str, Any]] = None
        self._current_metadata: Dict[str, Any] = {}
        self._current_user_images: Optional[List[Dict[str, Any]]] = None

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

        # Build orchestrator's builtin tools
        self.builtin_tools = self._build_builtin_tools()

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
        images: Optional[List[Dict[str, Any]]] = None,
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
            images: Optional list of image dicts (type, data, media_type)
            metadata: Optional message metadata

        Returns:
            AgentResult with response
        """
        if not self._initialized:
            await self.initialize()

        # Store request metadata for tool execution context
        self._current_metadata = metadata or {}
        self._current_user_images = images

        # Step 1: Prepare context
        context = await self.prepare_context(tenant_id, message, metadata)
        self._current_context = context

        # Store images in context so agent tools can access them (e.g. receipt scanning)
        if images:
            context["user_images"] = images

        # Step 2: Check if should process
        if not await self.should_process(message, context):
            return await self.reject_message(message, context)

        # Step 3: Check pending agents (WAITING_FOR_INPUT / WAITING_FOR_APPROVAL)
        agent_result = await self._check_pending_agents(tenant_id, message, context)
        if agent_result is not None:
            # Agent still waiting -> return prompt directly, don't enter ReAct
            if agent_result.status in (AgentStatus.WAITING_FOR_INPUT, AgentStatus.WAITING_FOR_APPROVAL):
                return await self.post_process(agent_result, context)
            # Agent completed -> return result directly.
            # The user's message was a response to the pending agent (e.g. an
            # approval like "yes"/"ok"), NOT a new task.  Feeding it into the ReAct
            # loop would cause the orchestrator to misinterpret the approval
            # word as a brand-new request and spawn unnecessary follow-up agents.
            return await self.post_process(agent_result, context)

        # Step 4: Intent Analysis — classify domains and detect multi-intent
        intent = await self._analyze_intent(message, context)

        # Step 4b: Multi-intent → DAG execution
        if intent.intent_type == "multi" and intent.sub_tasks:
            result = await self._execute_dag(intent, tenant_id, context, metadata)
            return await self.post_process(result, context)

        # Step 5: Build domain-filtered tool schemas
        tool_schemas = await self._build_tool_schemas(tenant_id, domains=intent.domains)

        # Step 5b: Inject notify_user tool for conditional cron delivery
        meta = metadata or {}
        if meta.get("cron_conditional_delivery"):
            notify_tool, notify_schema = self._build_notify_user_tool(context)
            self.builtin_tools.append(notify_tool)
            tool_schemas.append(notify_schema)

        logger.info(f"[Tools] {len(tool_schemas)} tools available for ReAct")

        # Step 6: Build LLM messages
        messages = await self._build_llm_messages(context, message)

        # Convert images to media format for LLM
        media = None
        if images:
            media = [
                {"type": "image", "data": img["data"], "media_type": img.get("media_type", "image/jpeg")}
                for img in images
            ]

        # Step 6: Run ReAct loop (consume events silently)
        exec_data: Dict[str, Any] = {}
        async for event in self._react_loop_events(
            messages,
            tool_schemas,
            tenant_id,
            context=context,
            user_message=message,
            media=media,
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

        result_metadata = {
            "react_turns": exec_data.get("turns", 0),
            "token_usage": exec_data.get("token_usage", {}),
            "duration_ms": exec_data.get("duration_ms", 0),
            "tool_calls_count": exec_data.get("tool_calls_count", 0),
            "total_tool_count": len(tool_schemas),
        }

        # Carry conditional notification from notify_user tool
        if context.get("cron_notification"):
            result_metadata["cron_notification"] = context["cron_notification"]

        # Clean up injected notify_user tool
        if meta.get("cron_conditional_delivery"):
            self.builtin_tools = [t for t in self.builtin_tools if t.name != "notify_user"]

        result = AgentResult(
            agent_type=self.__class__.__name__,
            status=status,
            raw_message=final_response,
            metadata=result_metadata,
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
        images: Optional[List[Dict[str, Any]]] = None,
        mode: StreamMode = StreamMode.EVENTS,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[AgentEvent]:
        """
        Stream agent execution events via ReAct loop.

        Same flow as handle_message but yielding streaming events at each stage.

        Args:
            tenant_id: Tenant identifier
            message: User message text
            images: Optional list of image dicts (type, data, media_type)
            mode: Stream mode
            metadata: Optional message metadata

        Yields:
            AgentEvent objects
        """
        if not self._initialized:
            await self.initialize()

        # Store request metadata for tool execution context
        self._current_metadata = metadata or {}
        self._current_user_images = images

        # Prepare context
        context = await self.prepare_context(tenant_id, message, metadata)
        self._current_context = context

        # Store images in context so agent tools can access them
        if images:
            context["user_images"] = images

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
            # Return the agent's result directly — whether still waiting or
            # completed.  The user's message was a response to the pending
            # agent (e.g. approval "yes"/"ok"), not a new task.  Entering the ReAct
            # loop would misinterpret it as a fresh request.
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

        # Intent Analysis — classify domains and detect multi-intent
        intent = await self._analyze_intent(message, context)

        # Multi-intent → streaming DAG execution
        if intent.intent_type == "multi" and intent.sub_tasks:
            final_response = ""
            async for event in self._stream_dag(intent, tenant_id, context, metadata):
                if event.type == EventType.EXECUTION_END:
                    final_response = event.data.get("final_response", "")
                yield event
            # Post-process: momex save, guardrails output, hooks
            result = AgentResult(
                agent_type=self.__class__.__name__,
                status=AgentStatus.COMPLETED,
                raw_message=final_response,
            )
            await self.post_process(result, context)
            return

        # Build domain-filtered tool schemas
        tool_schemas = await self._build_tool_schemas(tenant_id, domains=intent.domains)
        logger.info(f"[Tools] {len(tool_schemas)} tools available for ReAct")
        messages = await self._build_llm_messages(context, message)

        # Convert images to media format for LLM
        media = None
        if images:
            media = [
                {"type": "image", "data": img["data"], "media_type": img.get("media_type", "image/jpeg")}
                for img in images
            ]

        # Delegate to shared ReAct loop
        final_response = ""
        exec_data: Dict[str, Any] = {}
        async for event in self._react_loop_events(
            messages, tool_schemas, tenant_id,
            context=context, user_message=message, media=media,
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
        context: Optional[Dict[str, Any]] = None,
        user_message: str = "",
        media: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Unified ReAct loop implementation yielding streaming events.

        Both stream_message() and handle_message() delegate to this single
        implementation, eliminating the previous code duplication between
        the inline stream_message loop and react_loop().

        The final EXECUTION_END event carries all metadata (final_response,
        pending_approvals, token_usage, tool_calls records, etc.) so callers
        can build AgentResult or persist to memory as needed.
        """
        # --- Change A: Context window pre-flight guard ---
        CONTEXT_HARD_MIN = 16_000
        CONTEXT_WARN_BELOW = 32_000
        context_tokens = getattr(self.llm_client, 'context_window', 128_000)
        if context_tokens < CONTEXT_HARD_MIN:
            yield AgentEvent(
                type=EventType.ERROR,
                data={"message": f"Model context window too small: {context_tokens} tokens (minimum: {CONTEXT_HARD_MIN})"},
            )
            return
        if context_tokens < CONTEXT_WARN_BELOW:
            logger.warning(f"Low context window: {context_tokens} tokens")

        start_time = time.monotonic()
        turn = 0
        all_tool_records: List[ToolCallRecord] = []
        total_usage = TokenUsage()
        pending_approvals = []
        final_response = ""
        result_status = None
        _recent_tool_names: List[str] = []  # Change E: watchdog loop detection

        logger.info(f"[ReAct] tenant={tenant_id}")

        yield AgentEvent(
            type=EventType.EXECUTION_START,
            data={"tenant_id": tenant_id},
        )

        # Model routing: classify once before the loop, reuse for all turns.
        routed_llm_client = None
        routing_score = -1
        if self._model_router:
            try:
                from ..llm.router import RoutingDecision
                decision = await self._model_router.route(messages)
                routing_score = decision.score
                routed_llm_client = self._model_router.registry.get(decision.provider)
                if routed_llm_client:
                    logger.info(
                        f"[ReAct] ModelRouter selected provider='{decision.provider}' "
                        f"(score={decision.score}, {decision.latency_ms:.0f}ms)"
                    )
            except Exception as e:
                logger.warning(f"[ReAct] ModelRouter failed, using default LLM: {e}")

        # Enable reasoning for complex requests on the first turn
        enable_reasoning = routing_score >= self._react_config.reasoning_score_threshold
        if enable_reasoning:
            logger.info(f"[ReAct] Reasoning enabled (score={routing_score}, effort={self._react_config.reasoning_effort})")

        # ── Planning phase ──
        enable_planning = routing_score >= self._react_config.planning_score_threshold

        # Case 1: Pending plan from previous turn — user is responding to it
        if self._pending_plan and context:
            pending_plan_text = self._format_plan_text(self._pending_plan)
            self._pending_plan = None  # consumed
            logger.info("[ReAct] Pending plan found, injecting into prompt for LLM to handle")
            messages = await self._build_llm_messages(
                context, user_message, pending_plan=pending_plan_text,
            )
            enable_planning = False  # don't re-plan

        # Case 2: New complex request — generate plan and present to user
        elif enable_planning:
            logger.info(f"[ReAct] Planning phase triggered (score={routing_score})")
            try:
                plan_messages = await self._build_llm_messages(
                    context, user_message, include_planning=True,
                )
                plan_schemas = [GENERATE_PLAN_SCHEMA, COMPLETE_TASK_SCHEMA]
                plan_response = await self._llm_call_with_retry(
                    plan_messages, plan_schemas, tool_choice="auto",
                    llm_client_override=routed_llm_client,
                )
                plan_data = self._extract_plan_from_response(plan_response)
                if plan_data and self._react_config.planning_requires_approval:
                    # Present plan to user, pause execution
                    plan_text = self._format_plan_text(plan_data)
                    friendly = self._format_plan_for_user(plan_data)
                    self._pending_plan = plan_data
                    logger.info(f"[ReAct] Plan generated, awaiting approval: {plan_data.get('goal', '')}")
                    yield AgentEvent(
                        type=EventType.PLAN_GENERATED,
                        data={"plan": plan_data, "plan_text": plan_text},
                    )
                    # End this turn — return plan as the response
                    duration_ms = int((time.monotonic() - start_time) * 1000)
                    yield AgentEvent(
                        type=EventType.EXECUTION_END,
                        data={
                            "final_response": friendly,
                            "result_status": "WAITING_FOR_APPROVAL",
                            "turns": 0,
                            "tool_calls": [],
                            "token_usage": {"input_tokens": 0, "output_tokens": 0},
                            "duration_ms": duration_ms,
                            "pending_approvals": [],
                        },
                    )
                    return  # stop the generator — user needs to respond

                elif plan_data:
                    # Auto-execute without approval
                    plan_text = self._format_plan_text(plan_data)
                    yield AgentEvent(
                        type=EventType.PLAN_GENERATED,
                        data={"plan": plan_data, "plan_text": plan_text},
                    )
                    logger.info(f"[ReAct] Plan auto-approved: {plan_data.get('goal', '')}")
                    messages = await self._build_llm_messages(
                        context, user_message, approved_plan=plan_text,
                    )
                else:
                    logger.info("[ReAct] LLM did not generate a plan, proceeding directly")
            except Exception as e:
                logger.warning(f"[ReAct] Planning phase failed, proceeding without plan: {e}")

        for turn in range(1, self._react_config.max_turns + 1):
            # Context guard with summarization
            messages = await self._summarize_and_trim(messages)

            # Change B: Transcript repair before LLM call
            messages = repair_transcript(messages)

            # LLM call
            try:
                tool_choice = first_turn_tool_choice if turn == 1 else "auto"
                # Enable reasoning only on the first turn for complex requests
                extra_kwargs = {}
                if enable_reasoning and turn == 1:
                    extra_kwargs["reasoning_effort"] = self._react_config.reasoning_effort
                # Pass images only on the first turn
                if media and turn == 1:
                    extra_kwargs["media"] = media
                response = await self._llm_call_with_retry(
                    messages, tool_schemas, tool_choice=tool_choice,
                    llm_client_override=routed_llm_client,
                    **extra_kwargs,
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

            # No tool calls → LLM forgot to call complete_task.
            # Retry up to max_complete_task_retries times with tool_choice="required".
            if not tool_calls:
                max_retries = self._react_config.max_complete_task_retries
                for retry in range(1, max_retries + 1):
                    grace_msg = (
                        "You must call the `complete_task` tool with your final "
                        "response in the `result` parameter to finish. Do not "
                        "respond with plain text. Call `complete_task` now."
                    )
                    logger.warning(
                        f"[ReAct] turn={turn} no tool calls, "
                        f"grace retry {retry}/{max_retries}"
                    )
                    messages.append(self._assistant_message_from_response(response))
                    messages.append({"role": "user", "content": grace_msg})
                    try:
                        response = await self._llm_call_with_retry(
                            messages, tool_schemas, tool_choice="required",
                            llm_client_override=routed_llm_client,
                        )
                    except Exception as e:
                        yield AgentEvent(
                            type=EventType.ERROR,
                            data={"error": str(e), "error_type": type(e).__name__},
                        )
                        return
                    usage_retry = getattr(response, "usage", None)
                    if usage_retry:
                        total_usage.input_tokens += getattr(usage_retry, "prompt_tokens", 0)
                        total_usage.output_tokens += getattr(usage_retry, "completion_tokens", 0)
                    tool_calls = response.tool_calls
                    if tool_calls:
                        break  # success — proceed to tool execution

                # Exhausted all retries — ask LLM to produce a user-friendly
                # error message (no tools, so it can only return text).
                if not tool_calls:
                    logger.error(
                        f"[ReAct] turn={turn} exhausted {max_retries} "
                        f"grace retries, LLM still did not call complete_task"
                    )
                    messages.append(self._assistant_message_from_response(response))
                    messages.append({
                        "role": "user",
                        "content": (
                            "There was an internal issue processing your request. "
                            "Generate a short, friendly apology to the user in "
                            "their language, and suggest they try again later."
                        ),
                    })
                    try:
                        fallback_resp = await self._llm_call_with_retry(
                            messages, tool_schemas=[], tool_choice=None,
                            llm_client_override=routed_llm_client,
                        )
                        final_response = (
                            getattr(fallback_resp, "content", None)
                            or fallback_resp.choices[0].message.content
                            or "Sorry, something went wrong. Please try again later."
                        )
                    except Exception:
                        final_response = "Sorry, something went wrong. Please try again later."
                    self._audit.log_react_turn(
                        turn=turn, tool_calls=[], final_answer=True,
                        tenant_id=tenant_id,
                    )
                    yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                    yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
                    yield AgentEvent(type=EventType.MESSAGE_END, data={})
                    return

            if tool_calls:
                # Append assistant message with tool_calls
                messages.append(self._assistant_message_from_response(response))

                # ----------------------------------------------------------
                # Intercept complete_task: handle synchronously, skip execution
                # ----------------------------------------------------------
                complete_task_result: Optional[CompleteTaskResult] = None
                remaining_tool_calls = []
                for tc in tool_calls:
                    if tc.name == COMPLETE_TASK_TOOL_NAME:
                        try:
                            _ct_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
                        except (json.JSONDecodeError, TypeError):
                            _ct_args = {}
                        _ct_text = _ct_args.get("result", "")
                        if _ct_text:
                            complete_task_result = CompleteTaskResult(result=_ct_text)
                            messages.append(self._build_tool_result_message(tc.id, "Task completed."))
                            all_tool_records.append(ToolCallRecord(
                                name=COMPLETE_TASK_TOOL_NAME,
                                args_summary={"result": _ct_text[:100]},
                                duration_ms=0, success=True,
                                result_status="COMPLETED",
                                result_chars=len(_ct_text),
                            ))
                            logger.info(f"[ReAct] turn={turn} complete_task called ({len(_ct_text)} chars)")
                        else:
                            # Missing result — append error, let LLM retry
                            messages.append(self._build_tool_result_message(
                                tc.id,
                                'Error: "result" argument is required for complete_task.',
                                is_error=True,
                            ))
                            remaining_tool_calls.append(tc)
                    else:
                        remaining_tool_calls.append(tc)

                # Pure complete_task with no other tools — break immediately
                if complete_task_result and not remaining_tool_calls:
                    final_response = complete_task_result.result
                    self._audit.log_react_turn(
                        turn=turn, tool_calls=[COMPLETE_TASK_TOOL_NAME],
                        final_answer=True, tenant_id=tenant_id,
                    )
                    yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                    yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
                    yield AgentEvent(type=EventType.MESSAGE_END, data={})
                    break

                # Use remaining tools for execution (or original list if complete_task had no result)
                tool_calls = remaining_tool_calls if remaining_tool_calls else tool_calls

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
                        # Change C: Hard cap on tool result size
                        result_text = self._cap_tool_result(result_text)
                        result_text = self._context_manager.truncate_tool_result(result_text)
                        # Change D: Context isolation for agent-tools
                        if is_agent and len(result_text) > 2000:
                            result_text = result_text[:1500] + "\n...[full result available in agent context]"
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

                # complete_task was called alongside other tools — use its result
                if complete_task_result:
                    final_response = complete_task_result.result
                    self._audit.log_react_turn(
                        turn=turn, tool_calls=tool_names + [COMPLETE_TASK_TOOL_NAME],
                        final_answer=True, tenant_id=tenant_id,
                    )
                    yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                    yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
                    yield AgentEvent(type=EventType.MESSAGE_END, data={})
                    break

                # Change E: Watchdog loop detection
                for tn in tool_names:
                    _recent_tool_names.append(tn)
                if len(_recent_tool_names) >= 3:
                    last_3 = _recent_tool_names[-3:]
                    if len(set(last_3)) == 1:
                        logger.warning(
                            f"[ReAct] Loop detected: {last_3[0]} called 3 times "
                            f"consecutively, breaking ReAct loop"
                        )
                        final_response = (
                            f"I noticed I was repeating the same action ({last_3[0]}) "
                            f"without making progress. Let me provide what I have so far."
                        )
                        yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                        yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
                        yield AgentEvent(type=EventType.MESSAGE_END, data={})
                        break

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
                response = await self._llm_call_with_retry(
                    messages, tool_schemas=None,
                    llm_client_override=routed_llm_client,
                )
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
        llm_client_override: Optional[Any] = None,
        **extra_kwargs,
    ) -> Any:
        """LLM call with error recovery and model fallback chain.

        Recovery strategy:
        - RateLimitError -> exponential backoff
        - ContextOverflowError -> three-step recovery (trim -> truncate_all -> force_trim)
        - AuthError -> raise immediately
        - TimeoutError -> retry once

        If all retries on the primary client are exhausted, tries each
        fallback provider from ``ReactLoopConfig.fallback_providers`` in order.

        Args:
            tool_choice: Override for tool_choice param ("auto", "required", "none").
                         If None, the LLM client uses its default ("auto").
            llm_client_override: Optional LLM client to use instead of
                ``self.llm_client``.  Set by the model router when
                complexity-based routing is active.
        """
        client = llm_client_override or self.llm_client
        primary_error = await self._llm_call_single_client(
            client, messages, tool_schemas, tool_choice, **extra_kwargs,
        )
        if not isinstance(primary_error, Exception):
            return primary_error  # success — it's an LLMResponse

        # Primary failed — try fallback providers
        fallback_providers = self._react_config.fallback_providers
        if fallback_providers and self._model_router:
            registry = self._model_router.registry
            for provider_name in fallback_providers:
                fallback_client = registry.get(provider_name)
                if fallback_client is None or fallback_client is client:
                    continue
                logger.warning(f"[LLM] Primary failed, trying fallback provider: {provider_name}")
                result = await self._llm_call_single_client(
                    fallback_client, messages, tool_schemas, tool_choice, **extra_kwargs,
                )
                if not isinstance(result, Exception):
                    return result
                logger.warning(f"[LLM] Fallback provider {provider_name} also failed: {result}")

        raise primary_error

    async def _llm_call_single_client(
        self,
        client: Any,
        messages: List[Dict[str, Any]],
        tool_schemas: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any] = None,
        **extra_kwargs,
    ) -> Any:
        """Try a single LLM client with retries.

        Returns the LLMResponse on success, or the last Exception on failure.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self._react_config.llm_max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {"messages": messages, **extra_kwargs}
                if tool_schemas:
                    kwargs["tools"] = tool_schemas
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice
                    logger.info(f"[LLM] Sending {len(tool_schemas)} tools, tool_choice={tool_choice or 'auto'}, sample: {json.dumps(tool_schemas[0], ensure_ascii=False)[:200]}")
                else:
                    logger.info("[LLM] Sending request with NO tools")
                response = await client.chat_completion(**kwargs)
                # Debug: log what came back
                tc = getattr(response, 'tool_calls', None)
                sr = getattr(response, 'stop_reason', None)
                content_len = len(getattr(response, 'content', '') or '')
                logger.info(f"[LLM] Response: stop_reason={sr}, tool_calls={len(tc) if tc else 0}, content_len={content_len}")
                return response

            except Exception as e:
                last_error = e
                error_name = type(e).__name__.lower()

                # Auth errors: raise immediately (no fallback can help)
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
                    break  # let fallback chain handle it

                # Unknown error: retry with backoff
                if attempt < self._react_config.llm_max_retries:
                    delay = self._react_config.llm_retry_base_delay * (2 ** attempt)
                    logger.warning(f"LLM call failed ({e}), retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue

                break  # exhausted retries, let fallback chain handle it

        return last_error  # type: ignore[return-value]

    # ── Planning helpers ──

    @staticmethod
    def _extract_plan_from_response(response: Any) -> Optional[Dict[str, Any]]:
        """Extract generate_plan tool call arguments from an LLM response."""
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return None
        for tc in tool_calls:
            if tc.name == GENERATE_PLAN_TOOL_NAME:
                args = tc.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                return args
        return None

    @staticmethod
    def _format_plan_text(plan_data: Dict[str, Any]) -> str:
        """Format structured plan data into readable text for prompt injection."""
        lines = [f"**Goal:** {plan_data.get('goal', '')}"]
        for step in plan_data.get("steps", []):
            deps = step.get("depends_on", [])
            dep_str = (
                f" (after step {', '.join(map(str, deps))})"
                if deps
                else " (can start immediately)"
            )
            lines.append(
                f"{step['id']}. [{step.get('agent', '?')}] {step['action']}{dep_str}"
            )
            if step.get("reason"):
                lines.append(f"   Reason: {step['reason']}")
        return "\n".join(lines)

    @staticmethod
    def _format_plan_for_user(plan_data: Dict[str, Any]) -> str:
        """Format plan as a friendly user-facing message."""
        lines = [plan_data.get("goal", "")]
        lines.append("")
        for step in plan_data.get("steps", []):
            deps = step.get("depends_on", [])
            if deps:
                dep_str = f" (after step {', '.join(map(str, deps))})"
            else:
                dep_str = ""
            lines.append(f"{step['id']}. {step['action']}{dep_str}")
        lines.append("")
        lines.append("Ready to execute. You can approve, modify, or cancel.")
        return "\n".join(lines)

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
            # Builtin tool execution
            tool = next((t for t in getattr(self, 'builtin_tools', []) if t.name == tool_name), None)
            if not tool:
                return f"Error: Tool '{tool_name}' not found"

            context = AgentToolContext(
                tenant_id=tenant_id,
                credentials=self.credential_store,
                metadata=self._build_tool_metadata(),
            )
            return await tool.executor(args, context)

    def _is_agent_tool(self, tool_name: str) -> bool:
        """Check if tool_name corresponds to a registered agent."""
        if not self._agent_registry:
            return False
        return self._agent_registry.get_agent_class(tool_name) is not None

    def _build_tool_metadata(self) -> dict:
        """Build metadata dict for regular tool execution context."""
        # Start with request-level metadata (contains location, timezone, etc.)
        meta = dict(self._current_metadata)
        if self.database:
            from onevalet.builtin_agents.digest.important_dates_repo import ImportantDatesRepository
            meta["important_dates_store"] = ImportantDatesRepository(self.database)
        return meta

    HARD_MAX_TOOL_RESULT_CHARS = 400_000

    def _cap_tool_result(self, result_text: str) -> str:
        """Hard cap on tool result size to prevent context window overflow."""
        if len(result_text) <= self.HARD_MAX_TOOL_RESULT_CHARS:
            return result_text
        cut = self.HARD_MAX_TOOL_RESULT_CHARS
        newline_pos = result_text.rfind("\n", int(cut * 0.8), cut)
        if newline_pos > 0:
            cut = newline_pos
        logger.warning(
            f"[ReAct] Tool result truncated: {len(result_text)} -> {cut} chars"
        )
        return result_text[:cut] + "\n\n[truncated - result exceeded size limit]"

    async def _summarize_and_trim(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Summarize old messages via LLM before trimming, preserving context.

        If context is within threshold, returns messages unchanged.
        Otherwise, splits messages into old/recent, summarizes old via LLM,
        and replaces them with a single summary message.
        Falls back to simple trim if summarization fails.
        """
        split = self._context_manager.split_for_summarization(messages)
        if split is None:
            return messages

        system_msgs, old_msgs, recent_msgs = split

        # Build a compact representation of old messages for summarization
        old_text_parts = []
        for msg in old_msgs:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                # Truncate very long individual messages for the summary request
                if len(content) > 500:
                    content = content[:497] + "..."
                old_text_parts.append(f"{role}: {content}")

        if not old_text_parts:
            return self._context_manager.trim_if_needed(messages)

        old_text = "\n".join(old_text_parts)
        # Cap the input to the summarizer to avoid nested overflow
        if len(old_text) > 8000:
            old_text = old_text[:8000] + "\n...[truncated]"

        try:
            summary_response = await self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the following conversation excerpt in 2-4 sentences. "
                            "Preserve key facts, decisions, entities, and user preferences. "
                            "Be concise but retain actionable context."
                        ),
                    },
                    {"role": "user", "content": old_text},
                ],
            )
            summary = (summary_response.content or "").strip()
            if summary:
                logger.info(
                    f"[Context] Summarized {len(old_msgs)} old messages "
                    f"({len(old_text)} chars -> {len(summary)} chars)"
                )
                return self._context_manager.build_summarized_messages(
                    system_msgs, summary, recent_msgs,
                )
        except Exception as e:
            logger.warning(f"[Context] Summarization failed, falling back to trim: {e}")

        return self._context_manager.trim_if_needed(messages)

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

    async def _build_llm_messages(
        self,
        context: Dict[str, Any],
        user_message: str,
        *,
        include_planning: bool = False,
        approved_plan: str = "",
        pending_plan: str = "",
    ) -> List[Dict[str, Any]]:
        """Build the initial LLM message list.

        Contains:
        - System prompt + recalled memories
        - Conversation history (from Momex short-term memory)
        - Current user message

        Args:
            include_planning: If True, adds planning instructions to prompt.
            approved_plan: If set, injects the approved plan for execution.
            pending_plan: If set, injects a pending plan awaiting user response.
        """
        messages: List[Dict[str, Any]] = []

        # Dynamic system prompt: built from live agent registry
        agent_descriptions = ""
        if self._agent_registry:
            try:
                agent_descriptions = await self._agent_registry.get_agent_descriptions(
                    tenant_id=context.get("tenant_id"),
                    credential_store=self.credential_store,
                )
            except Exception as e:
                logger.warning(f"Failed to get agent descriptions: {e}")

        system_prompt = build_system_prompt(
            agent_descriptions=agent_descriptions,
            include_planning=include_planning,
            approved_plan=approved_plan,
            pending_plan=pending_plan,
        )

        system_parts = [system_prompt]
        if self.system_prompt:
            system_parts.append(self.system_prompt)

        # Runtime context
        now = datetime.now(timezone.utc)
        context_lines = [f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"]

        # Add user location if available in metadata
        meta = context.get("metadata") or {}
        location = meta.get("location")
        if location and isinstance(location, dict):
            lat = location.get("lat")
            lng = location.get("lng")
            if lat is not None and lng is not None:
                place = location.get("place_name", "")
                try:
                    loc_str = f"User location: {float(lat):.4f}, {float(lng):.4f}"
                except (TypeError, ValueError):
                    loc_str = f"User location: {lat}, {lng}"
                if place:
                    loc_str += f" ({place})"
                context_lines.append(loc_str)

        tz = meta.get("timezone")
        if tz and tz != "UTC":
            context_lines.append(f"User timezone: {tz}")

        system_parts.append("\n[Context]\n" + "\n".join(context_lines))

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

        return messages

    # Domain-based routing is now driven by @valet(domain=...) on each agent.
    # See AgentRegistry.get_domain_agent_tool_schemas() for the filtering logic.

    async def _build_tool_schemas(
        self,
        tenant_id: str,
        domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build combined tool schemas: builtin tools + domain-filtered agent-tools.

        Args:
            tenant_id: Tenant identifier for credential filtering.
            domains: List of domains to load agent tools for.
                If ``None`` or contains ``"all"``, loads all agent tools.
        """
        schemas: List[Dict[str, Any]] = []

        if not self._agent_registry:
            return schemas

        # Builtin tools (orchestrator-level, always included)
        for tool in getattr(self, 'builtin_tools', []):
            schemas.append(tool.to_openai_schema())

        # Agent-tools: domain-filtered
        if domains and "all" not in domains:
            agent_tool_schemas = await self._agent_registry.get_domain_agent_tool_schemas(
                domains=domains,
                tenant_id=tenant_id,
                credential_store=self.credential_store,
            )
        else:
            agent_tool_schemas = await self._agent_registry.get_all_agent_tool_schemas(
                tenant_id=tenant_id,
                credential_store=self.credential_store,
            )

        schemas.extend(agent_tool_schemas)

        # Always inject complete_task as the last tool
        schemas.append(COMPLETE_TASK_SCHEMA)

        # Apply tool policy filter if configured
        if self._tool_policy_filter:
            schemas = self._tool_policy_filter.filter_tools(schemas, tenant_id=tenant_id)

        logger.info(
            f"[Tools] {len(schemas)} total available "
            f"(agents={len(agent_tool_schemas)}, domains={domains or ['all']})"
        )

        return schemas

    # ==================================================================
    # INTENT ANALYSIS & DAG EXECUTION
    # ==================================================================

    async def _analyze_intent(
        self,
        message: str,
        context: Dict[str, Any],
    ) -> "IntentAnalysis":
        """Analyze user message to determine intent type and domain(s).

        Single lightweight LLM call (~200 tokens).
        Falls back to all-domains on failure.
        """
        from .intent_analyzer import IntentAnalyzer

        analyzer = IntentAnalyzer(self.llm_client)
        intent = await analyzer.analyze(message)

        logger.info(
            f"[IntentAnalyzer] type={intent.intent_type}, "
            f"domains={intent.domains}, "
            f"sub_tasks={len(intent.sub_tasks)}"
        )
        return intent

    async def _execute_dag(
        self,
        intent: "IntentAnalysis",
        tenant_id: str,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Execute multi-intent sub-tasks in DAG order.

        1. Topologically sort sub-tasks into execution levels
        2. Execute each level (parallel within, sequential across)
        3. Inject predecessor results into dependent tasks
        4. Synthesize final response
        """
        from .dag_executor import topological_sort, SubTaskResult

        start_time = time.monotonic()
        levels = topological_sort(intent.sub_tasks)
        all_results: Dict[int, SubTaskResult] = {}

        logger.info(
            f"[DAG] Starting execution: {len(intent.sub_tasks)} sub-tasks, "
            f"{len(levels)} levels"
        )

        for level_idx, level in enumerate(levels):
            logger.info(
                f"[DAG] Level {level_idx}: executing {len(level)} sub-task(s) "
                f"[{', '.join(st.description[:40] for st in level)}]"
            )

            async def _run_sub_task(sub_task):
                augmented_message = self._build_dag_augmented_message(
                    sub_task, all_results,
                )
                tool_schemas = await self._build_tool_schemas(
                    tenant_id, domains=[sub_task.domain],
                )
                messages = await self._build_llm_messages(context, augmented_message)

                exec_data: Dict[str, Any] = {}
                async for event in self._react_loop_events(
                    messages, tool_schemas, tenant_id,
                    context=context, user_message=augmented_message,
                ):
                    if event.type == EventType.EXECUTION_END:
                        exec_data = event.data

                return SubTaskResult(
                    sub_task_id=sub_task.id,
                    description=sub_task.description,
                    response=exec_data.get("final_response", ""),
                    status="completed",
                    duration_ms=exec_data.get("duration_ms", 0),
                    token_usage=exec_data.get("token_usage", {}),
                )

            level_results = await asyncio.gather(
                *[_run_sub_task(st) for st in level],
                return_exceptions=True,
            )

            for st, result in zip(level, level_results):
                if isinstance(result, BaseException):
                    logger.warning(f"[DAG] Sub-task {st.id} failed: {result}")
                    all_results[st.id] = SubTaskResult(
                        sub_task_id=st.id,
                        description=st.description,
                        response=f"Error: {result}",
                        status="error",
                    )
                else:
                    all_results[st.id] = result

        # Synthesize results into a unified response
        final_response = await self._synthesize_dag_results(
            intent.raw_message, all_results, context,
        )
        duration_ms = int((time.monotonic() - start_time) * 1000)

        logger.info(f"[DAG] Completed in {duration_ms}ms")

        return AgentResult(
            agent_type=self.__class__.__name__,
            status=AgentStatus.COMPLETED,
            raw_message=final_response,
            metadata={
                "dag_execution": True,
                "sub_tasks": len(intent.sub_tasks),
                "levels": len(levels),
                "duration_ms": duration_ms,
            },
        )

    async def _stream_dag(
        self,
        intent: "IntentAnalysis",
        tenant_id: str,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream events during DAG execution."""
        from .dag_executor import topological_sort, SubTaskResult

        start_time = time.monotonic()
        levels = topological_sort(intent.sub_tasks)
        all_results: Dict[int, SubTaskResult] = {}

        yield AgentEvent(
            type=EventType.WORKFLOW_START,
            data={"sub_tasks": len(intent.sub_tasks), "levels": len(levels)},
        )

        for level_idx, level in enumerate(levels):
            for st in level:
                yield AgentEvent(
                    type=EventType.STAGE_START,
                    data={
                        "sub_task_id": st.id,
                        "description": st.description,
                        "domain": st.domain,
                    },
                )

            if len(level) == 1:
                # Single task in level: stream events normally
                st = level[0]
                augmented_message = self._build_dag_augmented_message(
                    st, all_results,
                )
                tool_schemas = await self._build_tool_schemas(
                    tenant_id, domains=[st.domain],
                )
                messages = await self._build_llm_messages(context, augmented_message)

                exec_data: Dict[str, Any] = {}
                async for event in self._react_loop_events(
                    messages, tool_schemas, tenant_id,
                    context=context, user_message=augmented_message,
                ):
                    if event.type == EventType.EXECUTION_END:
                        exec_data = event.data
                    yield event

                all_results[st.id] = SubTaskResult(
                    sub_task_id=st.id,
                    description=st.description,
                    response=exec_data.get("final_response", ""),
                    status="completed",
                )
                yield AgentEvent(
                    type=EventType.STAGE_END,
                    data={"sub_task_id": st.id},
                )
            else:
                # Multiple parallel tasks: execute concurrently, collect results
                async def _run_silent(sub_task):
                    aug_msg = self._build_dag_augmented_message(
                        sub_task, all_results,
                    )
                    t_schemas = await self._build_tool_schemas(
                        tenant_id, domains=[sub_task.domain],
                    )
                    msgs = await self._build_llm_messages(context, aug_msg)
                    exec_d: Dict[str, Any] = {}
                    async for ev in self._react_loop_events(
                        msgs, t_schemas, tenant_id,
                        context=context, user_message=aug_msg,
                    ):
                        if ev.type == EventType.EXECUTION_END:
                            exec_d = ev.data
                    return SubTaskResult(
                        sub_task_id=sub_task.id,
                        description=sub_task.description,
                        response=exec_d.get("final_response", ""),
                        status="completed",
                    )

                level_results = await asyncio.gather(
                    *[_run_silent(st) for st in level],
                    return_exceptions=True,
                )
                for st, result in zip(level, level_results):
                    if isinstance(result, BaseException):
                        all_results[st.id] = SubTaskResult(
                            sub_task_id=st.id,
                            description=st.description,
                            response=f"Error: {result}",
                            status="error",
                        )
                    else:
                        all_results[st.id] = result
                    yield AgentEvent(
                        type=EventType.STAGE_END,
                        data={"sub_task_id": st.id},
                    )

        # Synthesis stage
        yield AgentEvent(
            type=EventType.STAGE_START,
            data={"sub_task_id": -1, "description": "Synthesizing results"},
        )
        final_response = await self._synthesize_dag_results(
            intent.raw_message, all_results, context,
        )
        yield AgentEvent(type=EventType.MESSAGE_START, data={})
        yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
        yield AgentEvent(type=EventType.MESSAGE_END, data={})
        yield AgentEvent(
            type=EventType.STAGE_END,
            data={"sub_task_id": -1},
        )

        yield AgentEvent(
            type=EventType.WORKFLOW_END,
            data={"sub_tasks_completed": len(all_results)},
        )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        yield AgentEvent(
            type=EventType.EXECUTION_END,
            data={
                "final_response": final_response,
                "dag_execution": True,
                "sub_tasks": len(intent.sub_tasks),
                "pending_approvals": [],
                "result_status": None,
                "turns": 0,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
                "duration_ms": duration_ms,
                "tool_calls": [],
            },
        )

    async def _synthesize_dag_results(
        self,
        original_message: str,
        results: Dict[int, "SubTaskResult"],
        context: Dict[str, Any],
    ) -> str:
        """Synthesize multiple sub-task results into a unified response."""
        # If only one sub-task completed successfully, return its result directly
        successful = [r for r in results.values() if r.status == "completed"]
        if len(successful) == 1:
            return successful[0].response

        result_parts = []
        for sub_id in sorted(results.keys()):
            r = results[sub_id]
            result_parts.append(
                f"## Sub-task {sub_id}: {r.description}\n{r.response}"
            )

        synthesis_message = (
            f'The user asked: "{original_message}"\n\n'
            "Here are the results from each sub-task:\n\n"
            + "\n\n".join(result_parts)
            + "\n\nSynthesize these into a single, coherent response for the user. "
            "Preserve all specific data points. Be concise."
        )

        messages = [
            {
                "role": "system",
                "content": "You synthesize multiple task results into a unified response.",
            },
            {"role": "user", "content": synthesis_message},
        ]

        try:
            response = await self.llm_client.chat_completion(messages=messages)
            return response.content or ""
        except Exception as e:
            logger.warning(f"[DAG] Synthesis failed: {e}")
            # Fallback: concatenate results
            return "\n\n".join(
                f"**{r.description}:**\n{r.response}"
                for r in sorted(results.values(), key=lambda r: r.sub_task_id)
            )

    @staticmethod
    def _build_dag_augmented_message(
        sub_task: "SubTask",
        prior_results: Dict[int, "SubTaskResult"],
    ) -> str:
        """Build an augmented user message with predecessor results injected."""
        if not sub_task.depends_on:
            return sub_task.description

        predecessor_context = []
        for dep_id in sub_task.depends_on:
            if dep_id in prior_results:
                pred = prior_results[dep_id]
                predecessor_context.append(
                    f"[Result from previous step: {pred.description}]\n{pred.response}"
                )

        if not predecessor_context:
            return sub_task.description

        return (
            "\n\n".join(predecessor_context)
            + f"\n\nBased on the above, please: {sub_task.description}"
        )

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
        """Plan tool policy dynamically with a router LLM call (no hardcoded intent map).

        The router sees all domain-filtered agent tools and selects the most
        relevant ones for the current user message.
        """
        tool_names = [
            self._tool_name_from_schema(schema)
            for schema in tool_schemas
            if self._tool_name_from_schema(schema)
        ]

        all_known_names = set(tool_names)

        if not all_known_names:
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

        # Show all available tools to the router
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
            if isinstance(name, str) and name in all_known_names
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

        if isinstance(force_first_tool, str) and force_first_tool in all_known_names:
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

    def _build_builtin_tools(self) -> List[Dict[str, Any]]:
        """Build the orchestrator's builtin tools as AgentTool instances.

        These are lightweight tools the orchestrator calls directly
        (not via an agent).
        """
        from ..builtin_agents.tools.google_search import (
            google_search_executor, GOOGLE_SEARCH_SCHEMA,
        )
        from ..builtin_agents.tools.web_fetch import (
            web_fetch_executor, WEB_FETCH_SCHEMA,
        )
        from ..builtin_agents.tools.important_dates import IMPORTANT_DATES_TOOL_DEFS
        from ..builtin_agents.tools.user_tools import (
            get_user_accounts_executor, get_user_profile_executor,
            GET_USER_ACCOUNTS_SCHEMA, GET_USER_PROFILE_SCHEMA,
        )
        from ..builtin_agents.location import (
            get_user_location_executor, GET_USER_LOCATION_SCHEMA,
            set_location_reminder_executor, SET_LOCATION_REMINDER_SCHEMA,
        )

        tools: List[AgentTool] = []

        # Google search
        tools.append(AgentTool(
            name="google_search",
            description="Search the web using Google. Returns titles, URLs, and snippets of top results.",
            parameters=GOOGLE_SEARCH_SCHEMA,
            executor=google_search_executor,
            category="web",
        ))

        # Web fetch
        tools.append(AgentTool(
            name="web_fetch",
            description="Fetch a URL and extract its readable content as text. Use this to read articles, documentation, or any web page. Returns the main content with boilerplate removed.",
            parameters=WEB_FETCH_SCHEMA,
            executor=web_fetch_executor,
            category="web",
        ))

        # Important dates (6 tools)
        for td in IMPORTANT_DATES_TOOL_DEFS:
            tools.append(AgentTool(
                name=td["name"],
                description=td["description"],
                parameters=td["parameters"],
                executor=td["executor"],
                category="user",
            ))

        # User tools
        tools.append(AgentTool(
            name="get_user_accounts",
            description="Get the user's connected accounts (email, calendar). Use this when user asks about their connected accounts.",
            parameters=GET_USER_ACCOUNTS_SCHEMA,
            executor=get_user_accounts_executor,
            category="user",
        ))
        tools.append(AgentTool(
            name="get_user_profile",
            description="Get the user's profile information (name, email, phone, timezone). Use this when you need to know about the user.",
            parameters=GET_USER_PROFILE_SCHEMA,
            executor=get_user_profile_executor,
            category="user",
        ))

        # Location tools
        tools.append(AgentTool(
            name="get_user_location",
            description="Get the user's current location (latitude, longitude, and place name). Use when you need the user's precise current location for finding nearby places, calculating distances, or checking proximity.",
            parameters=GET_USER_LOCATION_SCHEMA,
            executor=get_user_location_executor,
            category="location",
        ))
        tools.append(AgentTool(
            name="set_location_reminder",
            description="Set a location-based reminder that notifies the user when they arrive near a specific place. Use when the user says things like 'remind me to X when I'm near Y' or 'notify me when I get to Z'.",
            parameters=SET_LOCATION_REMINDER_SCHEMA,
            executor=set_location_reminder_executor,
            category="location",
        ))

        # Recall memory (only if momex is available)
        if self.momex:
            momex = self.momex

            async def recall_memory_executor(args: dict, context: AgentToolContext = None) -> str:
                query = args.get("query", "")
                if not query:
                    return "Error: query is required"
                tenant_id = context.tenant_id if context else "default"
                results = await momex.search(tenant_id=tenant_id, query=query, limit=5)
                if not results:
                    return "No relevant memories found."
                lines = []
                for r in results:
                    text = r.get("memory", r.get("text", str(r)))
                    lines.append(f"- {text}")
                return "\n".join(lines)

            tools.append(AgentTool(
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
                category="user",
            ))

        return tools

    def _build_notify_user_tool(self, context: Dict[str, Any]):
        """Build a notify_user tool for conditional cron delivery.

        When a cron job has conditional delivery, this tool is injected into
        the ReAct loop. The agent calls it ONLY when the monitored condition
        is met. If the agent never calls it, no notification is sent.

        Returns:
            (AgentTool, schema_dict) tuple
        """
        def make_executor(ctx):
            async def notify_user_executor(args: dict, tool_context=None) -> str:
                message = args.get("message", "")
                if not message:
                    return "Error: message is required."
                ctx["cron_notification"] = message
                return f"Notification queued: {message}"
            return notify_user_executor

        tool = AgentTool(
            name="notify_user",
            description=(
                "Send a notification to the user. ONLY call this when a monitored "
                "condition is actually met (e.g. price dropped below threshold, "
                "weather changed, etc.). If the condition is NOT met, do NOT call "
                "this tool — just respond normally without notifying."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The notification message to send to the user",
                    }
                },
                "required": ["message"],
            },
            executor=make_executor(context),
            category="cron",
        )

        return tool, tool.to_openai_schema()

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

        Conversation history is provided by the app layer via
        metadata["conversation_history"].

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

        # Conversation history is provided by the app layer via metadata
        external_history = meta.get("conversation_history")
        if external_history:
            context["conversation_history"] = external_history

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

        Extracts long-term knowledge via Momex, then runs guardrails
        output check and any registered post_process_hooks.

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
            # Long-term knowledge extraction — fire-and-forget so the
            # response is not blocked by embedding / LLM extraction.
            async def _bg_momex_add():
                try:
                    await self.momex.add(
                        tenant_id=tenant_id,
                        messages=messages,
                        infer=True,
                    )
                except Exception as e:
                    logger.warning(f"Background momex.add failed: {e}")

            asyncio.create_task(_bg_momex_add())

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










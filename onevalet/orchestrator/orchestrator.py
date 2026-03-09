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

import copy
import json
import asyncio
import logging
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
from .react_config import ReactLoopConfig
from ..constants import GENERATE_PLAN_TOOL_NAME
from .context_manager import ContextManager
from .prompts import build_system_prompt
from .audit_logger import AuditLogger
from .tool_policy import ToolPolicyFilter

from .react_loop import ReactLoopMixin
from .tool_manager import ToolManagerMixin
from .llm_manager import LLMManagerMixin

if TYPE_CHECKING:
    from ..checkpoint import CheckpointManager
    from ..llm.router import ModelRouter
    from ..msghub import MessageHub
    from ..protocols import LLMClientProtocol
    from ..memory.momex import MomexMemory

from ..standard_agent import StandardAgent
from ..config import AgentRegistry

logger = logging.getLogger(__name__)


class Orchestrator(ReactLoopMixin, ToolManagerMixin, LLMManagerMixin):
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
        system_prompt_mode: str = "append",
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
            system_prompt: Optional user-defined persona / custom instructions.
                Behavior depends on system_prompt_mode.
            system_prompt_mode: How system_prompt is applied:
                - "append" (default): appended after the built-in system prompt
                - "override": replaces the default preamble ("You are OneValet...")
                  while keeping all functional sections (tool routing, workflow, etc.)
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
        self.system_prompt_mode = system_prompt_mode

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
        self._tenant_plans: Dict[str, Any] = {}

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

        # Start cleanup loop for timed-out WAITING agents
        if self.config.session.enabled:
            await self.agent_pool.start_cleanup_loop()

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
        result = None
        async for event in self._execute_message(tenant_id, message, images, metadata):
            if event.type == EventType.EXECUTION_END:
                result = event.data
        return result

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
        async for event in self._execute_message(tenant_id, message, images, metadata):
            yield event

    # ==========================================================================
    # UNIFIED EXECUTION PIPELINE
    # ==========================================================================

    async def _execute_message(
        self,
        tenant_id: str,
        message: str,
        images: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Unified execution pipeline yielding streaming events.

        Both ``handle_message`` (consumes silently) and ``stream_message``
        (yields to caller) delegate to this single implementation.

        The final event is always EXECUTION_END carrying an ``AgentResult``
        in ``event.data`` so that ``handle_message`` can return it directly.
        """
        if not self._initialized:
            await self.initialize()

        metadata = metadata or {}

        # Step 1: Prepare context
        context = await self.prepare_context(tenant_id, message, metadata)

        # Store images in context so agent tools can access them (e.g. receipt scanning)
        if images:
            context["user_images"] = images

        # Step 2: Check if should process
        if not await self.should_process(message, context):
            result = await self.reject_message(message, context)
            yield AgentEvent(
                type=EventType.MESSAGE_CHUNK,
                data={"chunk": result.raw_message or ""},
            )
            yield AgentEvent(type=EventType.EXECUTION_END, data=result)
            return

        # Step 3: Check pending agents (WAITING_FOR_INPUT / WAITING_FOR_APPROVAL)
        agent_result = await self._check_pending_agents(tenant_id, message, context)
        if agent_result is not None:
            # Agent still waiting or completed -> return result directly.
            # The user's message was a response to the pending agent (e.g. an
            # approval like "yes"/"ok"), NOT a new task.  Feeding it into the ReAct
            # loop would cause the orchestrator to misinterpret the approval
            # word as a brand-new request and spawn unnecessary follow-up agents.
            agent_result = await self.post_process(agent_result, context)
            yield AgentEvent(
                type=EventType.MESSAGE_START,
                data={"agent_type": agent_result.agent_type},
            )
            yield AgentEvent(
                type=EventType.MESSAGE_CHUNK,
                data={"chunk": agent_result.raw_message or ""},
            )
            yield AgentEvent(type=EventType.MESSAGE_END, data={})
            yield AgentEvent(type=EventType.EXECUTION_END, data=agent_result)
            return

        # Step 4: Intent Analysis — classify domains and detect multi-intent
        intent = await self._analyze_intent(message, context)

        # Step 4b: Multi-intent → DAG execution
        if intent.intent_type == "multi" and intent.sub_tasks:
            final_response = ""
            dag_exec_data: Dict[str, Any] = {}
            async for event in self._stream_dag(intent, tenant_id, context, metadata):
                if event.type == EventType.EXECUTION_END:
                    dag_exec_data = event.data
                    final_response = dag_exec_data.get("final_response", "")
                yield event
            # Post-process
            pending_approvals = dag_exec_data.get("pending_approvals", [])
            if pending_approvals:
                status = AgentStatus.WAITING_FOR_APPROVAL
            else:
                status = AgentStatus.COMPLETED
            result = AgentResult(
                agent_type=self.__class__.__name__,
                status=status,
                raw_message=final_response,
            )
            tool_calls = dag_exec_data.get("tool_calls", [])
            context["tool_calls"] = tool_calls
            await self._save_tool_call_history(tenant_id, tool_calls)
            result = await self.post_process(result, context)
            yield AgentEvent(type=EventType.EXECUTION_END, data=result)
            return

        # Step 5: Build domain-filtered tool schemas
        tool_schemas = await self._build_tool_schemas(tenant_id, domains=intent.domains)

        # Step 5b: Inject notify_user tool for conditional cron delivery
        # Use a local copy of builtin_tools to avoid mutating the instance list
        request_tools = list(self.builtin_tools)
        if metadata.get("cron_conditional_delivery"):
            notify_tool, notify_schema = self._build_notify_user_tool(context)
            request_tools.append(notify_tool)
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

        # Step 7: Run ReAct loop
        final_response = ""
        exec_data: Dict[str, Any] = {}
        async for event in self._react_loop_events(
            messages, tool_schemas, tenant_id,
            context=context, user_message=message, media=media,
            metadata=metadata, request_tools=request_tools,
        ):
            if event.type == EventType.EXECUTION_END:
                exec_data = event.data
                final_response = exec_data.get("final_response", "")
            yield event

        # Step 8: Map loop results -> AgentResult
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

        # Expose tool call records to post-process hooks
        tool_calls = exec_data.get("tool_calls", [])
        context["tool_calls"] = tool_calls

        # Persist tool call history
        await self._save_tool_call_history(tenant_id, tool_calls)

        # Step 9: Post-process
        result = await self.post_process(result, context)
        yield AgentEvent(type=EventType.EXECUTION_END, data=result)

    # ==========================================================================
    # REACT LOOP — see react_loop.py (ReactLoopMixin)
    # LLM CALLS  — see llm_manager.py (LLMManagerMixin)
    # TOOLS      — see tool_manager.py (ToolManagerMixin)
    # ==========================================================================

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

        # Build system prompt with optional preamble override
        build_kwargs = dict(
            agent_descriptions=agent_descriptions,
            include_planning=include_planning,
            approved_plan=approved_plan,
            pending_plan=pending_plan,
        )
        if self.system_prompt and self.system_prompt_mode == "override":
            build_kwargs["preamble"] = self.system_prompt

        system_prompt = build_system_prompt(**build_kwargs)

        system_parts = [system_prompt]
        if self.system_prompt and self.system_prompt_mode != "override":
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

        # User profile (extracted from email, passed by app layer)
        profile_text = self._format_user_profile(meta.get("user_profile"))
        if profile_text:
            system_parts.append("\n[User Profile]\n" + profile_text)

        # Relevant memories from Momex (auto-recall based on user message)
        if self.momex:
            try:
                recalled = await self.momex.search(
                    tenant_id=context.get("tenant_id", ""),
                    query=user_message,
                    limit=10,
                )
                if recalled:
                    memory_lines = [f"- {r['text']}" for r in recalled]
                    system_parts.append(
                        "\n[Relevant Memories]\n"
                        + "\n".join(memory_lines)
                    )
            except Exception as e:
                logger.warning(f"Failed to auto-recall memories: {e}")

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

    @staticmethod
    def _format_user_profile(profile: Optional[Dict[str, Any]]) -> str:
        """Format user_profiles JSONB into concise text for system prompt.

        Returns empty string if profile is None or has no useful data.
        """
        if not profile:
            return ""

        lines: List[str] = []

        identity = profile.get("identity") or {}
        if identity.get("full_name"):
            lines.append(f"Name: {identity['full_name']}")
        if identity.get("birthday"):
            lines.append(f"Birthday: {identity['birthday']}")

        for addr in (profile.get("addresses") or []):
            parts = [addr.get("street"), addr.get("city"), addr.get("state")]
            loc = ", ".join(p for p in parts if p)
            if loc:
                lines.append(f"{addr.get('label', 'Address').title()}: {loc}")

        work = profile.get("work") or {}
        for job in (work.get("jobs") or []):
            if job.get("is_current"):
                parts = [job.get("title", ""), job.get("employer", "")]
                desc = " at ".join(p for p in parts if p)
                if desc:
                    lines.append(f"Work: {desc}")

        education = profile.get("education") or {}
        for school in (education.get("schools") or []):
            parts = [school.get("degree"), school.get("major")]
            desc = " in ".join(p for p in parts if p)
            name = school.get("name", "")
            if name:
                line = f"Education: {name}"
                if desc:
                    line += f" ({desc})"
                lines.append(line)

        relationships = profile.get("relationships") or {}
        for person in (relationships.get("family") or []):
            line = f"{person.get('relationship', 'Family')}: {person.get('name', '')}"
            if person.get("birthday"):
                line += f" (birthday: {person['birthday']})"
            lines.append(line)
        so = relationships.get("significant_other")
        if so and so.get("name"):
            line = f"Partner: {so['name']}"
            if so.get("birthday"):
                line += f" (birthday: {so['birthday']})"
            lines.append(line)

        lifestyle = profile.get("lifestyle") or {}
        for pet in (lifestyle.get("pets") or []):
            if pet.get("name"):
                lines.append(f"Pet: {pet['name']} ({pet.get('type', '')})")
        for vehicle in (lifestyle.get("vehicles") or []):
            if vehicle.get("is_current") and vehicle.get("make"):
                parts = [str(vehicle.get("year", "")), vehicle.get("make", ""), vehicle.get("model", "")]
                lines.append(f"Vehicle: {' '.join(p for p in parts if p)}")

        travel = profile.get("travel") or {}
        for prog in (travel.get("loyalty_programs") or []):
            name = prog.get("program", "")
            if name:
                line = f"Loyalty: {name}"
                if prog.get("status"):
                    line += f" ({prog['status']})"
                lines.append(line)

        return "\n".join(lines)

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

        Delegates to ``_stream_dag`` (the single implementation) and
        silently consumes its events, mirroring the pattern used by
        ``handle_message`` with ``_react_loop_events``.
        """
        exec_data: Dict[str, Any] = {}
        async for event in self._stream_dag(intent, tenant_id, context, metadata):
            if event.type == EventType.EXECUTION_END:
                exec_data = event.data

        final_response = exec_data.get("final_response", "")
        pending_approvals = exec_data.get("pending_approvals", [])

        status = AgentStatus.COMPLETED
        if pending_approvals:
            status = AgentStatus.WAITING_FOR_APPROVAL

        return AgentResult(
            agent_type=self.__class__.__name__,
            status=status,
            raw_message=final_response,
            metadata={
                "dag_execution": True,
                "sub_tasks": len(intent.sub_tasks),
                "levels": exec_data.get("levels", 0),
                "duration_ms": exec_data.get("duration_ms", 0),
                "token_usage": exec_data.get("token_usage", {}),
                "pending_approvals": pending_approvals,
            },
        )

    async def _stream_dag(
        self,
        intent: "IntentAnalysis",
        tenant_id: str,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream events during DAG execution.

        This is the single DAG implementation.  ``_execute_dag`` consumes
        this generator silently, so all fixes live here only.

        Fixes applied:
        - Skip sub-tasks whose dependencies failed (get_runnable_tasks)
        - Propagate pending_approvals from sub-task ReAct loops
        - Aggregate token usage, turns, and tool_calls across sub-tasks
        - DAG-level timeout guard
        - Collect and yield events from parallel sub-tasks
        """
        from .dag_executor import (
            topological_sort,
            SubTaskResult,
            get_runnable_tasks,
            aggregate_token_usage,
        )

        start_time = time.monotonic()
        levels = topological_sort(intent.sub_tasks)
        all_results: Dict[int, SubTaskResult] = {}
        all_pending_approvals: list = []
        total_turns = 0
        total_tool_calls = 0

        # Fix 5: DAG-level timeout (generous but bounded)
        dag_timeout_seconds = (
            self._react_config.max_turns
            * self._react_config.agent_tool_execution_timeout
        )
        deadline = start_time + dag_timeout_seconds

        yield AgentEvent(
            type=EventType.WORKFLOW_START,
            data={"sub_tasks": len(intent.sub_tasks), "levels": len(levels)},
        )

        for level_idx, level in enumerate(levels):
            # Fix 5: check DAG-level timeout before each level
            if time.monotonic() > deadline:
                logger.warning(
                    f"[DAG] Timeout exceeded ({dag_timeout_seconds}s) at level {level_idx}"
                )
                for st in level:
                    all_results[st.id] = SubTaskResult(
                        sub_task_id=st.id,
                        description=st.description,
                        response="Skipped: DAG timeout exceeded",
                        status="skipped",
                    )
                # Also mark tasks in remaining levels
                for remaining_level in levels[level_idx + 1:]:
                    for st in remaining_level:
                        all_results[st.id] = SubTaskResult(
                            sub_task_id=st.id,
                            description=st.description,
                            response="Skipped: DAG timeout exceeded",
                            status="skipped",
                        )
                break

            # Fix 1: split level into runnable / skipped tasks
            runnable, skipped = get_runnable_tasks(level, all_results)
            for st in skipped:
                all_results[st.id] = SubTaskResult(
                    sub_task_id=st.id,
                    description=st.description,
                    response="Skipped: dependency failed",
                    status="skipped",
                )
                yield AgentEvent(
                    type=EventType.STAGE_START,
                    data={
                        "sub_task_id": st.id,
                        "description": st.description,
                        "domain": st.domain,
                    },
                )
                yield AgentEvent(
                    type=EventType.STAGE_END,
                    data={"sub_task_id": st.id, "status": "skipped"},
                )

            if not runnable:
                continue

            for st in runnable:
                yield AgentEvent(
                    type=EventType.STAGE_START,
                    data={
                        "sub_task_id": st.id,
                        "description": st.description,
                        "domain": st.domain,
                    },
                )

            if len(runnable) == 1:
                # Single task in level: stream events in real-time
                st = runnable[0]
                augmented_message = self._build_dag_augmented_message(
                    st, all_results,
                )
                tool_schemas = await self._build_tool_schemas(
                    tenant_id, domains=[st.domain],
                )
                task_context = copy.deepcopy(context)
                messages = await self._build_llm_messages(task_context, augmented_message)

                exec_data: Dict[str, Any] = {}
                async for event in self._react_loop_events(
                    messages, tool_schemas, tenant_id,
                    context=task_context, user_message=augmented_message,
                ):
                    if event.type == EventType.EXECUTION_END:
                        exec_data = event.data
                    yield event

                # Fix 2: collect pending_approvals from sub-task
                sub_approvals = exec_data.get("pending_approvals", [])
                if sub_approvals:
                    all_pending_approvals.extend(sub_approvals)

                # Fix 3: accumulate turns / tool_calls
                total_turns += exec_data.get("turns", 0)
                total_tool_calls += exec_data.get("tool_calls_count", 0)

                all_results[st.id] = SubTaskResult(
                    sub_task_id=st.id,
                    description=st.description,
                    response=exec_data.get("final_response", ""),
                    status="completed",
                    duration_ms=exec_data.get("duration_ms", 0),
                    token_usage=exec_data.get("token_usage", {}),
                )
                yield AgentEvent(
                    type=EventType.STAGE_END,
                    data={"sub_task_id": st.id},
                )
            else:
                # Fix 6: Multiple parallel tasks — collect events in memory
                # during parallel execution, then yield them after.
                async def _run_collecting(sub_task):
                    aug_msg = self._build_dag_augmented_message(
                        sub_task, all_results,
                    )
                    t_schemas = await self._build_tool_schemas(
                        tenant_id, domains=[sub_task.domain],
                    )
                    task_context = copy.deepcopy(context)
                    msgs = await self._build_llm_messages(task_context, aug_msg)
                    exec_d: Dict[str, Any] = {}
                    events: list = []
                    async for ev in self._react_loop_events(
                        msgs, t_schemas, tenant_id,
                        context=task_context, user_message=aug_msg,
                    ):
                        if ev.type == EventType.EXECUTION_END:
                            exec_d = ev.data
                        else:
                            events.append(ev)
                    sub_result = SubTaskResult(
                        sub_task_id=sub_task.id,
                        description=sub_task.description,
                        response=exec_d.get("final_response", ""),
                        status="completed",
                        duration_ms=exec_d.get("duration_ms", 0),
                        token_usage=exec_d.get("token_usage", {}),
                    )
                    return sub_result, events, exec_d

                level_results = await asyncio.gather(
                    *[_run_collecting(st) for st in runnable],
                    return_exceptions=True,
                )

                for st, result in zip(runnable, level_results):
                    if isinstance(result, BaseException):
                        logger.warning(f"[DAG] Sub-task {st.id} failed: {result}")
                        all_results[st.id] = SubTaskResult(
                            sub_task_id=st.id,
                            description=st.description,
                            response=f"Error: {result}",
                            status="error",
                        )
                    else:
                        sub_result, events, exec_d = result
                        # Fix 6: yield collected events
                        for ev in events:
                            yield ev
                        # Fix 2: collect pending_approvals
                        sub_approvals = exec_d.get("pending_approvals", [])
                        if sub_approvals:
                            all_pending_approvals.extend(sub_approvals)
                        # Fix 3: accumulate turns / tool_calls
                        total_turns += exec_d.get("turns", 0)
                        total_tool_calls += exec_d.get("tool_calls_count", 0)
                        all_results[st.id] = sub_result
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

        # Fix 3: aggregate token usage across all sub-tasks
        aggregated_usage = aggregate_token_usage(all_results)

        duration_ms = int((time.monotonic() - start_time) * 1000)
        yield AgentEvent(
            type=EventType.EXECUTION_END,
            data={
                "final_response": final_response,
                "dag_execution": True,
                "sub_tasks": len(intent.sub_tasks),
                "levels": len(levels),
                "pending_approvals": all_pending_approvals,
                "result_status": None,
                "turns": total_turns,
                "token_usage": aggregated_usage,
                "duration_ms": duration_ms,
                "tool_calls_count": total_tool_calls,
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

            # Add to pool
            await self.agent_pool.add_agent(agent)

            logger.debug(f"Created agent {agent.agent_id} of type {agent_type}")
            return agent

        except Exception as e:
            logger.error(f"Failed to create agent {agent_type}: {e}", exc_info=True)
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










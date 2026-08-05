"""
Koa Orchestrator - Central coordinator using ReAct loop

This module provides an extensible Orchestrator using the Template Method pattern
combined with a ReAct (Reasoning + Acting) loop for tool/agent execution.

Extension Points (override in subclass):
    - prepare_context(): Add memories, user info, custom metadata
    - should_process(): Guardrails, rate limits, custom access control
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

import asyncio
import copy
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional

from ..memory.governance import MemoryGovernance
from ..memory.session_memory import SessionMemoryManager
from ..memory.true_memory import extract_true_memory_proposals
from ..message import Message
from ..result import AgentResult, AgentStatus
from ..streaming.models import AgentEvent, EventType, StreamMode
from .agent_lifecycle import AgentLifecycleMixin
from .ask_mirror import AskMirror
from .audit_logger import AuditLogger
from .context_manager import ContextManager
from .dag_loop import DagLoopMixin
from .execution_policy import ExecutionPolicyEngine
from .inbox import InboxStore
from .llm_manager import LLMManagerMixin
from .models import (
    CALLBACK_HANDLER_ATTR,
    AgentCallback,
    OrchestratorConfig,
)
from .pool import AgentPoolManager
from .prompt_builder import PromptBuilderMixin
from .react_config import ReactLoopConfig
from .react_loop import ReactLoopMixin
from .request_analysis import RequestAnalysisMixin
from .resume import ResumeMixin
from .run_control import RunControlRegistry
from .tool_manager import ToolManagerMixin
from .tool_policy import ToolPolicyFilter
from .transcript_store import TranscriptStore

if TYPE_CHECKING:
    from ..llm.router import ModelRouter
    from ..memory.momex import MomexMemory
    from ..msghub import MessageHub
    from ..protocols import LLMClientProtocol

from ..config import AgentRegistry

logger = logging.getLogger(__name__)

#: Database maintenance is deliberately coarse. Recovery is also triggered by
#: every answer and by a run handing itself on; this loop exists for process
#: death, not as the normal scheduler.
_INBOX_MAINTENANCE_INTERVAL = 300
_TRANSCRIPT_RETENTION_HOURS = 48
_ASK_RETENTION_DAYS = 30
_MAX_AUTOMATIC_RECOVERIES = 5
_RECOVERY_BACKOFF_BASE_SECONDS = 300


from .state_persistence import PlanStore  # noqa: E402
from .tool_pipeline import ToolPipeline, credential_check_hook, result_audit_hook  # noqa: E402


def _cancel_routing(task: "Optional[asyncio.Task]") -> None:
    """Drop a pre-started routing classification whose result is now unused.

    Cancelling matters because the task holds an in-flight model call; leaving
    it to finish would keep a connection open and log a decision nothing acts on.
    """
    if task is not None and not task.done():
        task.cancel()


def _merge_true_memory_proposals(
    existing: Optional[List[Dict[str, Any]]],
    new: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    def confidence(item: Dict[str, Any]) -> float:
        try:
            return float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    merged: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for proposal in list(existing or []) + list(new or []):
        if not isinstance(proposal, dict):
            continue
        key = (
            str(proposal.get("operation") or "upsert"),
            str(proposal.get("namespace") or ""),
            str(proposal.get("fact_key") or ""),
        )
        current = merged.get(key)
        if current is None or confidence(proposal) >= confidence(current):
            merged[key] = proposal
    return list(merged.values())


class Orchestrator(
    ReactLoopMixin,
    DagLoopMixin,
    AgentLifecycleMixin,
    RequestAnalysisMixin,
    PromptBuilderMixin,
    ResumeMixin,
    ToolManagerMixin,
    LLMManagerMixin,
):
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
            if hasattr(base, "_callback_handler_map"):
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
        message_hub: Optional["MessageHub"] = None,
        guardrails_checker: Optional[Any] = None,
        rate_limiter: Optional[Callable] = None,
        post_process_hooks: Optional[List[Callable]] = None,
        tool_policy_filter: Optional[ToolPolicyFilter] = None,
        model_router: Optional["ModelRouter"] = None,
        memory_governance: Optional[MemoryGovernance] = None,
        session_memory: Optional[SessionMemoryManager] = None,
        execution_policy: Optional[ExecutionPolicyEngine] = None,
        tenant_gate: Optional[Any] = None,
        idempotency_store: Optional[Any] = None,
        task_registry: Optional[Any] = None,
        intent_feedback_store: Optional[Any] = None,
        intent_embedding_router: Optional[Any] = None,
        enable_clarification: bool = True,
    ):
        """
        Initialize Orchestrator.

        Args:
            momex: Momex memory — conversation history + long-term knowledge
            config: Full orchestrator configuration
            llm_client: LLM client for the ReAct loop
            agent_registry: Pre-configured agent registry
            system_prompt: Optional user-defined persona / custom instructions.
                Behavior depends on system_prompt_mode.
            system_prompt_mode: How system_prompt is applied:
                - "append" (default): appended after the built-in system prompt
                - "override": replaces the default preamble ("You are Koa...")
                  while keeping all functional sections (tool routing, workflow, etc.)
            react_config: ReAct loop configuration (max_turns, timeouts, etc.)
            credential_store: CredentialStore for tool execution context
            trigger_engine: TriggerEngine for proactive trigger tasks
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
        self.memory_governance = memory_governance or MemoryGovernance()
        self.session_memory = session_memory or SessionMemoryManager()
        self._execution_policy = execution_policy or ExecutionPolicyEngine()

        # Production-readiness hooks (P0/P1):
        #   - tenant_gate: per-tenant rate/concurrency/budget admission control
        #   - idempotency_store: tool-call dedup for retry-safe side effects
        #   - task_registry: tracks fire-and-forget asyncio.Tasks for graceful
        #     shutdown and crash-visibility.  When unset, a local registry is
        #     created and cancelled by ``shutdown()``.
        self.tenant_gate = tenant_gate
        self.idempotency_store = idempotency_store
        from ..observability import TaskRegistry as _TR

        self.task_registry = task_registry or _TR(self.__class__.__name__)

        # Interruption + mid-run steering for in-flight requests. Surfaces call
        # request_interrupt()/queue_steering() with a tenant_id to signal the
        # run the ReAct loop is currently executing for that tenant.
        self._run_controls = RunControlRegistry()

        # Durable transcripts so a run can outlive the process that started it.
        self._transcript_store = TranscriptStore(database)

        # The human-attention queue: what the assistant is waiting on a person
        # for, and where their answer comes back in.
        self.inbox = InboxStore(database)
        # Channels an ask is delivered on. Populated by the app layer; without
        # any, asks still queue and surface in the app.
        self.ask_mirror = AskMirror()

        # Intent-recognition infrastructure.  See
        # :mod:`koa.orchestrator.intent_feedback` and
        # :mod:`koa.orchestrator.intent_embedding` for the full design.
        # Defaults to an in-memory feedback store so operators get
        # accuracy metrics out of the box; production should inject a
        # durable implementation.
        if intent_feedback_store is None:
            from .intent_feedback import InMemoryIntentFeedbackStore

            intent_feedback_store = InMemoryIntentFeedbackStore()
        self.intent_feedback_store = intent_feedback_store
        self.intent_embedding_router = intent_embedding_router
        self.enable_clarification = bool(enable_clarification)

        # Audit logging
        self._audit = AuditLogger()

        # Tool execution pipeline with before/after hooks
        self._tool_pipeline = ToolPipeline(idempotency_store=self.idempotency_store)
        self._tool_pipeline.add_before_hook(credential_check_hook)
        self._tool_pipeline.add_after_hook(result_audit_hook)

        # State
        self._initialized = False
        self._plan_store = PlanStore(database=database)
        self._tenant_plans: Dict[str, Any] = {}  # in-memory fallback for legacy code

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

        if getattr(self.task_registry, "closed", False):
            raise RuntimeError("Cannot reinitialize an Orchestrator after shutdown")

        # Initialize agent registry if not provided
        if not self._registry_initialized and self._agent_registry is None:
            logger.warning("No agent registry provided. Agent-Tools will not be available.")

        # Validate LLM client is available
        if not self.llm_client:
            raise RuntimeError("LLM client is required. Pass llm_client to Orchestrator().")

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
        if self._transcript_store.enabled and self.inbox.enabled:
            maintenance = self._inbox_maintenance_loop()
            try:
                self.task_registry.create_task(
                    maintenance,
                    name="inbox_maintenance",
                )
            except RuntimeError:
                maintenance.close()
                self._initialized = False
                raise
        logger.info("Orchestrator initialized")

    async def _inbox_maintenance_loop(self) -> None:
        """Recover work left by dead processes and bound durable history.

        Runs immediately at startup, then every few minutes. Claims and
        execution claims remain the arbiters, so every app instance may run
        the same maintenance loop without doing an action twice.
        """
        while True:
            try:
                await self._maintain_inbox()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Inbox maintenance failed: {e}", exc_info=True)
            await asyncio.sleep(_INBOX_MAINTENANCE_INTERVAL)

    async def _maintain_inbox(self) -> None:
        transcript_count = await self._transcript_store.prune(
            _TRANSCRIPT_RETENTION_HOURS
        )
        ask_count = await self.inbox.prune(_ASK_RETENTION_DAYS)
        if transcript_count or ask_count:
            logger.info(
                f"[Inbox] Pruned {transcript_count} transcript(s) and "
                f"{ask_count} ask(s)"
            )

        run_ids = await self.inbox.recoverable_runs(
            self._resume_lease_seconds(),
            _MAX_AUTOMATIC_RECOVERIES,
        )
        for run_id in run_ids:
            reserved = await self._transcript_store.reserve_recovery(
                run_id,
                self._resume_lease_seconds(),
                _MAX_AUTOMATIC_RECOVERIES,
                _RECOVERY_BACKOFF_BASE_SECONDS,
            )
            if reserved:
                self._schedule_maintenance_resume(run_id)

    def _schedule_maintenance_resume(self, run_id: str) -> None:
        active = getattr(self, "_maintenance_resumes", None)
        if active is None:
            active = set()
            self._maintenance_resumes = active
        if run_id in active:
            return

        async def _recover() -> None:
            try:
                await self.resume_when_free(run_id)
            finally:
                active.discard(run_id)

        recovery = _recover()
        active.add(run_id)
        try:
            self.task_registry.create_task(recovery, name=f"recover:{run_id}")
        except RuntimeError as e:
            active.discard(run_id)
            recovery.close()
            logger.warning(f"Could not schedule recovery of run {run_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown the orchestrator gracefully.

        Cancels tracked background tasks before tearing down the agent pool
        and registry so no in-flight coroutines are GC'd with unhandled
        exceptions.
        """
        # Cancel fire-and-forget tasks first (speculative search, memory
        # extraction, etc.) so their shutdown doesn't race the pool close.
        try:
            await self.task_registry.cancel_all(timeout=5.0)
        except Exception as exc:
            logger.warning("task_registry.cancel_all failed: %s", exc)
        if self.trigger_engine:
            await self.trigger_engine.stop()
        await self.agent_pool.close()
        if self._agent_registry:
            await self._agent_registry.shutdown()
        self._initialized = False
        logger.info("Orchestrator shutdown")

    def _resolve_model_fallback(self, loop_error: Any) -> Optional[Any]:
        """Resolve a fallback LLM client for model-level retry.

        Returns an LLM client different from the one that failed, or
        None if no fallback is available.
        """
        from .error_classifier import LLMErrorKind

        # Don't retry auth errors at model level
        if loop_error.error_kind == LLMErrorKind.AUTH:
            return None

        fallback_providers = self._react_config.fallback_providers
        if not fallback_providers:
            return None

        registry = self._get_llm_registry()
        if registry is None:
            return None

        for provider_name in fallback_providers:
            client = registry.get(provider_name)
            if client is not None and client is not self.llm_client:
                logger.info(f"[Orchestrator] Model-level fallback resolved: {provider_name}")
                return client

        return None

    # ==========================================================================
    # RUN CONTROL (interrupt / steering)
    # ==========================================================================

    def request_interrupt(self, tenant_id: str, reason: str = "user stop") -> bool:
        """Stop the tenant's in-flight run at its next boundary.

        Returns False when the tenant has no run in flight. The run unwinds
        cooperatively: pending tool calls are compensated with error results so
        the transcript never carries an orphaned tool_call, and the loop emits
        an INTERRUPTED event before its EXECUTION_END.
        """
        return self._run_controls.request_interrupt(tenant_id, reason)

    def queue_steering(self, tenant_id: str, text: str) -> bool:
        """Inject a user message into the tenant's in-flight run.

        The message is appended at the next turn boundary, letting the user
        redirect a long task without restarting it. Returns False when the
        tenant has no run in flight.
        """
        return self._run_controls.queue_steering(tenant_id, text)

    def is_running(self, tenant_id: str) -> bool:
        return self._run_controls.get(tenant_id) is not None

    # ==========================================================================
    # MAIN ENTRY POINT
    # ==========================================================================

    async def handle_message(
        self,
        tenant_id: str,
        message: str,
        images: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
        metadata: Optional[Dict[str, Any]] = None,
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

        Admission control:
            The optional :class:`~koa.tenant_gate.TenantGate` (injected via
            ``tenant_gate=`` in ``__init__``) enforces per-tenant rate limit,
            concurrency, and budget.  When the gate rejects, a single ERROR
            event is emitted followed by EXECUTION_END carrying a FAILED
            ``AgentResult`` — no LLM calls are made and no tools run.
        """
        from ..observability import bind_request_context, new_request_id, trace_span

        metadata = metadata or {}
        caller_idempotency_key = (metadata or {}).get("idempotency_key") or (metadata or {}).get(
            "X-Idempotency-Key"
        )
        # Bind request/tenant ContextVars so concurrent requests on the
        # same orchestrator instance don't corrupt each other's tracing.
        rid = new_request_id()
        with bind_request_context(
            request_id=rid,
            tenant_id=tenant_id,
            idempotency_key=caller_idempotency_key,
        ):
            with trace_span(
                "orchestrator.execute_message",
                tenant_id=tenant_id,
                has_images=bool(images),
            ):
                # Admission control — fail-closed if gate is configured.
                gate = getattr(self, "tenant_gate", None)
                if gate is not None:
                    from ..tenant_gate import GateRejected

                    try:
                        async with gate.acquire(tenant_id) as ticket:
                            async for ev in self._execute_message_inner(
                                tenant_id, message, images, metadata, rid, ticket
                            ):
                                yield ev
                        return
                    except GateRejected as gr:
                        logger.warning(
                            "[Orchestrator] Gate rejected tenant=%s reason=%s",
                            tenant_id,
                            gr.reason,
                        )
                        yield AgentEvent(
                            type=EventType.ERROR,
                            data={
                                "code": gr.reason,
                                "error": str(gr),
                                "retry_after": gr.retry_after,
                            },
                        )
                        rejected = AgentResult(
                            agent_type=self.__class__.__name__,
                            status=AgentStatus.FAILED,
                            raw_message=f"Request rejected: {gr.reason}. "
                            f"Please retry in {gr.retry_after:.0f}s.",
                            metadata={"gate_rejected": True, "reason": gr.reason},
                        )
                        yield AgentEvent(type=EventType.EXECUTION_END, data=rejected)
                        return
                # No gate configured — run inner pipeline directly.
                async for ev in self._execute_message_inner(
                    tenant_id, message, images, metadata, rid, None
                ):
                    yield ev

    async def _execute_message_inner(
        self,
        tenant_id: str,
        message: str,
        images: Optional[List[Dict[str, Any]]],
        metadata: Dict[str, Any],
        request_id: str,
        ticket: Optional[Any],
    ) -> AsyncIterator[AgentEvent]:
        """Inner pipeline body; split out so the gate context manager wraps only this.

        ``ticket`` is a :class:`koa.tenant_gate.GateTicket` or ``None`` — when
        present, token/cost usage is recorded back to it so per-tenant budgets
        are enforced.
        """
        from .graceful_response import generate_graceful_error

        routing_task: Optional[asyncio.Task] = None
        # Bound before the try so the handoff in the finally has something to
        # read even when the failure came before the context was built.
        context: Dict[str, Any] = {}
        try:
            if not self._initialized:
                await self.initialize()

            metadata = metadata or {}

            # ── Request tracing ──
            # start_request inherits the bound request_id ContextVar, so the
            # same id is used for the audit log.
            request_id = self._audit.start_request(
                tenant_id=tenant_id,
                message=message,
                request_id=request_id,
            )

            # Step 0: Clean up stale/completed agents to prevent cross-request state leakage
            await self._cleanup_stale_agents(tenant_id)

            # Step 1: Prepare context
            context = await self.prepare_context(tenant_id, message, metadata)
            context["request_id"] = request_id

            # Store images in context so agent tools can access them (e.g. receipt scanning)
            if images:
                context["user_images"] = images

            # Step 2: Check if should process
            if not await self.should_process(message, context):
                result = await self.reject_message(message, context)
                async for event in self._emit_direct_result(result, with_message_frame=False):
                    yield event
                return

            # Step 3: Check pending agents (WAITING_FOR_INPUT / WAITING_FOR_APPROVAL)
            agent_result = await self._check_pending_agents(tenant_id, message, context)
            if agent_result is not None:
                # The user's message was a reply to the pending agent (e.g. an
                # approval like "yes"/"ok"), NOT a new task. Feeding it into the
                # ReAct loop would read the approval word as a fresh request and
                # spawn unnecessary follow-up agents.
                agent_result = await self.post_process(agent_result, context)
                async for event in self._emit_direct_result(agent_result):
                    yield event
                return

            # Step 3a: The user may be answering something we asked them
            # earlier, on a surface that had no buttons. "yes" is an answer to
            # that question, not a new task.
            answered_run = await self.answer_from_message(tenant_id, message)
            if answered_run:
                async for event in self._continue_answered_run(answered_run, context):
                    yield event
                return

            # Step 3b: Speculative execution — kick off likely tools before LLM decides
            # For image requests, the LLM almost always calls google_search. Starting
            # it now lets us reuse the result later, saving 1-3 seconds of latency.
            speculative_tasks: Dict[str, asyncio.Task] = {}
            if images and message.strip():
                speculative_tasks = self._start_speculative_tasks(
                    message,
                    tenant_id,
                    metadata,
                )
                if speculative_tasks:
                    context["_speculative_tasks"] = speculative_tasks

            # Step 3c: Start model-complexity routing now so it overlaps with
            # intent analysis below -- both are classifiers over the same
            # message, and running them in sequence costs an extra round-trip.
            routing_task = self._start_routing(message, context)

            # Step 4: Intent Analysis — classify domains and detect multi-intent
            intent = await self._analyze_intent(message, context)
            context["intent_analysis"] = intent
            self._audit.log_phase(
                "intent_analysis",
                {
                    "intent_type": intent.intent_type,
                    "domains": intent.domains,
                    "sub_tasks": len(intent.sub_tasks) if intent.sub_tasks else 0,
                    "confidence": intent.confidence,
                    "needs_clarification": intent.needs_clarification,
                    "source": intent.source,
                },
            )

            # Step 4a: Ambiguous → ask the user to clarify instead of guessing.
            # Guarded by an orchestrator-level flag so operators who don't
            # want this UX (e.g. fully-automated flows) can opt out.
            if (
                getattr(self, "enable_clarification", True)
                and intent.needs_clarification
                and intent.source != "fallback"
            ):
                async for event in self._ask_for_clarification(
                    intent, tenant_id, message, context
                ):
                    yield event
                _cancel_routing(routing_task)
                return

            # Step 4b: Multi-intent → DAG execution
            if intent.intent_type == "multi" and intent.sub_tasks:
                # Sub-tasks route themselves, so this run's decision is unused.
                _cancel_routing(routing_task)
                async for event in self._run_multi_intent(
                    intent, tenant_id, message, context, metadata
                ):
                    yield event
                return

            # Step 5 & 6: Build tool schemas and LLM messages in parallel
            tool_schemas_task = self._build_tool_schemas_with_domain_fallback(
                tenant_id, domains=intent.domains
            )
            messages_task = self._build_llm_messages(
                context, message, needs_memory=intent.needs_memory
            )
            tool_schemas, messages = await asyncio.gather(tool_schemas_task, messages_task)

            # Step 5b: Inject notify_user tool for conditional cron delivery
            # Use a local copy of builtin_tools to avoid mutating the instance list
            request_tools = list(self.builtin_tools)
            if metadata.get("cron_conditional_delivery"):
                notify_tool, notify_schema = self._build_notify_user_tool(context)
                request_tools.append(notify_tool)
                tool_schemas.append(notify_schema)

            logger.info(f"[Tools] {len(tool_schemas)} tools available for ReAct")
            self._audit.log_phase(
                "tool_loading",
                {
                    "tool_count": len(tool_schemas),
                    "domains": intent.domains,
                },
            )

            # Convert images to media format for LLM
            media = None
            if images:
                media = [
                    {
                        "type": "image",
                        "data": img["data"],
                        "media_type": img.get("media_type", "image/jpeg"),
                    }
                    for img in images
                ]

            # Step 7: Run ReAct loop with model-level fallback
            final_response = ""
            exec_data: Dict[str, Any] = {}

            async for event in self._run_react_with_fallback(
                messages=messages,
                tool_schemas=tool_schemas,
                tenant_id=tenant_id,
                context=context,
                user_message=message,
                media=media,
                metadata=metadata,
                request_tools=request_tools,
                needs_memory=intent.needs_memory,
                routing_task=routing_task,
            ):
                if event.type == EventType.EXECUTION_END:
                    exec_data = event.data
                    final_response = exec_data.get("final_response", "")
                yield event

            # Step 8: Map loop results -> AgentResult
            result = self._build_result_from_exec_data(
                exec_data,
                final_response=final_response,
                context=context,
                total_tool_count=len(tool_schemas),
            )

            # Expose tool call records to post-process hooks
            tool_calls = exec_data.get("tool_calls", [])
            context["tool_calls"] = tool_calls

            # Persist tool call history
            await self._save_tool_call_history(tenant_id, tool_calls)

            # Step 9: Post-process
            self._audit.log_phase(
                "post_process",
                {"has_proposals": bool(result.metadata.get("true_memory_proposals"))},
            )
            result = await self.post_process(result, context)
            self._audit.end_request(
                status=result.status.value
                if hasattr(result.status, "value")
                else str(result.status),
                token_usage=result.metadata.get("token_usage"),
            )
            yield AgentEvent(type=EventType.EXECUTION_END, data=result)
        except Exception as e:
            _cancel_routing(routing_task)
            await self._fail_transcript(context)
            logger.error(f"[Orchestrator] Unhandled error in _execute_message: {e}", exc_info=True)
            fallback_msg = await generate_graceful_error(
                error=e,
                llm_client=getattr(self, "llm_client", None),
            )
            yield AgentEvent(
                type=EventType.ERROR,
                data={
                    "code": "internal_error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            yield AgentEvent(
                type=EventType.MESSAGE_CHUNK,
                data={"chunk": fallback_msg},
            )
            yield AgentEvent(type=EventType.MESSAGE_END, data={})
            yield AgentEvent(
                type=EventType.EXECUTION_END,
                data=AgentResult(
                    agent_type=self.__class__.__name__,
                    status=AgentStatus.ERROR,
                    raw_message=fallback_msg,
                ),
            )
        except BaseException:
            await self._fail_transcript(context)
            raise
        finally:
            # The run is done either way, so a continuation can safely claim
            # it if the user answered something while it was still working.
            # A request that failed does not cancel a decision they made.
            self.hand_off_unfinished(context)

    def _start_routing(
        self,
        message: str,
        context: Dict[str, Any],
    ) -> Optional["asyncio.Task"]:
        """Begin model-complexity classification without waiting for it.

        Routing and intent analysis are both classifiers over the same user
        message, and neither reads the other's output -- but run in sequence
        they cost two full round-trips before any real work starts.

        The classifier only reads recent user/assistant turns, which are
        already known here, so it does not need the assembled prompt that
        _build_llm_messages produces. Starting it now lets it overlap with
        intent analysis; _react_loop_events awaits the task instead of issuing
        its own call.
        """
        if not self._model_router:
            return None

        history = list(context.get("conversation_history") or [])
        classifier_input = [
            m for m in history if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        ]
        classifier_input.append({"role": "user", "content": message})

        async def _route():
            try:
                return await self._model_router.route(classifier_input)
            except Exception as e:
                logger.warning(f"[Orchestrator] Model routing failed: {e}")
                return None

        return asyncio.ensure_future(_route())

    async def _continue_answered_run(
        self,
        run_id: str,
        context: Dict[str, Any],
    ) -> AsyncIterator[AgentEvent]:
        """Stream the run the user just unblocked.

        A resume can legitimately produce nothing -- the run may still be
        waiting on another ask, or its transcript may have been pruned. The
        user still answered a question, so say so rather than replying with
        silence.
        """
        produced = False
        async for event in self.resume_run(run_id):
            produced = True
            yield event
        if produced:
            return

        # The run it belongs to is still finishing, so it holds its own claim.
        # Hand off to a background waiter rather than dropping the answer, and
        # tell the user it is recorded -- which it is.
        logger.info(f"[Orchestrator] Run {run_id} is busy; continuing it in the background")
        self.task_registry.create_task(
            self.resume_when_free(run_id), name=f"resume:{run_id}"
        )
        result = AgentResult(
            agent_type="Orchestrator",
            status=AgentStatus.COMPLETED,
            raw_message="Got it — I've recorded your answer and I'm on it.",
        )
        async for event in self._emit_direct_result(await self.post_process(result, context)):
            yield event

    async def _emit_direct_result(
        self,
        result: AgentResult,
        *,
        with_message_frame: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        """Stream a result that bypassed the ReAct loop.

        ``with_message_frame`` wraps the text in MESSAGE_START/MESSAGE_END so a
        client renders it as an assistant turn. Rejections skip the frame, since
        they are a refusal to answer rather than an answer.
        """
        if with_message_frame:
            yield AgentEvent(
                type=EventType.MESSAGE_START,
                data={"agent_type": result.agent_type},
            )
        yield AgentEvent(
            type=EventType.MESSAGE_CHUNK,
            data={"chunk": result.raw_message or ""},
        )
        if with_message_frame:
            yield AgentEvent(type=EventType.MESSAGE_END, data={})
        yield AgentEvent(type=EventType.EXECUTION_END, data=result)

    async def _ask_for_clarification(
        self,
        intent: Any,
        tenant_id: str,
        message: str,
        context: Dict[str, Any],
    ) -> AsyncIterator[AgentEvent]:
        """Short-circuit an ambiguous request by asking the user what they meant.

        The classification is recorded as a 'clarify' outcome so the feedback
        store can learn from whatever the user says next.
        """
        clarify_q = (
            intent.clarification_question
            or "Could you share a bit more detail so I can help correctly?"
        )
        logger.info(
            "[IntentAnalyzer] needs_clarification=true "
            "confidence=%.2f; short-circuiting to clarify path",
            intent.confidence,
        )
        result = AgentResult(
            agent_type=self.__class__.__name__,
            status=AgentStatus.COMPLETED,
            raw_message=clarify_q,
            metadata={
                "clarification": True,
                "confidence": intent.confidence,
                "original_message": message,
            },
        )
        self._record_intent_feedback(tenant_id=tenant_id, intent=intent, outcome="clarify")

        yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": clarify_q})
        result = await self.post_process(result, context)
        yield AgentEvent(type=EventType.EXECUTION_END, data=result)
    async def _run_multi_intent(
        self,
        intent: Any,
        tenant_id: str,
        message: str,
        context: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> AsyncIterator[AgentEvent]:
        """Run a decomposed request through the DAG and post-process the outcome.

        The DAG's own EXECUTION_END is forwarded as-is (callers watching the
        stream see per-sub-task progress), then a second EXECUTION_END carries
        the post-processed AgentResult for the request as a whole.
        """
        final_response = ""
        dag_exec_data: Dict[str, Any] = {}

        async for event in self._stream_dag(intent, tenant_id, context, metadata):
            if event.type == EventType.EXECUTION_END:
                dag_exec_data = event.data
                final_response = dag_exec_data.get("final_response", "")
            yield event

        status = (
            AgentStatus.WAITING_FOR_APPROVAL
            if dag_exec_data.get("pending_approvals")
            else AgentStatus.COMPLETED
        )
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

    def _build_result_from_exec_data(
        self,
        exec_data: Dict[str, Any],
        *,
        final_response: str,
        context: Dict[str, Any],
        total_tool_count: int,
    ) -> AgentResult:
        """Map a finished ReAct loop's EXECUTION_END payload onto an AgentResult.

        The status reflects why the loop stopped: a pending approval outranks
        everything (the turn is not done until the user answers), then a request
        for more input, otherwise the turn completed.
        """
        pending_approvals = exec_data.get("pending_approvals", [])

        if pending_approvals:
            status = AgentStatus.WAITING_FOR_APPROVAL
        elif exec_data.get("result_status") == "WAITING_FOR_INPUT":
            status = AgentStatus.WAITING_FOR_INPUT
        elif exec_data.get("result_status") == "FAILED":
            status = AgentStatus.ERROR
        else:
            status = AgentStatus.COMPLETED

        metadata: Dict[str, Any] = {
            "react_turns": exec_data.get("turns", 0),
            "token_usage": exec_data.get("token_usage", {}),
            "duration_ms": exec_data.get("duration_ms", 0),
            "tool_calls_count": exec_data.get("tool_calls_count", 0),
            "total_tool_count": total_tool_count,
        }

        # Carry conditional notification from the notify_user tool
        if context.get("cron_notification"):
            metadata["cron_notification"] = context["cron_notification"]

        if pending_approvals:
            metadata["pending_approvals"] = [
                {
                    "agent_name": a.agent_name,
                    "action_summary": a.action_summary,
                    "details": a.details,
                    "options": a.options,
                }
                for a in pending_approvals
            ]

        return AgentResult(
            agent_type=self.__class__.__name__,
            status=status,
            raw_message=final_response,
            metadata=metadata,
        )

    async def _run_react_with_fallback(
        self,
        *,
        messages: List[Dict[str, Any]],
        tool_schemas: List[Dict[str, Any]],
        tenant_id: str,
        context: Dict[str, Any],
        user_message: str,
        media: Optional[List[Dict[str, Any]]],
        metadata: Dict[str, Any],
        request_tools: List[Any],
        needs_memory: bool,
        routing_task: Optional["asyncio.Task"] = None,
        preserve_messages_on_fallback: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Run the ReAct loop, retrying once on another model if the first fails.

        A model that fails after exhausting its per-call retries raises
        _ReactLoopLLMError. When a fallback provider is configured, the whole
        loop is retried against it from freshly built messages -- the failed
        attempt's transcript is discarded rather than resumed, since it may be
        truncated mid-turn.

        If there is no fallback, or the fallback fails too, this emits an ERROR
        event and then a synthetic EXECUTION_END carrying a graceful,
        user-facing message, so the caller sees the same event shape either way.
        """
        from .error_classifier import error_code_for_kind
        from .graceful_response import generate_graceful_error
        from .react_loop import _ReactLoopLLMError

        retry_seed = copy.deepcopy(messages) if preserve_messages_on_fallback else None

        async def _failed(err: "_ReactLoopLLMError") -> AsyncIterator[AgentEvent]:
            await self._fail_transcript(context)
            yield AgentEvent(
                type=EventType.ERROR,
                data={
                    "code": error_code_for_kind(err.error_kind),
                    "error": str(err.original),
                    "error_type": type(err.original).__name__,
                },
            )
            yield AgentEvent(
                type=EventType.EXECUTION_END,
                data={
                    "final_response": await generate_graceful_error(
                        error=err.original,
                        llm_client=getattr(self, "llm_client", None),
                    ),
                    "result_status": "FAILED",
                    "turns": 0,
                    "token_usage": {},
                    "duration_ms": 0,
                    "tool_calls_count": 0,
                    "tool_calls": [],
                },
            )

        try:
            async for event in self._react_loop_events(
                messages,
                tool_schemas,
                tenant_id,
                context=context,
                user_message=user_message,
                media=media,
                metadata=metadata,
                request_tools=request_tools,
                routing_task=routing_task,
            ):
                yield event
            return
        except _ReactLoopLLMError as loop_err:
            fallback_client = self._resolve_model_fallback(loop_err)
            if fallback_client is None:
                logger.error(
                    f"[Orchestrator] ReAct loop failed, no fallback available: "
                    f"{loop_err.original}"
                )
                async for event in _failed(loop_err):
                    yield event
                return

            logger.warning(
                f"[Orchestrator] ReAct loop failed (turn={loop_err.turn}, "
                f"kind={loop_err.error_kind.value}), retrying with fallback model"
            )

        # Rebuild messages so the retry starts from a clean transcript.
        if retry_seed is not None:
            retry_messages = retry_seed
        else:
            retry_messages = await self._build_llm_messages(
                context, user_message, needs_memory=needs_memory
            )
        try:
            async for event in self._react_loop_events(
                retry_messages,
                tool_schemas,
                tenant_id,
                context=context,
                user_message=user_message,
                media=media,
                metadata=metadata,
                request_tools=request_tools,
                _llm_client_override=fallback_client,
            ):
                yield event
        except _ReactLoopLLMError as retry_err:
            logger.error(f"[Orchestrator] Fallback model also failed: {retry_err.original}")
            async for event in _failed(retry_err):
                yield event

    # ==========================================================================
    # SPECULATIVE EXECUTION
    # ==========================================================================

    _SPECULATIVE_TIMEOUT = 10.0  # seconds before giving up on speculative task

    # ==========================================================================
    # REACT LOOP — see react_loop.py (ReactLoopMixin)
    # LLM CALLS  — see llm_manager.py (LLMManagerMixin)
    # TOOLS      — see tool_manager.py (ToolManagerMixin)
    # ==========================================================================

    # ── Planning helpers ──

    # ==========================================================================
    # MESSAGE BUILDING
    # ==========================================================================

    # ==================================================================
    # DOMAIN-FILTERED TOOL LOADING WITH FALLBACK
    # ==================================================================

    async def _build_tool_schemas_with_domain_fallback(
        self,
        tenant_id: str,
        domains: List[str],
    ) -> List[Dict[str, Any]]:
        """Build tool schemas for a set of domains, falling back to all tools.

        Domain filtering is only as good as the intent classifier. When it
        names a domain no agent serves -- a misclassification, or a domain
        whose agents the tenant has not connected -- the filter yields zero
        agent-tools. The model then has nothing to call and answers from its
        own knowledge, which reads as a confident wrong answer rather than a
        failure.

        Falling back to the full toolset costs prompt tokens but keeps the
        request answerable, so a classifier mistake degrades to "slower and
        broader" instead of "silently wrong".
        """
        schemas = await self._build_tool_schemas(tenant_id, domains=domains)

        # Count how many are actual agent-tools (not builtin tools)
        builtin_names = {t.name for t in getattr(self, "builtin_tools", [])}
        agent_tool_count = sum(
            1
            for s in schemas
            if s.get("function", {}).get("name", s.get("name", "")) not in builtin_names
        )

        if agent_tool_count == 0:
            logger.warning(
                f"[Tools] Domain filter {domains} yielded 0 agent-tools; "
                f"falling back to all tools"
            )
            schemas = await self._build_tool_schemas(tenant_id, domains=None)

        return schemas

    # ==================================================================
    # INTENT ANALYSIS & DAG EXECUTION
    # ==================================================================

    # ==========================================================================
    # EXTENSION POINTS - Override these in subclasses
    # ==========================================================================

    async def prepare_context(
        self, tenant_id: str, message: str, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Prepare context for processing.

        Conversation history is provided by the app layer via
        metadata["conversation_history"].

        Override to add:
        - User preferences
        - Custom metadata

        Args:
            tenant_id: Tenant identifier
            message: User message
            metadata: Request metadata

        Returns:
            Context dict passed to all subsequent methods
        """
        # Lazy restore if needed
        if self.config.session.lazy_restore and not self.agent_pool.has_agents_in_memory(tenant_id):
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

        session_state = self.session_memory.prepare_session(
            session_id,
            message,
            has_active_agents=bool(active_agents),
        )
        context["session_working_memory"] = session_state
        session_prompt = self.session_memory.build_prompt_section(session_id)
        if session_prompt:
            context["session_memory_prompt"] = session_prompt

        return context

    async def should_process(self, message: str, context: Dict[str, Any]) -> bool:
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

    async def reject_message(self, message: str, context: Dict[str, Any]) -> AgentResult:
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

    async def post_process(self, result: AgentResult, context: Dict[str, Any]) -> AgentResult:
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

        # A resumed run replays a message the user sent once, so recording it
        # again would put the same turn in their history for every resume.
        # Only what the assistant produced this time around is new.
        if context.get("resumed"):
            user_message = ""

        self._record_execution_outcome(result, tenant_id, session_id, context)

        status_value = (
            result.status.value if hasattr(result.status, "value") else str(result.status)
        )

        # Build conversation messages for storage
        messages = []
        if user_message:
            messages.append({"role": "user", "content": user_message})
        if result.raw_message:
            messages.append({"role": "assistant", "content": result.raw_message})

        session_snapshot = self.session_memory.update_from_result(
            session_id,
            user_message=user_message,
            assistant_message=result.raw_message,
            result_status=status_value,
            tool_calls=context.get("tool_calls"),
            metadata=result.metadata,
        )
        context["session_working_memory"] = session_snapshot
        session_prompt = self.session_memory.build_prompt_section(session_id)
        if session_prompt:
            context["session_memory_prompt"] = session_prompt

        self._persist_long_term_memory(
            result, context, tenant_id, messages, status_value, user_message
        )

        # True Memory proposal extraction — runs synchronously so proposals
        # are available in result.metadata before the response is returned.
        try:
            proposals = await extract_true_memory_proposals(
                self.llm_client,
                user_message=user_message,
                assistant_response=result.raw_message or "",
                existing_true_memory=(context.get("metadata") or {}).get("true_memory"),
                user_profile=(context.get("metadata") or {}).get("user_profile"),
            )
            if proposals:
                result.metadata["true_memory_proposals"] = _merge_true_memory_proposals(
                    result.metadata.get("true_memory_proposals"),
                    proposals,
                )
        except Exception as e:
            logger.warning(f"True memory proposal extraction failed: {e}")

        # Guardrails output check
        if self.guardrails_checker and result.raw_message:
            try:
                safety_result = await self.guardrails_checker.check_output(
                    result.raw_message,
                    tenant_id,
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

    def _record_execution_outcome(
        self,
        result: AgentResult,
        tenant_id: str,
        session_id: str,
        context: Dict[str, Any],
    ) -> None:
        """Feed the request's outcome back to the intent feedback store.

        Only requests that reached execution land here; the ambiguous path
        records its own 'clarify' outcome when it short-circuits. Fire-and-
        forget: a feedback failure must never affect the user's response.
        """
        intent = context.get("intent_analysis")
        if intent is None:
            return
        try:
            status_name = getattr(result.status, "value", str(result.status)).lower()
            if "cancel" in status_name:
                outcome = "cancelled"
            elif "fail" in status_name or "error" in status_name:
                outcome = "error"
            else:
                outcome = "completed"
            self._record_intent_feedback(
                tenant_id=tenant_id,
                intent=intent,
                outcome=outcome,
                extra={"session_id": session_id},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("post_process feedback record failed: %s", exc)

    def _persist_long_term_memory(
        self,
        result: AgentResult,
        context: Dict[str, Any],
        tenant_id: str,
        messages: List[Dict[str, Any]],
        status_value: str,
        user_message: str,
    ) -> None:
        """Store the turn in long-term memory when governance allows it.

        ``user_message`` is passed rather than read back off the context: a
        resumed run replays a message that is already in memory, and judging
        the turn on it would admit an assistant-only record on the strength of
        something the record does not contain.

        The decision is recorded on the result either way, so a caller can see
        why a turn was or was not kept. The write itself is fire-and-forget:
        embedding and extraction are slow, and the user should not wait on them.
        """
        if not (messages and self.momex):
            return

        decision = self.memory_governance.decide_storage(
            user_message=user_message,
            assistant_message=result.raw_message,
            result_status=status_value,
            metadata={**(context.get("metadata") or {}), **(result.metadata or {})},
        )
        result.metadata["memory_write"] = {
            "stored": decision.should_store,
            "reason": decision.reason,
            "tags": list(decision.tags),
        }
        if not decision.should_store:
            return

        async def _bg_momex_add():
            try:
                moderator = getattr(self.memory_governance, "_moderator", None)
                kwargs: Dict[str, Any] = {
                    "tenant_id": tenant_id,
                    "messages": messages,
                    "infer": True,
                }
                # Only pass moderator when the target accepts it -- this keeps
                # 3rd-party / test doubles that implement the basic
                # `add(tenant_id, messages, infer)` shape working.
                if moderator is not None:
                    try:
                        import inspect

                        sig = inspect.signature(self.momex.add)
                        if "moderator" in sig.parameters or any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in sig.parameters.values()
                        ):
                            kwargs["moderator"] = moderator
                    except (TypeError, ValueError):
                        pass
                await self.momex.add(**kwargs)
            except Exception as e:
                logger.warning(f"Background momex.add failed: {e}")

        self.task_registry.create_task(_bg_momex_add(), name="momex_add")

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

        async def invoke_callback(name: str, data: Optional[Dict[str, Any]] = None) -> Any:
            callback = AgentCallback(event=name, tenant_id=tenant_id, data=data or {})
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

    # ==========================================================================
    # REQUEST-SCOPED AGENT CLEANUP
    # ==========================================================================

    # Default threshold: agents in terminal states (COMPLETED, ERROR, CANCELLED)
    # are removed immediately; non-terminal agents older than this threshold
    # (in seconds) are also purged to prevent cross-session state leakage.
    STALE_AGENT_THRESHOLD_SECONDS = 3600  # 1 hour

    # ==========================================================================
    # SESSION RESTORATION
    # ==========================================================================

    # ==========================================================================
    # AGENT MANAGEMENT API
    # ==========================================================================

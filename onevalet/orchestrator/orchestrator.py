"""
OneValet Orchestrator - Central coordinator for all agents

This module provides an extensible Orchestrator using the Template Method pattern.
Subclass and override extension points to customize behavior.

Extension Points:
    - prepare_context(): Add memories, user info, custom metadata
    - should_process(): Guardrails, rate limits, tier access control
    - reject_message(): Custom rejection handling
    - route_message(): Custom routing logic
    - create_agent(): Custom agent instantiation
    - execute_agent(): Custom agent execution
    - post_process(): Save to memory, notifications, response wrapping

Example:
    class MyOrchestrator(Orchestrator):
        async def should_process(self, message, context):
            # Add guardrails check
            if not await self.safety_checker.check(message):
                return False
            return True

        async def post_process(self, result, context):
            # Save to memory
            await self.memory.save(result)
            # Wrap with personality
            result.raw_message = await self.personality.wrap(result.raw_message)
            return result
"""

import logging
from typing import Dict, List, Optional, Any, AsyncIterator, Callable, TYPE_CHECKING

from ..message import Message
from ..result import AgentResult, AgentStatus
from ..streaming.models import StreamMode, AgentEvent, EventType

from .models import (
    RoutingAction,
    RoutingReason,
    RoutingDecision,
    OrchestratorConfig,
    AgentPoolEntry,
    AgentCallback,
    CALLBACK_HANDLER_ATTR,
    callback_handler,
)
from .pool import AgentPoolManager
from .router import MessageRouter

if TYPE_CHECKING:
    from ..checkpoint import CheckpointManager
    from ..msghub import MessageHub
    from ..workflow import WorkflowExecutor
    from ..protocols import LLMClientProtocol

from ..standard_agent import StandardAgent
from ..config import AgentRegistry

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central coordinator for all agents with extensible design.

    Uses Template Method pattern - override extension points to customize:

    1. prepare_context() - Build context before routing
    2. should_process() - Gate for message processing
    3. reject_message() - Handle rejected messages
    4. route_message() - Custom routing logic
    5. create_agent() - Custom agent instantiation
    6. execute_agent() - Custom execution logic
    7. post_process() - Post-processing before response

    Callback Handlers:
        Use @callback_handler decorator to register handlers that agents can invoke:

        class MyOrchestrator(Orchestrator):
            @callback_handler("get_cache")
            async def get_cache(self, callback: AgentCallback) -> Any:
                return self.cache.get(callback.data["key"])

            @callback_handler("send_sms")
            async def send_sms(self, callback: AgentCallback) -> None:
                await self.sms.send(callback.data["message"])

    Basic Usage:
        orchestrator = Orchestrator(config_dir="./config", llm_client=llm_client)
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
        config: Optional[OrchestratorConfig] = None,
        config_dir: Optional[str] = None,
        llm_client: Optional["LLMClientProtocol"] = None,
        agent_registry: Optional[AgentRegistry] = None,
        checkpoint_manager: Optional["CheckpointManager"] = None,
        message_hub: Optional["MessageHub"] = None,
        workflow_executor: Optional["WorkflowExecutor"] = None,
    ):
        """
        Initialize Orchestrator.

        Args:
            config: Full orchestrator configuration
            config_dir: Path to YAML config directory (alternative to config)
            llm_client: LLM client for agents
            agent_registry: Pre-configured agent registry
            checkpoint_manager: Checkpoint manager for state persistence
            message_hub: Message hub for multi-agent communication
            workflow_executor: Executor for workflows
        """
        # Configuration
        if config:
            self.config = config
        else:
            self.config = OrchestratorConfig(
                config_dir=config_dir or "./config"
            )

        # Core dependencies
        self.llm_client = llm_client
        self.checkpoint_manager = checkpoint_manager
        self.message_hub = message_hub
        self.workflow_executor = workflow_executor

        # Agent registry (lazy loaded if not provided)
        self._agent_registry: Optional[AgentRegistry] = agent_registry
        self._registry_initialized = agent_registry is not None

        # Agent pool manager
        self.agent_pool = AgentPoolManager(config=self.config.session)

        # Message router (initialized after registry)
        self._router: Optional[MessageRouter] = None

        # State
        self._initialized = False

    @property
    def agent_registry(self) -> Optional[AgentRegistry]:
        """Get the agent registry"""
        return self._agent_registry

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
        if not self._registry_initialized:
            await self._init_registry()

        # Validate LLM client is available
        if not self.llm_client:
            raise RuntimeError(
                "LLM client is required. Either pass llm_client to Orchestrator() "
                "or configure it in onevalet.yaml under llm.routing"
            )

        # Validate default_agent_type is registered
        if self.config.default_agent_type:
            if not self._agent_registry:
                raise RuntimeError(
                    f"default_agent_type '{self.config.default_agent_type}' specified but no agent registry available"
                )
            if not self._agent_registry.get_agent_class(self.config.default_agent_type):
                registered_agents = self._agent_registry.get_all_agent_names()
                raise RuntimeError(
                    f"default_agent_type '{self.config.default_agent_type}' not found in agent registry. "
                    f"Available agents: {registered_agents}"
                )

        # Initialize router
        await self._init_router()

        # Restore sessions if configured
        if self.config.session.enabled and self.config.session.auto_restore_on_start:
            await self._restore_sessions()

        # Start auto-backup if configured
        if self.config.session.enabled and self.config.session.auto_backup_interval_seconds > 0:
            await self.agent_pool.start_auto_backup()

        self._initialized = True
        logger.info("Orchestrator initialized")

    async def _init_registry(self) -> None:
        """Initialize agent registry. Override to customize."""
        try:
            self._agent_registry = AgentRegistry(
                config_dir=self.config.config_dir
            )
            await self._agent_registry.initialize()
            self._registry_initialized = True

            # Auto-get routing LLM from registry if not provided
            if self.llm_client is None:
                from ..llm.registry import LLMRegistry
                self.llm_client = LLMRegistry.get_instance().get_routing()
                if self.llm_client:
                    logger.info("Using routing LLM from registry")
        except Exception as e:
            logger.warning(f"Failed to initialize agent registry: {e}")

    async def _init_router(self) -> None:
        """Initialize message router. Override to use custom router."""
        self._router = MessageRouter(
            agent_registry=self._agent_registry,
            llm_client=self.llm_client,
            enable_llm_routing=self.llm_client is not None,
            default_agent_type=self.config.default_agent_type
        )

    async def shutdown(self) -> None:
        """Shutdown the orchestrator gracefully."""
        await self.agent_pool.close()
        if self._agent_registry:
            await self._agent_registry.shutdown()
        self._initialized = False
        logger.info("Orchestrator shutdown")

    # ==========================================================================
    # MAIN ENTRY POINT - TEMPLATE METHOD
    # ==========================================================================

    async def handle_message(
        self,
        tenant_id: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Main entry point - handle user message.

        This implements the Template Method pattern. Override the extension
        points (prepare_context, should_process, etc.) to customize behavior.

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

        # Step 3: Route message
        decision = await self.route_message(message, context)

        # Step 4: Handle workflow execution
        if decision.action == RoutingAction.EXECUTE_WORKFLOW:
            result = await self._execute_workflow(tenant_id, decision, context)
            return await self.post_process(result, context)

        # Step 5: Get or create agent (handles ROUTE_TO_EXISTING, CREATE_NEW, ROUTE_TO_DEFAULT)
        agent = await self.get_or_create_agent(tenant_id, decision, context)
        if not agent:
            result = await self.handle_agent_error(decision, context)
            return await self.post_process(result, context)

        # Step 6: Execute agent
        result = await self.execute_agent(agent, message, context)

        # Step 7: Update pool after execution
        await self.update_pool_after_execution(tenant_id, agent, result)

        # Step 8: Post-process result
        return await self.post_process(result, context)

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
        Prepare context for routing and execution.

        Override to add:
        - User memories from vector DB
        - User preferences/tier info
        - Chat history
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

        return {
            "tenant_id": tenant_id,
            "message": message,
            "metadata": metadata or {},
            "active_agents": active_agents,
        }

    async def should_process(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Check if message should be processed.

        Override to add:
        - Guardrails/safety checks
        - Rate limiting
        - Tier access control
        - Feature flags
        - Input validation

        Args:
            message: User message
            context: Context from prepare_context()

        Returns:
            True to continue processing, False to reject
        """
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

    async def route_message(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> RoutingDecision:
        """
        Route message to appropriate agent.

        Override for custom routing logic, e.g.:
        - Custom LLM prompt with memories
        - Rule-based routing
        - ML-based intent classification

        Args:
            message: User message
            context: Context from prepare_context()

        Returns:
            RoutingDecision specifying what to do
        """
        if self._router:
            return await self._router.route(
                tenant_id=context["tenant_id"],
                message=message,
                active_agents=context.get("active_agents", []),
                metadata=context.get("metadata", {})
            )

        # No router available - route to default agent
        return RoutingDecision(
            action=RoutingAction.ROUTE_TO_DEFAULT,
            agent_type=self.config.default_agent_type,
            confidence=0.0,
            reason=RoutingReason.NO_ROUTER
        )

    async def get_or_create_agent(
        self,
        tenant_id: str,
        decision: RoutingDecision,
        context: Dict[str, Any]
    ) -> Optional[StandardAgent]:
        """
        Get existing agent or create new one based on routing decision.

        Override to customize agent creation, e.g.:
        - Inject custom dependencies
        - Add tenant-specific configuration
        - Modify context_hints

        Args:
            tenant_id: Tenant identifier
            decision: Routing decision
            context: Context from prepare_context()

        Returns:
            Agent instance or None if creation failed
        """
        if decision.action == RoutingAction.ROUTE_TO_EXISTING:
            return await self.agent_pool.get_agent(tenant_id, decision.agent_id)

        if decision.action in (RoutingAction.CREATE_NEW, RoutingAction.ROUTE_TO_DEFAULT):
            return await self.create_agent(
                tenant_id=tenant_id,
                agent_type=decision.agent_type,
                context_hints=decision.context_hints,
                context=context
            )

        return None

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
            context_hints: Hints extracted from message
            context: Full context dict

        Returns:
            New agent instance or None if failed
        """
        if not self._agent_registry:
            logger.error("Cannot create agent: no registry available")
            return None

        try:
            # Use AgentRegistry.create_agent() to respect agent's LLM setting
            # Priority: 1. agent's llm, 2. default llm, 3. orchestrator's llm
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

    async def execute_agent(
        self,
        agent: StandardAgent,
        message: str,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Execute agent with message.

        Override to customize execution:
        - Add pre/post execution hooks
        - Modify message before sending
        - Add execution metrics

        Args:
            agent: Agent to execute
            message: User message
            context: Context dict (may contain "sender_name" and "sender_role")

        Returns:
            AgentResult from agent
        """
        try:
            # Pre-execution: inject recalled memories if enable_memory is set
            await self._inject_memories(agent, message, context)

            metadata = context.get("metadata", {})
            msg = Message(
                name=metadata.get("sender_name", ""),
                content=message,
                role=metadata.get("sender_role", ""),
                metadata=metadata
            )

            result = await agent.reply(msg)

            # Post-execution: store memories if enable_memory is set
            await self._store_memories(agent, message, result, context)

            # Update agent status
            agent.status = result.status

            return result

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return AgentResult(
                agent_type=agent.agent_type,
                status=AgentStatus.ERROR,
                error_message=str(e),
                agent_id=agent.agent_id
            )

    async def _inject_memories(
        self,
        agent: StandardAgent,
        message: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Inject recalled memories into agent if enable_memory is set.

        Override to customize memory recall logic.

        Args:
            agent: The agent to inject memories into
            message: User message (used for memory search)
            context: Context dict
        """
        # Check if agent has enable_memory set in config
        if not self._agent_registry:
            return

        agent_config = self._agent_registry.get_agent_config(agent.__class__.__name__)
        if not agent_config or not agent_config.enable_memory:
            return

        # Memory manager must be provided by subclass
        memory_manager = getattr(self, 'memory_manager', None)
        if not memory_manager:
            logger.debug(f"Memory enabled for {agent.__class__.__name__} but no memory_manager available")
            return

        # Recall memories
        try:
            tenant_id = context.get("tenant_id", agent.tenant_id)
            memories = await memory_manager.search(
                query=message,
                user_id=tenant_id,
                limit=10
            )

            if memories:
                agent.set_recalled_memories(memories)
                logger.debug(f"Injected {len(memories)} memories into {agent.agent_id}")

        except Exception as e:
            logger.warning(f"Failed to recall memories for {agent.agent_id}: {e}")

    async def _store_memories(
        self,
        agent: StandardAgent,
        message: str,
        result: AgentResult,
        context: Dict[str, Any]
    ) -> None:
        """
        Store memories after agent execution if enable_memory is set.

        Override to customize memory storage logic.

        Args:
            agent: The executed agent
            message: Original user message
            result: Agent execution result
            context: Context dict
        """
        # Check if agent has enable_memory set in config
        if not self._agent_registry:
            return

        agent_config = self._agent_registry.get_agent_config(agent.__class__.__name__)
        if not agent_config or not agent_config.enable_memory:
            return

        # Memory manager must be provided by subclass
        memory_manager = getattr(self, 'memory_manager', None)
        if not memory_manager:
            return

        # Only store if agent completed successfully
        if result.status not in (AgentStatus.COMPLETED,):
            return

        # Store the interaction
        try:
            tenant_id = context.get("tenant_id", agent.tenant_id)

            # Create memory content from conversation
            memory_content = f"User: {message}\nAssistant: {result.raw_message}"

            await memory_manager.add(
                messages=memory_content,
                user_id=tenant_id
            )
            logger.debug(f"Stored memory for {agent.agent_id}")

        except Exception as e:
            logger.warning(f"Failed to store memory for {agent.agent_id}: {e}")

    async def handle_agent_error(
        self,
        decision: RoutingDecision,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Handle agent creation/retrieval failure.

        Override to customize error handling.

        Args:
            decision: The routing decision that failed
            context: Context dict

        Returns:
            AgentResult - subclasses define the response
        """
        return AgentResult(
            agent_type=self.__class__.__name__,
            status=AgentStatus.ERROR,
        )

    async def update_pool_after_execution(
        self,
        tenant_id: str,
        agent: StandardAgent,
        result: AgentResult
    ) -> None:
        """
        Update agent pool after execution.

        Override to customize pool management:
        - Custom cleanup logic
        - Metrics tracking
        - State persistence

        Args:
            tenant_id: Tenant identifier
            agent: The executed agent
            result: Execution result
        """
        if agent.status in AgentStatus.terminal_states():
            await self.agent_pool.remove_agent(tenant_id, agent.agent_id)
        else:
            await self.agent_pool.update_agent(agent)

    async def post_process(
        self,
        result: AgentResult,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Post-process result before returning to user.

        Override to add:
        - Save to memory/database
        - Send notifications (SMS, push, email)
        - Wrap with personality/style
        - Add analytics/logging
        - Record API usage

        Args:
            result: Agent result
            context: Context dict

        Returns:
            Modified result
        """
        return result

    # ==========================================================================
    # STREAMING - Uses same extension points
    # ==========================================================================

    async def stream_message(
        self,
        tenant_id: str,
        message: str,
        mode: StreamMode = StreamMode.EVENTS,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[AgentEvent]:
        """
        Stream agent execution events.

        Uses the same extension points as handle_message().

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

        # Prepare context (uses extension point)
        context = await self.prepare_context(tenant_id, message, metadata)

        # Check if should process (uses extension point)
        if not await self.should_process(message, context):
            result = await self.reject_message(message, context)
            yield AgentEvent(
                type=EventType.MESSAGE_CHUNK,
                data={"chunk": result.raw_message}
            )
            return

        # Route message (uses extension point)
        decision = await self.route_message(message, context)

        # Yield routing event
        yield AgentEvent(
            type=EventType.STATE_CHANGE,
            data={
                "routing_action": decision.action.value,
                "agent_type": decision.agent_type,
                "agent_id": decision.agent_id,
                "reason": decision.reason,
            }
        )

        # Handle workflow execution
        if decision.action == RoutingAction.EXECUTE_WORKFLOW:
            result = await self._execute_workflow(tenant_id, decision, context)
            result = await self.post_process(result, context)
            yield AgentEvent(
                type=EventType.MESSAGE_CHUNK,
                data={"chunk": result.raw_message}
            )
            return

        # Get or create agent (handles ROUTE_TO_EXISTING, CREATE_NEW, ROUTE_TO_DEFAULT)
        agent = await self.get_or_create_agent(tenant_id, decision, context)

        if not agent:
            result = await self.handle_agent_error(decision, context)
            result = await self.post_process(result, context)
            yield AgentEvent(
                type=EventType.MESSAGE_CHUNK,
                data={"chunk": result.raw_message}
            )
            return

        # Stream agent execution
        metadata = context.get("metadata", {})
        msg = Message(
            name=metadata.get("sender_name", ""),
            content=message,
            role=metadata.get("sender_role", ""),
            metadata=metadata
        )
        async for event in agent.stream(msg, mode=mode):
            yield event

        # After streaming, construct result for pool update
        result = AgentResult(
            agent_type=agent.agent_type,
            status=agent.status,
            raw_message="",
            agent_id=agent.agent_id
        )

        # Update agent pool
        await self.update_pool_after_execution(tenant_id, agent, result)

    # ==========================================================================
    # INTERNAL METHODS
    # ==========================================================================

    async def _execute_workflow(
        self,
        tenant_id: str,
        decision: RoutingDecision,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Execute a workflow."""
        if not self.workflow_executor:
            return AgentResult(
                agent_type=self.__class__.__name__,
                status=AgentStatus.ERROR,
            )

        try:
            result = await self.workflow_executor.execute(
                workflow_id=decision.workflow_id,
                tenant_id=tenant_id,
                inputs=decision.context_hints or {}
            )
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return AgentResult(
                agent_type=self.__class__.__name__,
                status=AgentStatus.ERROR,
                error_message=str(e),
            )

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
            """
            Invoke a registered callback handler.

            Args:
                name: Name of the callback handler (registered via @callback_handler)
                data: Callback parameters

            Returns:
                Result from the callback handler, or None if not found
            """
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

        Args:
            callback: The callback request from an agent

        Returns:
            Result from the handler, or None if no handler found
        """
        # Look up method name from class-level handler map
        method_name = self._callback_handler_map.get(callback.event)
        if method_name is None:
            logger.warning(f"No callback handler registered for '{callback.event}'")
            return None

        # Get bound method from instance
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
            List of agent info dicts with name, description, group, etc.

        Usage in agent:
            agents = await self.orchestrator_callback("list_agents")
        """
        if not self._agent_registry:
            return []

        result = []
        for name, config in self._agent_registry.get_all_agents().items():
            result.append({
                "name": name,
                "description": config.description,
                "triggers": config.triggers,
                "capabilities": config.capabilities,
            })
        return result

    @callback_handler("get_agent_config")
    async def _builtin_get_agent_config(self, callback: AgentCallback) -> Optional[Dict[str, Any]]:
        """
        Built-in callback: Get configuration for a specific agent.

        Args (in callback.data):
            agent_name: Name of the agent to look up

        Returns:
            Agent config dict or None if not found

        Usage in agent:
            config = await self.orchestrator_callback("get_agent_config", {"agent_name": "WeatherAgent"})
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
            "triggers": config.triggers,
            "capabilities": config.capabilities,
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
                self._create_agent_from_entry
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
                self._create_agent_from_entry
            )
        except Exception as e:
            logger.error(f"Failed to restore session for tenant {tenant_id}: {e}")

    def _create_agent_from_entry(self, entry: AgentPoolEntry) -> StandardAgent:
        """Create agent from pool entry for session restoration."""
        if not self._agent_registry:
            raise RuntimeError("Cannot restore agent: no registry available")

        # Use AgentRegistry.create_agent() to respect agent's LLM setting
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

        context = {"tenant_id": tenant_id}

        if message:
            result = await self.execute_agent(agent, message, context)
        else:
            result = await agent.resume()

        await self.update_pool_after_execution(tenant_id, agent, result)
        return result

"""
OneValet StandardAgent - State-driven agent with field collection

This is the core agent class that provides:
- State machine for conversation flow
- Required field collection pattern (via InputField)
- Approval workflow
- State handlers for each lifecycle phase
- Built-in streaming support

Example with InputField/OutputField (recommended):
    from onevalet import valet, StandardAgent, InputField, OutputField, AgentStatus

    @valet()
    class SendEmailAgent(StandardAgent):
        '''Send emails to users'''

        recipient = InputField(
            prompt="Who should I send to?",
            validator=lambda x: None if "@" in x else "Invalid email",
        )
        subject = InputField("Subject?", required=False)

        message_id = OutputField(str, "ID of sent message")

        async def on_running(self, msg):
            # Access inputs directly
            to = self.recipient

            # Set outputs
            self.message_id = "123"

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Email sent to {to}!"
            )

Legacy Example (still supported):
    class GreetingAgent(StandardAgent):
        def define_required_fields(self):
            return [RequiredField("name", "User's name", "What's your name?")]

        async def on_running(self, msg):
            name = self.collected_fields["name"]
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Hello, {name}!"
            )
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any, AsyncIterator, Tuple, TYPE_CHECKING
from uuid import uuid4

from .base_agent import BaseAgent
from .fields import InputField, OutputField
from .llm.base import LLMResponse, ToolCall as LLMToolCall
from .message import Message
from .models import RequiredField, AgentState, AgentToolContext, AgentTool
from .protocols import LLMClientProtocol
from .result import AgentResult, AgentStatus, ApprovalResult
from .streaming.engine import StreamEngine
from .streaming.models import StreamMode, EventType, AgentEvent

if TYPE_CHECKING:
    from .agents.decorator import InputSpec, OutputSpec

logger = logging.getLogger(__name__)

# Re-declared here to avoid circular import with orchestrator package.
# Canonical definition lives in onevalet.orchestrator.react_config.
__COMPLETE_TASK_TOOL_NAME = "complete_task"
__COMPLETE_TASK_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": __COMPLETE_TASK_TOOL_NAME,
        "description": (
            "Call this tool to signal that you have completed the user's request "
            "and provide your final response. Use this when you have finished all "
            "necessary tool calls and are ready to deliver the final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": (
                        "Your final response to the user. This should be comprehensive "
                        "and include all relevant information gathered from tool calls."
                    ),
                },
            },
            "required": ["result"],
        },
    },
}


# ===== State Transitions =====

# Valid state transitions
STATE_TRANSITIONS = {
    AgentStatus.INITIALIZING: [
        AgentStatus.RUNNING,
        AgentStatus.WAITING_FOR_INPUT,
        AgentStatus.WAITING_FOR_APPROVAL,
        AgentStatus.PAUSED,
        AgentStatus.COMPLETED,
        AgentStatus.ERROR
    ],
    AgentStatus.RUNNING: [
        AgentStatus.COMPLETED,
        AgentStatus.ERROR,
        AgentStatus.PAUSED,
        AgentStatus.WAITING_FOR_INPUT,
        AgentStatus.WAITING_FOR_APPROVAL
    ],
    AgentStatus.WAITING_FOR_INPUT: [
        AgentStatus.RUNNING,
        AgentStatus.WAITING_FOR_APPROVAL,
        AgentStatus.PAUSED,
        AgentStatus.COMPLETED,
        AgentStatus.ERROR,
        AgentStatus.WAITING_FOR_INPUT
    ],
    AgentStatus.WAITING_FOR_APPROVAL: [
        AgentStatus.RUNNING,
        AgentStatus.WAITING_FOR_INPUT,
        AgentStatus.WAITING_FOR_APPROVAL,  # Allow re-approval after modification
        AgentStatus.PAUSED,
        AgentStatus.COMPLETED,
        AgentStatus.CANCELLED,
        AgentStatus.ERROR
    ],
    AgentStatus.PAUSED: [
        AgentStatus.INITIALIZING,  # Resume to any previous state
        AgentStatus.RUNNING,
        AgentStatus.WAITING_FOR_INPUT,
        AgentStatus.WAITING_FOR_APPROVAL,
        AgentStatus.CANCELLED,
        AgentStatus.ERROR
    ],
    AgentStatus.COMPLETED: [],  # Terminal state
    AgentStatus.ERROR: [AgentStatus.CANCELLED],
    AgentStatus.CANCELLED: []  # Terminal state
}


class StandardAgent(BaseAgent):
    """
    State-driven agent with field collection.

    Use InputField and OutputField class variables to define inputs/outputs:

        @valet()
        class SendEmailAgent(StandardAgent):
            '''Send emails'''

            recipient = InputField("Who to send to?")
            subject = InputField("Subject?", required=False)

            message_id = OutputField(str)

            async def on_running(self, msg):
                # Access inputs: self.recipient, self.subject
                # Set outputs: self.message_id = "123"
                return self.make_result(...)

    Override state handlers to customize behavior:
    - on_initializing: Called when agent first starts
    - on_waiting_for_input: Called when collecting fields from user
    - on_waiting_for_approval: Called when waiting for user approval
    - on_running: Called when all fields collected and approved
    - on_paused: Called when agent is paused
    - on_error: Called when an error occurs
    """

    # Class-level field specs (populated by @valet decorator)
    _input_specs: List["InputSpec"] = []
    _output_specs: List["OutputSpec"] = []

    # Domain ReAct loop configuration (active when tools is non-empty)
    domain_system_prompt: str = ""
    tools: List[AgentTool] = []
    max_turns: int = 5
    max_complete_task_retries: int = 3
    tool_timeout: float = 30.0  # seconds per tool call
    max_tool_result_chars: int = 4000  # truncate tool results beyond this

    _COMPLETE_TASK_INSTRUCTION = (
        "\n\nIMPORTANT: When you have finished the task, you MUST call the "
        "`complete_task` tool with your final answer in the `result` parameter. "
        "This is the ONLY way to finish. Never respond with plain text without "
        "calling `complete_task`."
    )

    def __init__(
        self,
        tenant_id: str = "",
        llm_client: Optional[LLMClientProtocol] = None,
        orchestrator_callback: Optional[Callable] = None,
        context_hints: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize StandardAgent

        Args:
            tenant_id: Tenant identifier for multi-tenant isolation (default: "default")
            llm_client: LLM client (usually auto-injected by registry)
            orchestrator_callback: Callback function for events
            context_hints: Pre-extracted fields from orchestrator
        """
        super().__init__(name=kwargs.get('name'))

        # Core attributes
        self.tenant_id = tenant_id
        self.agent_id = self._generate_agent_id()
        self.llm_client = llm_client
        self.orchestrator_callback = orchestrator_callback

        # State management
        self.status = AgentStatus.INITIALIZING
        self.collected_fields: Dict[str, Any] = {}
        self._output_values: Dict[str, Any] = {}  # Store output values
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.error_message: Optional[str] = None

        # Build required_fields from InputField specs or legacy define_required_fields()
        self.required_fields = self._build_required_fields()

        # Track validation errors for custom error messages
        self._validation_error: Optional[str] = None

        # Instance metadata - for custom per-instance properties (e.g., user_id, session_id)
        self.metadata: Dict[str, Any] = {}

        # Pause management
        self._pause_requested = False
        self._status_before_pause: Optional[AgentStatus] = None

        # Execution state (for checkpoint/restore)
        self.execution_state: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        self._message_history: List["Message"] = []

        # Context hints from orchestrator
        self.context_hints = context_hints or {}

        # Recalled memories from orchestrator (when enable_memory=true)
        self._recalled_memories: List[Dict[str, Any]] = []

        # Domain ReAct loop state (only active when tools is non-empty)
        self._react_messages: List[Dict[str, Any]] = []
        self._react_turn: int = 0
        self._pending_tool_call: Optional[Tuple[LLMToolCall, AgentTool, Dict[str, Any]]] = None
        self._remaining_tool_calls: List[LLMToolCall] = []
        self._tool_trace: List[Dict[str, Any]] = []

        # Pre-populate collected_fields with context_hints (validate each field)
        if context_hints:
            for field_name, value in context_hints.items():
                if not value:
                    continue
                # Check if field has a validator via RequiredField
                field_def = next((f for f in self.required_fields if f.name == field_name), None)
                if field_def and field_def.validator:
                    if not field_def.validator(str(value)):
                        logger.debug(f"context_hints field '{field_name}' failed validation, skipping")
                        continue
                self.collected_fields[field_name] = value
            logger.debug(f"Pre-populated fields from context_hints: {list(self.collected_fields.keys())}")

        # Initialize optional fields with defaults
        self._init_optional_fields()

        # Built-in streaming engine
        self._stream_engine = StreamEngine(
            agent_id=self.agent_id,
            agent_type=self.__class__.__name__
        )
        self._streaming_enabled = False

        logger.debug(f"Initialized {self.__class__.__name__} (ID: {self.agent_id}, Tenant: {tenant_id})")

    def _build_required_fields(self) -> List[RequiredField]:
        """Build RequiredField list from InputField specs or legacy method."""
        # First check for InputField specs from decorator
        input_specs = getattr(self.__class__, '_input_specs', [])
        if input_specs:
            return [
                RequiredField(
                    name=spec.name,
                    description=spec.description,
                    prompt=spec.prompt,
                    validator=self._wrap_validator(spec.validator) if spec.validator else None,
                    required=spec.required,
                )
                for spec in input_specs
            ]
        # Fallback to legacy method
        return self.define_required_fields()

    def _wrap_validator(self, validator: Callable) -> Callable[[str], bool]:
        """
        Wrap a validator that returns error message into one that returns bool.
        Store the error message for later use.
        """
        def wrapped(value: str) -> bool:
            result = validator(value)
            if result is None:
                self._validation_error = None
                return True
            else:
                self._validation_error = result
                return False
        return wrapped

    def _init_optional_fields(self) -> None:
        """Initialize optional fields with their defaults."""
        input_specs = getattr(self.__class__, '_input_specs', [])
        for spec in input_specs:
            if not spec.required and spec.default is not None:
                if spec.name not in self.collected_fields:
                    self.collected_fields[spec.name] = spec.default

    # ===== Agent ReAct Support =====

    def get_system_prompt(self) -> str:
        """Return the system prompt for the mini ReAct loop.

        Override in subclasses to customize. Only used when tools is non-empty.
        """
        return self.domain_system_prompt + self._COMPLETE_TASK_INSTRUCTION

    # ===== Required Methods (Must Override) =====

    def define_required_fields(self) -> List[RequiredField]:
        """
        Define what information this agent needs.

        Returns:
            List of RequiredField objects

        Example:
            def define_required_fields(self):
                return [
                    RequiredField("name", "User's name", "What's your name?"),
                    RequiredField("email", "Email", "What's your email?", lambda v: "@" in v)
                ]
        """
        return []  # Default: no required fields

    # ===== State Handlers (Override to customize) =====

    async def on_initializing(self, msg: Message) -> AgentResult:
        """
        Called when agent first receives a message.

        Default behavior: Extract fields and transition to appropriate state.
        Override for custom initialization logic.
        """
        # Extract fields from initial message
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        # Check if we have all required fields
        missing = self._get_missing_fields()

        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing
            )

        # All fields collected - check approval
        if self.needs_approval():
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )

        # No approval needed - go directly to running
        self.transition_to(AgentStatus.RUNNING)
        return await self.on_running(msg)

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        """
        Called when waiting for user to provide missing fields.

        If tools is active and a ReAct loop is in progress,
        resumes the ReAct loop with the user's answer.
        Otherwise, continues InputField collection.
        """
        # Domain ReAct path: resume loop with user's follow-up
        if self.tools and self._react_messages:
            user_text = msg.get_text() if msg else ""
            if not user_text:
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message="Please provide the requested information.",
                )
            self._react_messages.append({"role": "user", "content": user_text})
            self.transition_to(AgentStatus.RUNNING)
            return await self._run_react()

        # InputField collection path
        if msg:
            success = await self._extract_and_collect_fields(msg.get_text())

            # Validation failed - show error and re-ask
            if not success and self._validation_error:
                prompt = self._get_next_prompt() or ""
                error_message = f"{self._validation_error} {prompt}"
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message=error_message,
                    missing_fields=self._get_missing_fields()
                )

        missing = self._get_missing_fields()

        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing
            )

        # All fields collected - check approval
        if self.needs_approval():
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )

        # No approval needed - execute
        self.transition_to(AgentStatus.RUNNING)
        return await self.on_running(msg)

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        """
        Called when waiting for user approval.

        If a domain tool call is pending, uses LLM-based approval parsing.
        Otherwise, uses the InputField-based approval flow.
        """
        if self._pending_tool_call:
            # Domain ReAct path: LLM-based approval parsing
            user_input = msg.get_text() if msg else ""
            approval = await self._parse_approval_with_llm(user_input)

            if approval == ApprovalResult.APPROVED:
                return await self._resume_after_approval()
            if approval == ApprovalResult.REJECTED:
                self._pending_tool_call = None
                self._remaining_tool_calls = []
                self._tool_trace.append(
                    {"tool": "approval", "status": "rejected", "summary": "User rejected approval."}
                )
                return self.make_result(
                    status=AgentStatus.CANCELLED,
                    raw_message="Operation cancelled.",
                    metadata={
                        "tool_trace": list(self._tool_trace),
                        "tool_calls_count": len(self._tool_trace),
                    },
                )
            # MODIFY
            self._pending_tool_call = None
            self._remaining_tool_calls = []
            self._tool_trace.append({
                "tool": "approval",
                "status": "modified",
                "summary": f"User requested modification: {user_input[:180]}",
            })
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message=f"Operation cancelled. User said: {user_input}",
                metadata={
                    "tool_trace": list(self._tool_trace),
                    "tool_calls_count": len(self._tool_trace),
                },
            )

        # InputField-based approval path
        user_input = msg.get_text() if msg else ""
        approval = self.parse_approval(user_input)

        if approval == ApprovalResult.APPROVED:
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

        elif approval == ApprovalResult.REJECTED:
            return self.make_result(status=AgentStatus.CANCELLED)

        else:  # MODIFY
            # Try to extract new field values
            await self._extract_and_collect_fields(user_input)

            missing = self._get_missing_fields()
            if missing:
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message=self._get_next_prompt(),
                    missing_fields=missing
                )

            # Still have all fields, ask for approval again
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )

    async def on_running(self, msg: Message) -> AgentResult:
        """
        Called when all fields are collected and approved.

        If tools is non-empty, runs the mini ReAct loop automatically.
        Otherwise, subclasses override this for custom business logic.

        Example:
            async def on_running(self, msg):
                name = self.collected_fields["name"]
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Hello, {name}!"
                )
        """
        if self.tools:
            # Domain ReAct path
            if self._pending_tool_call:
                return await self._resume_after_approval()

            instruction = self.collected_fields.get("task_instruction", "")
            if not instruction and msg:
                instruction = msg.get_text()

            if not instruction:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="No task instruction provided.",
                    metadata={
                        "tool_trace": list(self._tool_trace),
                        "tool_calls_count": len(self._tool_trace),
                    },
                )

            self._react_messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": instruction},
            ]
            self._react_turn = 0
            self._tool_trace = []
            return await self._run_react()

        # Default for non-domain subclasses (they override this)
        return self.make_result(
            status=AgentStatus.COMPLETED
        )

    async def on_error(self, msg: Message) -> AgentResult:
        """
        Called when agent is in error state.

        Override to implement error recovery logic.
        """
        return self.make_result(
            status=AgentStatus.ERROR,
            error_message=self.error_message
        )

    async def on_paused(self, msg: Message) -> AgentResult:
        """
        Called when agent is in paused state and receives a message.

        Override to implement pause handling logic.

        Args:
            msg: Message received while paused

        Returns:
            AgentResult - call self.resume() to resume, or return CANCELLED/PAUSED status
        """
        # Default: stay paused. Subclasses implement their own logic.
        return self.make_result(status=AgentStatus.PAUSED)

    # ===== Pause Control =====

    def request_pause(self) -> bool:
        """
        Request the agent to pause at the next safe point.

        This sets a flag that the agent checks during execution.
        The actual pause happens when the agent reaches a safe state.

        Returns:
            True if pause request was accepted, False if agent cannot be paused
        """
        # Can only pause from active states
        pauseable_states = {
            AgentStatus.RUNNING,
            AgentStatus.WAITING_FOR_INPUT,
            AgentStatus.WAITING_FOR_APPROVAL,
            AgentStatus.INITIALIZING
        }

        if self.status not in pauseable_states:
            logger.warning(f"Cannot pause agent in {self.status} state")
            return False

        self._pause_requested = True
        logger.debug(f"Pause requested for {self.agent_id}")
        return True

    def pause(self) -> AgentResult:
        """
        Immediately pause the agent.

        Saves the current status so it can be restored on resume.

        Returns:
            AgentResult with PAUSED status
        """
        if self.status == AgentStatus.PAUSED:
            return self.make_result(status=AgentStatus.PAUSED)

        # Save status before pausing
        self._status_before_pause = self.status
        self._pause_requested = False

        return self.make_result(status=AgentStatus.PAUSED)

    async def resume(self) -> AgentResult:
        """
        Resume the agent from paused state.

        Restores the previous status and continues execution.

        Returns:
            AgentResult from the resumed handler
        """
        if self.status != AgentStatus.PAUSED:
            return self.make_result(status=self.status)

        # Restore previous status
        previous_status = self._status_before_pause or AgentStatus.WAITING_FOR_INPUT
        self._status_before_pause = None
        self._pause_requested = False

        # Transition to previous status
        self.transition_to(previous_status)

        # Return appropriate result based on restored status
        if previous_status == AgentStatus.WAITING_FOR_INPUT:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt() or ""
            )
        elif previous_status == AgentStatus.WAITING_FOR_APPROVAL:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )
        else:
            return self.make_result(status=previous_status)

    def is_paused(self) -> bool:
        """Check if agent is currently paused."""
        return self.status == AgentStatus.PAUSED

    def is_pause_requested(self) -> bool:
        """Check if a pause has been requested."""
        return self._pause_requested

    # ===== Result Factory =====

    def make_result(
        self,
        status: AgentStatus,
        raw_message: str = "",
        data: Optional[Dict[str, Any]] = None,
        missing_fields: Optional[List[str]] = None,
        **kwargs
    ) -> AgentResult:
        """
        Factory method to create AgentResult with auto-filled agent_type and agent_id.

        This method also automatically transitions the agent to the new status.

        Args:
            status: Target agent status (agent will transition to this status)
            raw_message: The response message to show user
            data: Collected field data (defaults to self.collected_fields)
            missing_fields: List of missing field names
            **kwargs: Additional fields to pass to AgentResult

        Example:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Hello, {name}!"
            )
        """
        # Auto transition to the new status
        self.transition_to(status)

        return AgentResult(
            agent_type=self.__class__.__name__,
            agent_id=self.agent_id,
            status=status,
            raw_message=raw_message,
            data=data if data is not None else self.collected_fields,
            missing_fields=missing_fields,
            **kwargs
        )

    # ===== Approval Control =====

    def needs_approval(self) -> bool:
        """
        Whether agent requires user approval before execution.

        Returns:
            True if approval needed, False otherwise

        Override for specific behavior. Default is False.
        """
        return False

    def get_approval_prompt(self) -> str:
        """
        Generate approval prompt for user.

        Override to provide custom approval messages.
        If needs_approval() returns True, this MUST be overridden.

        Example:
            def get_approval_prompt(self):
                return f"Send email to {self.collected_fields['to']}? (yes/no)"
        """
        return ""

    # ===== Main Entry Point =====

    async def reply(self, msg: Message = None) -> AgentResult:
        """
        Main entry point - dispatches to appropriate state handler.

        This method routes to the correct on_xxx handler based on current status.
        You typically don't need to override this.
        """
        try:
            self.last_active = datetime.now()

            # Add message to history
            self.add_to_history(msg)

            # Dispatch to state handler
            if self.status == AgentStatus.INITIALIZING:
                return await self.on_initializing(msg)

            elif self.status == AgentStatus.WAITING_FOR_INPUT:
                return await self.on_waiting_for_input(msg)

            elif self.status == AgentStatus.WAITING_FOR_APPROVAL:
                return await self.on_waiting_for_approval(msg)

            elif self.status == AgentStatus.RUNNING:
                return await self.on_running(msg)

            elif self.status == AgentStatus.ERROR:
                return await self.on_error(msg)

            elif self.status == AgentStatus.PAUSED:
                return await self.on_paused(msg)

            elif self.status == AgentStatus.COMPLETED:
                return self.make_result(status=AgentStatus.COMPLETED)

            elif self.status == AgentStatus.CANCELLED:
                return self.make_result(status=AgentStatus.CANCELLED)

            else:
                return self.make_result(status=AgentStatus.ERROR)

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            self.error_message = str(e)
            self.transition_to(AgentStatus.ERROR)
            return await self.on_error(msg)

    # ===== Field Extraction =====

    async def _extract_and_collect_fields(self, user_input: str) -> bool:
        """
        Extract fields from user input and add to collected_fields.

        Returns:
            True if field was collected successfully, False if validation failed
        """
        if not user_input:
            return False

        extracted = await self.extract_fields(user_input)

        for field_name, value in extracted.items():
            if value is None:
                continue

            # Try to validate using InputField descriptor first
            input_field = self._get_input_field(field_name)
            if input_field:
                error = input_field.validate(value)
                if error:
                    self._validation_error = error
                    return False

            # Fallback to legacy RequiredField validator
            field_def = next((f for f in self.required_fields if f.name == field_name), None)
            if field_def and field_def.validator:
                if not field_def.validator(str(value)):
                    # _validation_error is set by wrapped validator
                    return False

            self.collected_fields[field_name] = value
            self._validation_error = None

        return True

    def _get_input_field(self, name: str) -> Optional[InputField]:
        """Get InputField descriptor by name."""
        for attr_name in dir(self.__class__):
            attr = getattr(self.__class__, attr_name, None)
            if isinstance(attr, InputField) and attr.name == name:
                return attr
        return None

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """
        Extract field values from user input using LLM.

        Override for custom extraction logic.

        Args:
            user_input: User's message

        Returns:
            Dict of field_name -> extracted_value
        """
        missing = self._get_missing_fields()
        if not missing:
            return {}

        # Use LLM for extraction
        if self.llm_client:
            extracted = await self._extract_fields_with_llm(user_input, missing)
            if extracted:
                return extracted

        # Fallback: one field at a time
        if len(missing) == 1:
            return {missing[0]: user_input.strip()}

        return {}

    async def _extract_fields_with_llm(
        self,
        user_input: str,
        missing_fields: List[str]
    ) -> Dict[str, Any]:
        """Use LLM to extract field values from user input."""
        # Build field descriptions
        field_info = []
        for field_name in missing_fields:
            input_field = self._get_input_field(field_name)
            desc = input_field.description if input_field else field_name
            field_info.append(f"- {field_name}: {desc}")

        # Build context from original request and already-collected fields
        context_parts = []
        task_instr = self.collected_fields.get("task_instruction") or self.context_hints.get("task_instruction")
        if task_instr:
            context_parts.append(f"Original request: \"{task_instr}\"")
        known_field_names = {f.name for f in self.required_fields}
        collected = {k: v for k, v in self.collected_fields.items()
                     if k in known_field_names and v}
        if collected:
            context_parts.append("Already collected: " + ", ".join(f"{k}={v}" for k, v in collected.items()))
        context_block = "\n".join(context_parts) + "\n" if context_parts else ""

        prompt = f"""Extract field values from the user message AND infer related values from context.

RULES:
1. Extract values explicitly stated in the user message.
2. Infer values that can be calculated from context + extracted values.
   - Duration + one date → calculate the other date.
   - Example: original request says "三天" (3 days), user says start is "tomorrow" → end_date = start + 3 days.
3. Return dates as YYYY-MM-DD when you can calculate them. Today is {datetime.now().strftime("%Y-%m-%d")}.
4. Fill as many fields as possible. Do NOT leave a field empty if it can be inferred.

{context_block}Fields to extract:
{chr(10).join(field_info)}

User message: "{user_input}"

Return JSON only."""

        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            config={"response_format": {"type": "json_object"}}
        )

        # Parse JSON response
        content = response.content if hasattr(response, 'content') else str(response)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}


    # ===== Approval Parsing =====

    def parse_approval(self, user_input: str) -> ApprovalResult:
        """
        Parse user's approval response.

        MUST be overridden by subclasses that use approval flow.

        Args:
            user_input: User's response to approval prompt

        Returns:
            ApprovalResult.APPROVED, ApprovalResult.REJECTED, or ApprovalResult.MODIFY
        """
        # Default: treat as modify (ask again)
        return ApprovalResult.MODIFY

    # ===== Helper Methods =====

    def _get_missing_fields(self) -> List[str]:
        """Get list of missing required field names."""
        return [f.name for f in self.required_fields
                if f.required and f.name not in self.collected_fields]

    def _get_next_prompt(self) -> Optional[str]:
        """Get the next question to ask user."""
        for field in self.required_fields:
            if field.required and field.name not in self.collected_fields:
                return field.prompt
        return None

    def get_state_summary(self) -> Dict[str, Any]:
        """Get standardized state summary."""
        missing = self._get_missing_fields()
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "required_fields": [f.name for f in self.required_fields],
            "collected_fields": dict(self.collected_fields),
            "missing_fields": missing,
            "next_prompt": self._get_next_prompt() if missing else None,
            "last_active": self.last_active.isoformat(),
            "error_message": self.error_message
        }

    def is_completed(self) -> bool:
        """Check if agent has completed its task."""
        return self.status == AgentStatus.COMPLETED

    def get_message_history(self) -> List["Message"]:
        """Get copy of message history for checkpoint."""
        return self._message_history.copy()

    def add_to_history(self, msg: "Message") -> None:
        """Add a message to history."""
        if msg:
            self._message_history.append(msg)

    def _generate_agent_id(self) -> str:
        """Generate unique agent ID."""
        return f"{self.__class__.__name__}_{uuid4().hex[:8]}"

    # ===== State Transitions =====

    def can_transition(self, from_state: AgentStatus, to_state: AgentStatus) -> bool:
        """Validate state transition."""
        if to_state == AgentStatus.CANCELLED:
            return True
        allowed = STATE_TRANSITIONS.get(from_state, [])
        return to_state in allowed

    def transition_to(self, new_status: AgentStatus) -> bool:
        """Transition to new status with validation."""
        if not self.can_transition(self.status, new_status):
            logger.warning(f"Invalid transition: {self.status} -> {new_status}")
            return False

        old_status = self.status
        self.status = new_status
        self.last_active = datetime.now()

        logger.debug(f"{self.agent_id}: {old_status.value} -> {new_status.value}")

        # Emit state change event if streaming is enabled
        if self._streaming_enabled:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._stream_engine.emit_state_change(
                    old_status.value, new_status.value
                ))
            except RuntimeError:
                pass  # No running loop

        return True

    # ===== Streaming Support =====

    async def stream(
        self,
        msg: Message = None,
        mode: StreamMode = StreamMode.EVENTS
    ) -> AsyncIterator[AgentEvent]:
        """
        Stream agent execution events.

        This is the streaming version of reply(). It yields events as the agent
        executes, including state changes, message chunks, tool calls, etc.

        Args:
            msg: Input message
            mode: Streaming mode (EVENTS, MESSAGES, UPDATES, VALUES)

        Yields:
            AgentEvent objects

        Example:
            async for event in agent.stream(msg):
                if event.type == EventType.MESSAGE_CHUNK:
                    print(event.data["chunk"], end="")
                elif event.type == EventType.STATE_CHANGE:
                    print(f"State: {event.data['new_status']}")
                elif event.type == EventType.TOOL_CALL_START:
                    print(f"Calling: {event.data['tool_name']}")
        """
        self._streaming_enabled = True

        # Execute reply in background (emits events to stream engine)
        reply_task = asyncio.create_task(self._execute_with_streaming(msg))

        # Yield events as they come
        try:
            async for event in self._stream_engine.stream(mode):
                yield event

                # Check if reply is done
                if reply_task.done():
                    # Emit final events
                    result = reply_task.result()
                    if result:
                        await self._stream_engine.emit(
                            EventType.EXECUTION_END,
                            {
                                "status": result.status.value,
                                "raw_message": result.raw_message,
                            }
                        )
                    break

        finally:
            self._streaming_enabled = False
            self._stream_engine.close()

    async def _execute_with_streaming(self, msg: Message) -> AgentResult:
        """Execute reply with streaming events."""
        # Emit execution start
        await self._stream_engine.emit(
            EventType.EXECUTION_START,
            {
                "agent_id": self.agent_id,
                "agent_type": self.__class__.__name__,
                "status": self.status.value,
            }
        )

        # Execute reply
        result = await self.reply(msg)

        return result

    async def emit_message_chunk(self, chunk: str) -> None:
        """
        Emit a message chunk during streaming.

        Call this from your on_running() handler when streaming LLM responses.

        Args:
            chunk: Text chunk to emit

        Example:
            async def on_running(self, msg):
                async for chunk in self.llm_client.stream_completion(messages):
                    await self.emit_message_chunk(chunk.content)
                return self.make_result(...)
        """
        if self._streaming_enabled:
            await self._stream_engine.emit_message_chunk(chunk)

    async def emit_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        call_id: Optional[str] = None
    ) -> None:
        """
        Emit a tool call event during streaming.

        Args:
            tool_name: Name of the tool being called
            tool_input: Input arguments for the tool
            call_id: Optional call identifier
        """
        if self._streaming_enabled:
            await self._stream_engine.emit_tool_call(tool_name, tool_input, call_id)

    async def emit_tool_result(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        error: Optional[str] = None,
        call_id: Optional[str] = None
    ) -> None:
        """
        Emit a tool result event during streaming.

        Args:
            tool_name: Name of the tool that was called
            result: Result from the tool
            success: Whether the tool call succeeded
            error: Error message if failed
            call_id: Optional call identifier
        """
        if self._streaming_enabled:
            await self._stream_engine.emit_tool_result(
                tool_name, result, success, error, call_id
            )

    async def emit_progress(
        self,
        current: int,
        total: int,
        message: Optional[str] = None
    ) -> None:
        """
        Emit a progress event during streaming.

        Args:
            current: Current progress value
            total: Total progress value
            message: Optional progress message
        """
        if self._streaming_enabled:
            await self._stream_engine.emit_progress(current, total, message)

    @property
    def agent_type(self) -> str:
        """Get the agent type (class name)."""
        return self.__class__.__name__

    @property
    def stream_engine(self) -> StreamEngine:
        """Get the stream engine for advanced usage."""
        return self._stream_engine

    # ===== Memory Support =====

    @property
    def recalled_memories(self) -> List[Dict[str, Any]]:
        """
        Get recalled memories for this agent.

        Memories can be set externally via set_recalled_memories().
        The orchestrator provides a recall_memory tool for on-demand LLM queries
        rather than auto-injecting memories before each agent call.

        Each memory dict contains:
            - memory: The memory text
            - user_id: Associated user ID
            - created_at: When memory was created
            - ... other mem0 fields

        Usage in agent:
            async def on_running(self, msg):
                if self.recalled_memories:
                    context = "Relevant memories:\\n"
                    for mem in self.recalled_memories:
                        context += f"- {mem['memory']}\\n"
                    # Use context in your LLM prompt
        """
        return self._recalled_memories

    def set_recalled_memories(self, memories: List[Dict[str, Any]]) -> None:
        """
        Set recalled memories (called by orchestrator).

        Args:
            memories: List of memory dicts from MemoryManager.search()
        """
        self._recalled_memories = memories or []
        if memories:
            logger.debug(f"Set {len(memories)} recalled memories for {self.agent_id}")

    # ===== Agent ReAct Loop =====

    async def _run_react(self) -> AgentResult:
        """Core mini ReAct loop with agent tools."""
        tool_schemas = [t.to_openai_schema() for t in self.tools]
        # Always inject complete_task
        tool_schemas.append(_COMPLETE_TASK_SCHEMA)
        messages = self._react_messages

        if self._remaining_tool_calls:
            result = await self._execute_tool_calls(self._remaining_tool_calls, messages)
            self._remaining_tool_calls = []
            if result is not None:
                return result

        for turn in range(self._react_turn, self.max_turns):
            self._react_turn = turn + 1
            # First turn: force tool use since orchestrator already routed here.
            # Subsequent turns: let LLM decide freely.
            tool_choice = "required" if turn == 0 and tool_schemas else "auto"
            response: LLMResponse = await self.llm_client.chat_completion(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice=tool_choice,
            )

            if not response.has_tool_calls:
                text = response.content or ""
                # If the LLM responded with text on the first turn (no tool
                # called), it likely needs info from the user (e.g. missing
                # email address).  Return WAITING_FOR_INPUT so the agent stays
                # alive and the user can reply.
                if turn == 0 and text and self._looks_like_question(text):
                    self._react_messages = messages
                    return self.make_result(
                        status=AgentStatus.WAITING_FOR_INPUT,
                        raw_message=text,
                        metadata={
                            "tool_trace": list(self._tool_trace),
                            "tool_calls_count": len(self._tool_trace),
                        },
                    )

                # Grace turn: LLM forgot to call complete_task.
                # Retry up to max_complete_task_retries times.
                max_retries = self.max_complete_task_retries
                for retry in range(1, max_retries + 1):
                    logger.warning(
                        f"[{self.__class__.__name__}:{self.name}] turn={turn} no tool calls, "
                        f"grace retry {retry}/{max_retries}"
                    )
                    messages.append(self._format_assistant_msg(response))
                    messages.append({
                        "role": "user",
                        "content": (
                            "You must call the `complete_task` tool with your final "
                            "response in the `result` parameter to finish. Call it now."
                        ),
                    })
                    try:
                        response = await self.llm_client.chat_completion(
                            messages=messages,
                            tools=tool_schemas,
                            tool_choice="required",
                        )
                    except Exception as e:
                        logger.error(
                            f"[{self.__class__.__name__}:{self.name}] grace retry {retry} "
                            f"LLM call failed: {e}"
                        )
                        return self.make_result(
                            status=AgentStatus.ERROR,
                            raw_message=(
                                "Internal error: failed to complete the task. "
                                "Please try again."
                            ),
                            metadata={
                                "tool_trace": list(self._tool_trace),
                                "tool_calls_count": len(self._tool_trace),
                            },
                        )
                    if response.has_tool_calls:
                        break  # success — proceed to tool execution

                # Exhausted all retries — ask LLM to produce a user-friendly
                # error message (no tools, so it can only return text).
                if not response.has_tool_calls:
                    logger.error(
                        f"[{self.__class__.__name__}:{self.name}] exhausted {max_retries} "
                        f"grace retries, LLM still did not call complete_task"
                    )
                    messages.append(self._format_assistant_msg(response))
                    messages.append({
                        "role": "user",
                        "content": (
                            "There was an internal issue processing your request. "
                            "Generate a short, friendly apology to the user in "
                            "their language, and suggest they try again later."
                        ),
                    })
                    try:
                        fallback_resp = await self.llm_client.chat_completion(
                            messages=messages,
                            tools=None,
                        )
                        friendly_msg = fallback_resp.content or (
                            "Sorry, something went wrong. Please try again later."
                        )
                    except Exception:
                        friendly_msg = "Sorry, something went wrong. Please try again later."
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=friendly_msg,
                        metadata={
                            "tool_trace": list(self._tool_trace),
                            "tool_calls_count": len(self._tool_trace),
                            "complete_task_fallback": True,
                        },
                    )

            messages.append(self._format_assistant_msg(response))
            result = await self._execute_tool_calls(response.tool_calls, messages)
            if result is not None:
                return result

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=(
                "I wasn't able to complete the task within the allowed steps. "
                "Please try again with more specific information."
            ),
            metadata={
                "tool_trace": list(self._tool_trace),
                "tool_calls_count": len(self._tool_trace),
            },
        )

    async def _execute_tool_calls(
        self,
        tool_calls: List[LLMToolCall],
        messages: List[Dict[str, Any]],
    ) -> Optional[AgentResult]:
        """Execute tool calls. Returns AgentResult if paused for approval, None otherwise."""
        for i, tc in enumerate(tool_calls):
            # Intercept complete_task — extract result and finish
            if tc.name == _COMPLETE_TASK_TOOL_NAME:
                try:
                    args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result_text = args.get("result", "")
                if result_text:
                    self._tool_trace.append({
                        "tool": _COMPLETE_TASK_TOOL_NAME,
                        "status": "ok",
                        "summary": result_text[:240],
                    })
                    logger.info(
                        f"[{self.__class__.__name__}:{self.name}] complete_task called "
                        f"({len(result_text)} chars)"
                    )
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=result_text,
                        metadata={
                            "tool_trace": list(self._tool_trace),
                            "tool_calls_count": len(self._tool_trace),
                        },
                    )
                else:
                    # Missing result — append error and continue
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": 'Error: "result" argument is required for complete_task.',
                    })
                    continue

            tool = self._find_tool(tc.name)
            if tool is None:
                error_text = f"Error: Unknown tool '{tc.name}'"
                self._tool_trace.append(
                    {"tool": tc.name, "status": "error", "summary": error_text}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": error_text,
                    }
                )
                continue

            if isinstance(tc.arguments, dict):
                args = tc.arguments
            elif isinstance(tc.arguments, str):
                try:
                    args = json.loads(tc.arguments)
                except (json.JSONDecodeError, ValueError) as e:
                    error_text = (
                        f"Error: Failed to parse arguments for tool '{tc.name}': {e}. "
                        "Please retry with valid JSON arguments."
                    )
                    self._tool_trace.append(
                        {"tool": tc.name, "status": "error", "summary": error_text[:240]}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": error_text,
                        }
                    )
                    continue
            else:
                args = {}

            requires_approval = tool.needs_approval or tool.risk_level in ("write", "destructive")
            if requires_approval:
                if tool.get_preview:
                    try:
                        preview = await tool.get_preview(args, self._build_tool_context())
                    except Exception as e:
                        logger.error(f"Preview generation failed for {tc.name}: {e}")
                        preview = f"About to execute: {tc.name}({json.dumps(args, ensure_ascii=False)})"
                else:
                    preview = f"About to execute: {tc.name}({json.dumps(args, ensure_ascii=False)})"
                if tool.risk_level == "destructive":
                    preview = f"[DESTRUCTIVE] {preview}"

                self._pending_tool_call = (tc, tool, args)
                self._remaining_tool_calls = list(tool_calls[i + 1 :])
                self._react_messages = messages
                self._tool_trace.append(
                    {
                        "tool": tc.name,
                        "status": "waiting_for_approval",
                        "summary": preview[:240],
                    }
                )
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_APPROVAL,
                    raw_message=preview,
                    metadata={
                        "tool_trace": list(self._tool_trace),
                        "tool_calls_count": len(self._tool_trace),
                    },
                )

            try:
                result_text = await asyncio.wait_for(
                    tool.executor(args, self._build_tool_context()),
                    timeout=self.tool_timeout,
                )
                result_str = str(result_text)
                if len(result_str) > self.max_tool_result_chars:
                    result_str = result_str[: self.max_tool_result_chars] + "\n...[truncated]"
                self._tool_trace.append(
                    {
                        "tool": tc.name,
                        "status": "ok",
                        "summary": result_str[:240],
                    }
                )
            except asyncio.TimeoutError:
                logger.error(f"Tool {tc.name} timed out after {self.tool_timeout}s")
                result_str = f"Error: tool '{tc.name}' timed out after {self.tool_timeout}s"
                self._tool_trace.append(
                    {
                        "tool": tc.name,
                        "status": "error",
                        "summary": result_str[:240],
                    }
                )
            except Exception as e:
                logger.error(f"Tool {tc.name} failed: {e}", exc_info=True)
                result_str = f"Error executing {tc.name}: {e}"
                self._tool_trace.append(
                    {
                        "tool": tc.name,
                        "status": "error",
                        "summary": result_str[:240],
                    }
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                }
            )

        return None

    async def _parse_approval_with_llm(self, user_input: str) -> ApprovalResult:
        """Use LLM to classify user's approval intent in any language."""
        if not self.llm_client or not user_input.strip():
            return ApprovalResult.MODIFY
        try:
            response = await self.llm_client.chat_completion(messages=[{
                "role": "user",
                "content": (
                    f'The user was asked to approve an action. They replied: "{user_input}"\n'
                    "Classify their intent as exactly one word: APPROVE, REJECT, or MODIFY."
                ),
            }])
            result = (response.content or "").strip().upper()
            if "APPROVE" in result:
                return ApprovalResult.APPROVED
            if "REJECT" in result:
                return ApprovalResult.REJECTED
            return ApprovalResult.MODIFY
        except Exception as e:
            logger.warning(f"LLM approval parsing failed: {e}")
            return ApprovalResult.MODIFY

    async def _resume_after_approval(self) -> AgentResult:
        """Execute approved tool and continue mini ReAct loop."""
        if not self._pending_tool_call:
            return self.make_result(
                status=AgentStatus.ERROR,
                raw_message="No pending tool call to resume.",
                metadata={
                    "tool_trace": list(self._tool_trace),
                    "tool_calls_count": len(self._tool_trace),
                },
            )

        tc, tool, args = self._pending_tool_call
        self._pending_tool_call = None

        try:
            result_text = await asyncio.wait_for(
                tool.executor(args, self._build_tool_context()),
                timeout=self.tool_timeout,
            )
            result_str = str(result_text)
            if len(result_str) > self.max_tool_result_chars:
                result_str = result_str[: self.max_tool_result_chars] + "\n...[truncated]"
            self._tool_trace.append(
                {
                    "tool": tc.name,
                    "status": "ok",
                    "summary": result_str[:240],
                }
            )
        except asyncio.TimeoutError:
            logger.error(f"Approved tool {tc.name} timed out after {self.tool_timeout}s")
            result_str = f"Error: tool '{tc.name}' timed out after {self.tool_timeout}s"
            self._tool_trace.append(
                {
                    "tool": tc.name,
                    "status": "error",
                    "summary": result_str[:240],
                }
            )
        except Exception as e:
            logger.error(f"Approved tool {tc.name} failed: {e}", exc_info=True)
            result_str = f"Error executing {tc.name}: {e}"
            self._tool_trace.append(
                {
                    "tool": tc.name,
                    "status": "error",
                    "summary": result_str[:240],
                }
            )

        self._react_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            }
        )
        return await self._run_react()

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        """Heuristic: does the LLM response look like it's asking the user something?"""
        t = text.strip()
        if t.endswith("?") or t.endswith("\uff1f"):
            return True
        # Common question patterns in Chinese and English
        question_signals = [
            "\u8bf7\u63d0\u4f9b", "\u8bf7\u544a\u8bc9", "\u8bf7\u95ee",
            "\u80fd\u5426\u63d0\u4f9b", "\u9700\u8981\u4f60\u63d0\u4f9b",
            "what is", "what's", "could you", "can you", "please provide",
            "what email", "which email",
        ]
        t_lower = t.lower()
        return any(s in t_lower for s in question_signals)

    def _find_tool(self, name: str) -> Optional[AgentTool]:
        """Find an agent tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def _build_tool_context(self) -> AgentToolContext:
        """Create AgentToolContext from agent state."""
        return AgentToolContext(
            llm_client=self.llm_client,
            tenant_id=self.tenant_id,
            user_profile=self.context_hints.get("user_profile") if self.context_hints else None,
            context_hints=self.context_hints,
        )

    @staticmethod
    def _format_assistant_msg(response: LLMResponse) -> Dict[str, Any]:
        """Convert LLMResponse to OpenAI-format assistant message."""
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": response.content or None,
        }
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        if isinstance(tc.arguments, dict)
                        else tc.arguments,
                    },
                }
                for tc in response.tool_calls
            ]
        return msg

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

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, AsyncIterator, TYPE_CHECKING
from datetime import datetime
from uuid import uuid4
import logging
import re

from .base_agent import BaseAgent
from .message import Message
from .result import AgentResult, AgentStatus, ApprovalResult
from .protocols import LLMClientProtocol
from .streaming.models import StreamMode, EventType, AgentEvent
from .streaming.engine import StreamEngine
from .fields import InputField, OutputField

if TYPE_CHECKING:
    from .agents.decorator import InputSpec, OutputSpec

logger = logging.getLogger(__name__)


# ===== Field Definition =====

@dataclass
class RequiredField:
    """
    Defines a required field for an agent

    Attributes:
        name: Field name (e.g., "recipient", "subject")
        description: Human-readable description
        prompt: Question to ask user when field is missing
        validator: Optional validation function (returns bool)
        required: Whether this field is required (default: True)

    Example:
        RequiredField(
            name="email",
            description="Recipient email address",
            prompt="What email address should I send to?",
            validator=lambda v: "@" in v,  # Custom validator
            required=True
        )
    """
    name: str
    description: str
    prompt: str
    validator: Optional[Callable[[str], bool]] = None
    required: bool = True


@dataclass
class AgentState:
    """
    Complete agent state snapshot for serialization
    """
    agent_id: str
    agent_type: str
    tenant_id: str
    status: AgentStatus
    required_fields: List[RequiredField]
    collected_fields: Dict[str, Any]
    context_summary: str
    created_at: datetime
    last_active: datetime
    error_message: Optional[str] = None


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

        Default behavior: Extract fields from message and check completion.
        Override for custom field collection logic.
        """
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

        Default behavior: Parse approval response and act accordingly.
        Override for custom approval logic.
        """
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

        THIS IS WHERE YOU PUT YOUR BUSINESS LOGIC.

        Override this method to implement your agent's main functionality.

        Example:
            async def on_running(self, msg):
                name = self.collected_fields["name"]
                # Do something with the collected data
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Hello, {name}!"
                )
        """
        # Default implementation - just complete with collected fields
        # Subclasses MUST override this to provide meaningful raw_message
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
        import json
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
            import asyncio
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
        import asyncio

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

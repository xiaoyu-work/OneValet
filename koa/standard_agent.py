"""
Koa StandardAgent - State-driven agent with field collection

This is the core agent class that provides:
- State machine for conversation flow
- Required field collection pattern (via InputField)
- Approval workflow
- State handlers for each lifecycle phase
- Built-in streaming support

Example with InputField/OutputField (recommended):
    from koa import valet, StandardAgent, InputField, OutputField, AgentStatus

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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from .base_agent import BaseAgent
from .constants import COMPLETE_TASK_SCHEMA, COMPLETE_TASK_TOOL_NAME
from .fields import InputField
from .llm.base import LLMResponse
from .llm.base import ToolCall as LLMToolCall
from .media_dedup import append_unique_media
from .message import Message
from .models import AgentTool, AgentToolContext, RequiredField, ToolOutput
from .orchestrator.inbox import action_key, is_approval
from .protocols import LLMClientProtocol
from .result import AgentResult, AgentStatus, ApprovalResult
from .streaming.engine import StreamEngine
from .streaming.models import AgentEvent, EventType, StreamMode

if TYPE_CHECKING:
    from .agents.decorator import InputSpec, OutputSpec

logger = logging.getLogger(__name__)

#: What the Inbox says to do with an action on an unattended run.
APPROVAL_APPROVED = "approved"
APPROVAL_DECLINED = "declined"
APPROVAL_ASKED = "asked"
APPROVAL_UNAVAILABLE = "unavailable"


def _log_task_exception(task: asyncio.Task) -> None:
    """Log exceptions from fire-and-forget tasks instead of silently dropping them."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Background task failed: %s", exc, exc_info=exc)


# ===== State Transitions =====

# Valid state transitions
STATE_TRANSITIONS = {
    AgentStatus.INITIALIZING: [
        AgentStatus.RUNNING,
        AgentStatus.WAITING_FOR_INPUT,
        AgentStatus.WAITING_FOR_APPROVAL,
        AgentStatus.PAUSED,
        AgentStatus.COMPLETED,
        AgentStatus.ERROR,
    ],
    AgentStatus.RUNNING: [
        AgentStatus.COMPLETED,
        AgentStatus.ERROR,
        AgentStatus.PAUSED,
        AgentStatus.WAITING_FOR_INPUT,
        AgentStatus.WAITING_FOR_APPROVAL,
    ],
    AgentStatus.WAITING_FOR_INPUT: [
        AgentStatus.RUNNING,
        AgentStatus.WAITING_FOR_APPROVAL,
        AgentStatus.PAUSED,
        AgentStatus.COMPLETED,
        AgentStatus.ERROR,
        AgentStatus.WAITING_FOR_INPUT,
    ],
    AgentStatus.WAITING_FOR_APPROVAL: [
        AgentStatus.RUNNING,
        AgentStatus.WAITING_FOR_INPUT,
        AgentStatus.WAITING_FOR_APPROVAL,  # Allow re-approval after modification
        AgentStatus.PAUSED,
        AgentStatus.COMPLETED,
        AgentStatus.CANCELLED,
        AgentStatus.ERROR,
    ],
    AgentStatus.PAUSED: [
        AgentStatus.INITIALIZING,  # Resume to any previous state
        AgentStatus.RUNNING,
        AgentStatus.WAITING_FOR_INPUT,
        AgentStatus.WAITING_FOR_APPROVAL,
        AgentStatus.CANCELLED,
        AgentStatus.ERROR,
    ],
    AgentStatus.COMPLETED: [],  # Terminal state
    AgentStatus.ERROR: [AgentStatus.CANCELLED],
    AgentStatus.CANCELLED: [],  # Terminal state
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

    # Agent ReAct loop configuration (active when tools is non-empty)
    domain_system_prompt: str = ""
    tools: tuple = ()
    max_turns: int = 15
    tool_timeout: float = 30.0  # seconds per tool call
    max_tool_result_chars: int = 4000  # truncate tool results beyond this

    _COMPLETE_TASK_INSTRUCTION = (
        "\n\nIMPORTANT: When you have finished the task, reply with your final answer "
        "as plain text and no tool calls. That ends the turn."
    )

    def __init__(
        self,
        tenant_id: str = "",
        llm_client: Optional[LLMClientProtocol] = None,
        orchestrator_callback: Optional[Callable] = None,
        context_hints: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize StandardAgent

        Args:
            tenant_id: Tenant identifier for multi-tenant isolation (default: "default")
            llm_client: LLM client (usually auto-injected by registry)
            orchestrator_callback: Callback function for events
            context_hints: Pre-extracted fields from orchestrator
        """
        super().__init__(name=kwargs.get("name"))

        # Core attributes
        self.tenant_id = tenant_id
        self.agent_id = self._generate_agent_id()
        self.llm_client = llm_client
        self.orchestrator_callback = orchestrator_callback
        self.execution_policy = kwargs.get("execution_policy")

        # State management
        self.status = AgentStatus.INITIALIZING
        self.collected_fields: Dict[str, Any] = {}
        self._output_values: Dict[str, Any] = {}  # Store output values
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.error_message: Optional[str] = None

        self._tool_validator: Optional[Any] = None

        # Instance metadata - for custom per-instance properties (e.g., user_id, session_id)
        self.metadata: Dict[str, Any] = {}

        # Pause management
        self._pause_requested = False
        self._status_before_pause: Optional[AgentStatus] = None

        # Free-form runtime state a subclass can use across state handlers
        self.execution_state: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        self._message_history: List["Message"] = []

        # Context hints from orchestrator
        self.context_hints = context_hints or {}

        # Recalled memories from orchestrator (when enable_memory=true)
        self._recalled_memories: List[Dict[str, Any]] = []

        # Agent ReAct loop state (only active when tools is non-empty)
        self._react_messages: List[Dict[str, Any]] = []
        self._react_turn: int = 0
        self._pending_tool_call: Optional[Tuple[LLMToolCall, AgentTool, Dict[str, Any]]] = None
        self._remaining_tool_calls: List[LLMToolCall] = []
        self._tool_trace: List[Dict[str, Any]] = []
        self._collected_media: List[Dict[str, Any]] = []  # media from ToolOutput results

        # Built-in streaming engine
        self._stream_engine = StreamEngine(
            agent_id=self.agent_id, agent_type=self.__class__.__name__
        )
        self._streaming_enabled = False

        logger.debug(
            f"Initialized {self.__class__.__name__} (ID: {self.agent_id}, Tenant: {tenant_id})"
        )

    def _user_now(self) -> tuple:
        """Return (datetime, tz_name) in user's timezone from context_hints.
        Falls back to UTC if timezone not available."""
        tz_str = self.context_hints.get("timezone", "")
        if tz_str and tz_str != "UTC":
            try:
                from zoneinfo import ZoneInfo

                tz = ZoneInfo(tz_str)
                return datetime.now(tz), tz_str
            except Exception:
                pass
        return datetime.now(timezone.utc), "UTC"

    # ===== Agent ReAct Support =====

    def get_system_prompt(self) -> str:
        """Return the system prompt for the mini ReAct loop.

        Injects user context (True Memory, User Profile) from the orchestrator
        handoff so domain agents can personalize task execution.

        Note: Personality (AI-to-user tone) is NOT injected here — it only
        controls how the orchestrator talks to the user, not how subagents
        execute tasks. User communication preferences (e.g. "formal tone for
        work emails") are conveyed via True Memory feedback entries instead.

        Override in subclasses to customize. Only used when tools is non-empty.
        """
        parts = [self.domain_system_prompt]

        user_context = self._build_user_context_section()
        if user_context:
            parts.append(user_context)

        parts.append(self._COMPLETE_TASK_INSTRUCTION)
        return "\n\n".join(parts)

    def _build_user_context_section(self) -> str:
        """Build user context section from orchestrator-passed data.

        Includes True Memory (canonical facts + behavioral feedback) and
        basic User Profile identity. Does NOT include Personality — that
        controls AI-to-user tone at the orchestrator level only.
        """
        sections: List[str] = []

        # True Memory (canonical facts + behavioral feedback)
        true_memory = self.context_hints.get("true_memory")
        if true_memory and isinstance(true_memory, list):
            lines = []
            for fact in true_memory[:10]:
                if not isinstance(fact, dict):
                    continue
                summary = str(fact.get("summary") or "").strip()
                if not summary:
                    continue
                line = f"- {summary}"
                why = str(fact.get("why") or "").strip()
                how = str(fact.get("how_to_apply") or "").strip()
                if why:
                    line += f" Why: {why}"
                if how:
                    line += f" Apply: {how}"
                lines.append(line)
            if lines:
                sections.append("[True Memory]\n" + "\n".join(lines))

        # User Profile (basic identity only — full profile stays at orchestrator)
        user_profile = self.context_hints.get("user_profile")
        if user_profile and isinstance(user_profile, dict):
            profile_lines = []
            identity = user_profile.get("identity") or {}
            if identity.get("full_name"):
                profile_lines.append(f"Name: {identity['full_name']}")
            if identity.get("birthday"):
                profile_lines.append(f"Birthday: {identity['birthday']}")
            if profile_lines:
                sections.append("[User Profile]\n" + "\n".join(profile_lines))

        return "\n\n".join(sections)

    # ===== Required Methods (Must Override) =====

    # ===== State Handlers (Override to customize) =====

    async def on_initializing(self, msg: Message) -> AgentResult:
        """
        Called when agent first receives a message.

        Default behavior: run the agent's ReAct loop. Override for custom
        initialization logic.
        """
        if self.needs_approval():
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL, raw_message=self.get_approval_prompt()
            )

        self.transition_to(AgentStatus.RUNNING)
        return await self.on_running(msg)

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        """
        Called when waiting for the user to answer a question the agent asked.

        Resumes the ReAct loop with the user's reply so the model can continue
        from where it paused.
        """
        user_text = msg.get_text() if msg else ""
        if not user_text:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="Please provide the requested information.",
            )

        if self.tools and self._react_messages:
            self._react_messages.append({"role": "user", "content": user_text})
            self.transition_to(AgentStatus.RUNNING)
            return await self._run_react()

        if self.needs_approval():
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL, raw_message=self.get_approval_prompt()
            )

        self.transition_to(AgentStatus.RUNNING)
        return await self.on_running(msg)

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        """
        Called when waiting for user approval.

        If a domain tool call is pending, uses LLM-based approval parsing.
        Otherwise, uses the InputField-based approval flow.
        """
        if self._pending_tool_call:
            # Agent ReAct path: LLM-based approval parsing
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
            self._tool_trace.append(
                {
                    "tool": "approval",
                    "status": "modified",
                    "summary": f"User requested modification: {user_input[:180]}",
                }
            )
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
            # The user wants something changed; re-run the loop with their words
            # so the model can adjust its tool call.
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

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
            # Agent ReAct path
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
            self._collected_media = []
            return await self._run_react()

        # Default for non-domain subclasses (they override this)
        return self.make_result(status=AgentStatus.COMPLETED)

    async def on_error(self, msg: Message) -> AgentResult:
        """
        Called when agent is in error state.

        Override to implement error recovery logic.
        """
        return self.make_result(status=AgentStatus.ERROR, error_message=self.error_message)

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
            AgentStatus.INITIALIZING,
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
                raw_message="Please provide the requested information.",
            )
        elif previous_status == AgentStatus.WAITING_FOR_APPROVAL:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL, raw_message=self.get_approval_prompt()
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
        **kwargs,
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
            **kwargs,
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

    def get_state_summary(self) -> Dict[str, Any]:
        """Get standardized state summary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "collected_fields": dict(self.collected_fields),
            "last_active": self.last_active.isoformat(),
            "error_message": self.error_message,
        }

    def is_completed(self) -> bool:
        """Check if agent has completed its task."""
        return self.status == AgentStatus.COMPLETED

    def get_message_history(self) -> List["Message"]:
        """Get a copy of this agent's message history."""
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
                task = loop.create_task(
                    self._stream_engine.emit_state_change(old_status.value, new_status.value)
                )
                task.add_done_callback(_log_task_exception)
            except RuntimeError:
                pass  # No running loop

        return True

    # ===== Streaming Support =====

    async def stream(
        self, msg: Message = None, mode: StreamMode = StreamMode.EVENTS
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
                            },
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
            },
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
                for part in some_long_answer:
                    await self.emit_message_chunk(part)
                return self.make_result(...)
        """
        if self._streaming_enabled:
            await self._stream_engine.emit_message_chunk(chunk)

    async def emit_tool_call(
        self, tool_name: str, tool_input: Dict[str, Any], call_id: Optional[str] = None
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
        call_id: Optional[str] = None,
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
            await self._stream_engine.emit_tool_result(tool_name, result, success, error, call_id)

    async def emit_progress(self, current: int, total: int, message: Optional[str] = None) -> None:
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

    @property
    def session_working_memory(self) -> Dict[str, Any]:
        """Structured session notes passed in from the orchestrator."""
        if not self.context_hints:
            return {}
        return dict(self.context_hints.get("session_working_memory") or {})

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
        if self.llm_client is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has tools but no llm_client; "
                "it cannot run its ReAct loop."
            )
        llm = self.llm_client
        tool_schemas = [t.to_openai_schema() for t in self.tools]
        messages = self._react_messages

        # Anything the user approved while this run was away happens now, from
        # the arguments they saw -- before the model gets a turn and could
        # propose something else.
        for note in await self._carry_out_approved_actions():
            messages.append({"role": "user", "content": note})

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
            response: LLMResponse = await llm.chat_completion(
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

                # No tool calls → the agent is done. Its text is the answer.
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=text,
                    metadata={
                        "tool_trace": list(self._tool_trace),
                        "tool_calls_count": len(self._tool_trace),
                    },
                )

            messages.append(self._format_assistant_msg(response))
            result = await self._execute_tool_calls(response.tool_calls, messages)
            if result is not None:
                return result

        # Exhausted max_turns — ask LLM to summarize whatever data it
        # collected so far instead of returning a generic failure.
        logger.warning(
            f"[{self.__class__.__name__}:{self.name}] exhausted {self.max_turns} turns, "
            f"asking LLM to summarize partial results"
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have run out of allowed steps. Summarize whatever information "
                    "you have gathered so far and present it to the user. "
                    "If some tool calls failed, briefly note what didn't work. "
                    "Do NOT say you failed — give the user what you have."
                ),
            }
        )
        try:
            summary_resp = await llm.chat_completion(
                messages=messages,
                tools=None,
            )
            summary_msg = summary_resp.content or (
                "I wasn't able to complete the task within the allowed steps. "
                "Please try again with more specific information."
            )
        except Exception as e:
            logger.warning(
                f"[{self.__class__.__name__}:{self.name}] summary after max_turns failed: {e}"
            )
            summary_msg = (
                "I wasn't able to complete the task within the allowed steps. "
                "Please try again with more specific information."
            )

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=summary_msg,
            metadata={
                "tool_trace": list(self._tool_trace),
                "tool_calls_count": len(self._tool_trace),
                "partial_result": True,
                "media": self._collected_media or None,
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
            if tc.name == COMPLETE_TASK_TOOL_NAME:
                try:
                    args = (
                        tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
                    )
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result_text = args.get("result", "")
                if result_text:
                    self._tool_trace.append(
                        {
                            "tool": COMPLETE_TASK_TOOL_NAME,
                            "status": "ok",
                            "summary": result_text[:240],
                        }
                    )
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
                            "media": self._collected_media or None,
                        },
                    )
                else:
                    # Missing result — append error and continue
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": 'Error: "result" argument is required for complete_task.',
                        }
                    )
                    continue

            tool = self._find_tool(tc.name)
            if tool is None:
                error_text = f"Error: Unknown tool '{tc.name}'"
                self._tool_trace.append({"tool": tc.name, "status": "error", "summary": error_text})
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

            # Validate tool arguments against declared JSON schema (P0-5).
            # The tool_manager / agent-tool layer defines `input_schema`; we
            # lazily instantiate a validator so hot-path cost is ~zero when
            # tools omit schemas.
            validation_error: Optional[str] = None
            try:
                from koa.llm.tool_validator import ToolSchemaValidator

                if self._tool_validator is None:
                    self._tool_validator = ToolSchemaValidator.from_agent_tools(self.tools)
                vr = self._tool_validator.validate(tc.name, args)
                if not vr.ok:
                    details = vr.details or {}
                    msg = details.get("message") or vr.reason
                    path = details.get("path")
                    validation_error = f"{msg}" + (f" at {path}" if path else "")
            except Exception as exc:
                logger.debug("Tool-arg validator init failed, skipping: %s", exc)

            if validation_error is not None:
                error_text = (
                    f"Error: Invalid arguments for tool '{tc.name}': "
                    f"{validation_error}. Please retry with arguments matching the schema."
                )
                self._tool_trace.append(
                    {"tool": tc.name, "status": "invalid_args", "summary": error_text[:240]}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": error_text,
                    }
                )
                continue

            policy_decision = self._evaluate_tool_policy(tool, args)
            if policy_decision is not None and not policy_decision.allowed:
                error_text = f"Permission denied for tool '{tc.name}': {policy_decision.reason}"
                self._tool_trace.append(
                    {"tool": tc.name, "status": "denied", "summary": error_text[:240]}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": error_text,
                    }
                )
                continue

            requires_approval = (
                tool.needs_approval
                or tool.risk_level in ("write", "destructive")
                or bool(policy_decision and policy_decision.require_approval)
            )
            if requires_approval and not self._is_attended():
                # Nobody is watching this run, so pausing inline would strand
                # it. The Inbox carries the question instead: the user answers
                # from wherever they are, and a later run finds their answer
                # here rather than asking again.
                decision = await self._ask_inbox_for_approval(tc, tool, args)

                if decision == APPROVAL_APPROVED:
                    # They already said yes. Fall through and do it -- this is
                    # the run their approval was waiting for.
                    requires_approval = False
                else:
                    if decision == APPROVAL_ASKED:
                        error_text = (
                            f"'{tc.name}' needs approval. I've asked the user and will "
                            "continue once they answer. Do not retry it in this run."
                        )
                        status = "asked"
                    elif decision == APPROVAL_DECLINED:
                        error_text = (
                            f"The user declined '{tc.name}'. Do not run it; "
                            "continue without it and say so."
                        )
                        status = "declined"
                    else:
                        error_text = (
                            f"'{tc.name}' needs the user's approval, and this run is "
                            "unattended (scheduled job or trigger), so nobody can give it. "
                            "Skip this action and say what you would have done."
                        )
                        status = "needs_approval"
                    logger.info(
                        f"[{self.__class__.__name__}:{self.name}] {tc.name}: "
                        f"approval required on an unattended run ({status})"
                    )
                    self._tool_trace.append(
                        {"tool": tc.name, "status": status, "summary": error_text[:240]}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": error_text,
                        }
                    )
                    continue

            if requires_approval:
                if tool.get_preview:
                    try:
                        preview = await tool.get_preview(args, self._build_tool_context())
                    except Exception as e:
                        logger.error(f"Preview generation failed for {tc.name}: {e}")
                        preview = (
                            f"About to execute: {tc.name}({json.dumps(args, ensure_ascii=False)})"
                        )
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
                tool_result = await asyncio.wait_for(
                    tool.executor(args, self._build_tool_context()),
                    timeout=self.tool_timeout,
                )
                # Extract media from ToolOutput before converting to string
                if isinstance(tool_result, ToolOutput):
                    result_str = tool_result.text
                    if tool_result.media:
                        append_unique_media(self._collected_media, tool_result.media)
                else:
                    result_str = str(tool_result)
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
            response = await self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f'The user was asked to approve an action. They replied: "{user_input}"\n'
                            "Classify their intent as exactly one word: APPROVE, REJECT, or MODIFY."
                        ),
                    }
                ]
            )
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
            "\u8bf7\u63d0\u4f9b",
            "\u8bf7\u544a\u8bc9",
            "\u8bf7\u95ee",
            "\u80fd\u5426\u63d0\u4f9b",
            "\u9700\u8981\u4f60\u63d0\u4f9b",
            "what is",
            "what's",
            "could you",
            "can you",
            "please provide",
            "what email",
            "which email",
        ]
        t_lower = t.lower()
        return any(s in t_lower for s in question_signals)

    async def _carry_out_approved_actions(self) -> List[str]:
        """Perform what the user approved while this run was away.

        An approval is a contract about one specific action -- this tool,
        these arguments -- and the arguments are the part that matters. The
        earlier attempt at this run recorded them; this does not ask the model
        to produce them again. It could not safely: a model rewriting an email
        body between runs would either be sending something the user never
        saw, or, if we insisted the text match, be asked to approve the same
        message forever.

        So the stored arguments are executed verbatim. ``claim_execution`` is
        a compare-and-swap, so an approval is honoured once even if two
        instances resume the same run together, and a claim is only handed
        back when the action never reached the tool.

        Returns a line per action for the model to read.
        """
        hints = self.context_hints or {}
        inbox = hints.get("inbox")
        run_id = hints.get("run_id")
        if inbox is None or not run_id or not getattr(inbox, "enabled", False):
            return []

        try:
            asks = await inbox.for_run(run_id)
        except Exception as e:
            logger.warning(f"Could not read approved actions for run {run_id}: {e}")
            return []

        notes: List[str] = []
        for ask in asks:
            if not ask.awaits_execution or ask.kind != "approval":
                continue
            data = ask.data or {}
            tool_name = data.get("tool")
            if not is_approval(ask.resolution):
                # Declined. Stamp it so a later resume does not reconsider,
                # and tell the model so it can say what it did not do.
                if await inbox.claim_execution(ask.id):
                    notes.append(f"The user declined '{tool_name}'. It was not run.")
                continue

            tool = self._find_tool(tool_name) if tool_name else None
            if tool is None:
                logger.warning(f"Approved tool {tool_name!r} is not available on {self.name}")
                continue
            if not await inbox.claim_execution(ask.id):
                logger.info(f"[Inbox] Ask {ask.id} already carried out elsewhere")
                continue

            note = await self._run_approved_tool(inbox, ask, tool, data.get("args") or {})
            notes.append(note)
        return notes

    async def _run_approved_tool(self, inbox: Any, ask: Any, tool: Any, args: Dict[str, Any]) -> str:
        """Execute one approved action and describe the outcome."""
        name = tool.name
        try:
            result = await asyncio.wait_for(
                tool.executor(args, self._build_tool_context()),
                timeout=self.tool_timeout,
            )
        except asyncio.TimeoutError:
            # Never reached the point of doing anything observable, so the
            # claim goes back and a later resume may try again.
            await inbox.release_execution(ask.id)
            logger.warning(f"Approved action {name} timed out")
            return f"'{name}' was approved but timed out. It has not been done."
        except Exception as e:
            logger.error(f"Approved action {name} failed: {e}", exc_info=True)
            return f"'{name}' was approved but failed: {e}"

        text = result.text if isinstance(result, ToolOutput) else str(result)
        if isinstance(result, ToolOutput) and result.media:
            append_unique_media(self._collected_media, result.media)
        if len(text) > self.max_tool_result_chars:
            text = text[: self.max_tool_result_chars] + "\n...[truncated]"
        self._tool_trace.append({"tool": name, "status": "approved_and_run", "summary": text[:240]})
        logger.info(f"[{self.__class__.__name__}:{self.name}] carried out approved {name}")
        return f"The user approved '{name}' and it has now been done. Result: {text}"

    async def _ask_inbox_for_approval(self, tc, tool, args: Dict[str, Any]) -> str:
        """Consult the Inbox about this action. Returns what to do with it.

        An approval outlives the run that requested it. The run that asks
        finishes without acting; the user answers minutes or hours later; a
        resumed run replays the same reasoning and arrives back here. So the
        question this method answers is not only "shall I ask" but "have they
        already told me" -- and the answer has to be found again in a process
        that never saw the original request.

        That rules out matching on tool_call_id: those are minted fresh by the
        model on every run and mean nothing outside the loop that produced
        them. A decision is looked up by what it was a decision *about* --
        this run, this tool, these arguments -- which is stable across replays.

        Returns one of APPROVAL_APPROVED, APPROVAL_DECLINED, APPROVAL_ASKED,
        or APPROVAL_UNAVAILABLE.
        """
        hints = self.context_hints or {}
        inbox = hints.get("inbox")
        run_id = hints.get("run_id")
        if inbox is None or not run_id or not getattr(inbox, "enabled", False):
            return APPROVAL_UNAVAILABLE

        prior = await self._prior_inbox_decision(inbox, run_id, tc.name, args)
        if prior is not None:
            if prior == "pending":
                # Already asked on an earlier attempt at this run. Asking again
                # would put a second copy of the same question in front of the
                # user, so wait for the one that is already out there.
                return APPROVAL_ASKED
            approved = is_approval(prior)
            logger.info(
                f"[{self.__class__.__name__}:{self.name}] {tc.name}: "
                f"user already {'approved' if approved else 'declined'} this"
            )
            return APPROVAL_APPROVED if approved else APPROVAL_DECLINED

        preview = await self._approval_preview(tc, tool, args)
        try:
            ask = await inbox.create(
                tenant_id=self.tenant_id,
                run_id=run_id,
                tool_call_id=tc.id,
                kind="approval",
                action_key=action_key(tc.name, args),
                title=f"Approve {tc.name}?",
                body=preview,
                options=["approve", "reject"],
                data={"tool": tc.name, "args": args, "agent": self.__class__.__name__},
            )
        except Exception as e:
            logger.error(f"Could not record approval ask for {tc.name}: {e}")
            return APPROVAL_UNAVAILABLE
        if ask is None:
            return APPROVAL_UNAVAILABLE
        # A conflict returns the row that already existed, which may carry an
        # answer given while this run was away.
        if not ask.is_open:
            return APPROVAL_APPROVED if is_approval(ask.resolution) else APPROVAL_DECLINED

        # Deliver it to wherever the user is. Best-effort: the ask is durable,
        # so a failed notification means they find it in the app instead.
        mirror = hints.get("ask_mirror")
        if mirror is not None and getattr(mirror, "enabled", False) and ask.is_open:
            try:
                await mirror.mirror(ask)
            except Exception as e:
                logger.warning(f"Could not mirror ask {ask.id}: {e}")
        return APPROVAL_ASKED

    async def _prior_inbox_decision(
        self, inbox: Any, run_id: str, tool_name: str, args: Dict[str, Any]
    ) -> Optional[str]:
        """What the user already said about this exact action on this run.

        ``"pending"`` when the question is out but unanswered, the resolution
        text when answered, and None when it has never been asked.
        """
        wanted = action_key(tool_name, args)
        try:
            asks = await inbox.for_run(run_id)
        except Exception as e:
            # Treating an unreadable Inbox as "never asked" would ask again and
            # could act twice; treating it as pending stalls safely instead.
            logger.warning(f"Could not read prior asks for run {run_id}: {e}")
            return "pending"
        for ask in asks:
            data = ask.data or {}
            if action_key(data.get("tool"), data.get("args")) != wanted:
                continue
            if ask.is_open:
                return "pending"
            return (ask.resolution or "").strip().lower()
        return None

    async def _approval_preview(self, tc, tool, args: Dict[str, Any]) -> str:
        """What the user actually sees on their phone.

        Generated the same way as for an inline prompt, so it names the real
        file, recipient, or amount rather than just the tool.
        """
        if tool.get_preview:
            try:
                preview = await tool.get_preview(args, self._build_tool_context())
            except Exception as e:
                logger.error(f"Preview generation failed for {tc.name}: {e}")
                preview = f"Run {tc.name}?"
        else:
            preview = f"Run {tc.name}?"
        if tool.risk_level == "destructive":
            preview = f"[DESTRUCTIVE] {preview}"
        return preview

    def _is_attended(self) -> bool:
        """Whether a human can answer if this agent pauses for approval.

        Defaults to True when the orchestrator did not say, so an unknown
        surface keeps asking rather than acting unapproved.
        """
        if not self.context_hints:
            return True
        return bool(self.context_hints.get("attended", True))

    def _find_tool(self, name: str) -> Optional[AgentTool]:
        """Find an agent tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def _build_tool_context(self) -> AgentToolContext:
        """Create AgentToolContext from agent state."""
        metadata: Dict[str, Any] = {}
        if self.context_hints:
            if self.context_hints.get("timezone"):
                metadata["timezone"] = self.context_hints["timezone"]
            if self.context_hints.get("user_location"):
                metadata["location"] = self.context_hints["user_location"]
            if self.context_hints.get("session_id"):
                metadata["session_id"] = self.context_hints["session_id"]
            if self.context_hints.get("permissions"):
                metadata["permissions"] = self.context_hints["permissions"]
        return AgentToolContext(
            llm_client=self.llm_client,
            tenant_id=self.tenant_id,
            user_profile=self.context_hints.get("user_profile") if self.context_hints else None,
            context_hints=self.context_hints,
            metadata=metadata,
        )

    def _evaluate_tool_policy(self, tool: AgentTool, args: Dict[str, Any]):
        """Evaluate runtime execution policy for a tool call if configured."""
        if not self.execution_policy:
            return None
        return self.execution_policy.evaluate(
            tool,
            tenant_id=self.tenant_id,
            args=args,
            metadata=self._build_tool_context().metadata,
            request_context=self.context_hints,
            agent_type=self.agent_type,
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
                        "arguments": (
                            json.dumps(tc.arguments, ensure_ascii=False)
                            if isinstance(tc.arguments, dict)
                            else tc.arguments
                        ),
                    },
                }
                for tc in response.tool_calls
            ]
        return msg

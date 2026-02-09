"""
OneValet MsgHub - Multi-agent message sharing hub

This module provides:
- MsgHub: Main class for multi-agent conversation sharing
- Support for sequential, parallel, and selective visibility modes
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Protocol, AsyncIterator, Union
from contextlib import asynccontextmanager

from .models import (
    Message,
    MessageRole,
    MessageType,
    ParticipantInfo,
    MsgHubConfig,
    MsgHubState,
    SharedContext,
    VisibilityMode,
    HubExecutionResult,
)


class AgentProtocol(Protocol):
    """Protocol for agents that can participate in MsgHub"""
    agent_id: str
    agent_type: str

    async def reply(self, message: Any) -> Any: ...


class MsgHubError(Exception):
    """Raised when MsgHub operation fails"""
    pass


class MsgHub:
    """
    Multi-agent message sharing hub.

    Allows multiple agents to share conversation context.
    Supports different visibility modes for message sharing.

    Example (sequential execution with shared context):
        async with MsgHub(participants=[researcher, writer, reviewer]) as hub:
            # Researcher finds information
            result1 = await hub.execute(researcher, "Research Python frameworks")

            # Writer can SEE researcher's messages
            result2 = await hub.execute(writer, "Write a comparison")

            # Reviewer can SEE both previous messages
            result3 = await hub.execute(reviewer, "Review the document")

    Example (configuration-based):
        hub = MsgHub(
            config=MsgHubConfig(
                visibility_mode=VisibilityMode.SEQUENTIAL,
                shared_context_keys=["research_findings", "draft"]
            )
        )
        async with hub:
            await hub.add_participant(researcher)
            result = await hub.execute(researcher, "Research topic")
    """

    def __init__(
        self,
        participants: Optional[List[AgentProtocol]] = None,
        config: Optional[MsgHubConfig] = None,
        hub_id: Optional[str] = None,
    ):
        """
        Initialize MsgHub.

        Args:
            participants: Initial list of participant agents
            config: Hub configuration
            hub_id: Optional hub identifier
        """
        self.config = config or MsgHubConfig()
        self.config.hub_id = hub_id or self.config.hub_id or f"hub_{uuid.uuid4().hex[:8]}"

        self.state = MsgHubState(
            hub_id=self.config.hub_id,
            context=SharedContext()
        )

        # Message callbacks
        self._message_callbacks: List[Callable[[Message], Any]] = []

        # Initial participants
        self._initial_participants = participants or []

    async def __aenter__(self) -> "MsgHub":
        """Async context manager entry"""
        # Add initial participants
        for agent in self._initial_participants:
            await self.add_participant(agent)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        await self.close()

    async def add_participant(
        self,
        agent: AgentProtocol,
        can_see_all: bool = True,
        visible_roles: Optional[List[MessageRole]] = None
    ) -> ParticipantInfo:
        """
        Add an agent as a participant.

        Args:
            agent: Agent to add
            can_see_all: Whether agent can see messages from before it joined
            visible_roles: Which message roles are visible (for SELECTIVE mode)

        Returns:
            ParticipantInfo for the added agent
        """
        if not self.state.is_active:
            raise MsgHubError("Cannot add participant to closed hub")

        agent_id = getattr(agent, 'agent_id', str(id(agent)))
        agent_type = getattr(agent, 'agent_type', type(agent).__name__)

        participant = ParticipantInfo(
            agent_id=agent_id,
            agent_type=agent_type,
            joined_at_message_count=len(self.state.messages),
            can_see_all=can_see_all,
            visible_roles=visible_roles or list(MessageRole)
        )

        self.state.participants[agent_id] = participant
        return participant

    async def remove_participant(self, agent_id: str) -> bool:
        """
        Remove a participant from the hub.

        Args:
            agent_id: ID of agent to remove

        Returns:
            True if removed, False if not found
        """
        if agent_id in self.state.participants:
            self.state.participants[agent_id].is_active = False
            return True
        return False

    async def broadcast(
        self,
        content: str,
        sender_id: str,
        sender_type: Optional[str] = None,
        role: MessageRole = MessageRole.AGENT,
        message_type: MessageType = MessageType.TEXT,
        data: Optional[Dict[str, Any]] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Broadcast a message to all participants.

        Args:
            content: Message content
            sender_id: ID of sender
            sender_type: Type of sender
            role: Message role
            message_type: Type of message
            data: Structured data
            reply_to: ID of message being replied to
            metadata: Additional metadata

        Returns:
            The broadcast message
        """
        if not self.state.is_active:
            raise MsgHubError("Cannot broadcast to closed hub")

        message = Message(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            role=role,
            content=content,
            sender_id=sender_id,
            sender_type=sender_type,
            message_type=message_type,
            data=data or {},
            reply_to=reply_to,
            metadata=metadata or {},
        )

        # Check message limit
        if len(self.state.messages) >= self.config.max_messages:
            # Remove oldest message
            self.state.messages.pop(0)

        self.state.messages.append(message)

        # Update participant stats
        if sender_id in self.state.participants:
            self.state.participants[sender_id].messages_sent += 1

        # Notify callbacks
        for callback in self._message_callbacks:
            try:
                result = callback(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass  # Ignore callback errors

        return message

    async def broadcast_user_message(
        self,
        content: str,
        user_id: str = "user",
        data: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Convenience method to broadcast a user message"""
        return await self.broadcast(
            content=content,
            sender_id=user_id,
            role=MessageRole.USER,
            message_type=MessageType.TEXT,
            data=data
        )

    async def broadcast_system_message(
        self,
        content: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Convenience method to broadcast a system message"""
        return await self.broadcast(
            content=content,
            sender_id="system",
            role=MessageRole.SYSTEM,
            message_type=MessageType.TEXT,
            data=data
        )

    def get_messages(
        self,
        participant_id: Optional[str] = None,
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
        role: Optional[MessageRole] = None
    ) -> List[Message]:
        """
        Get messages from the hub.

        Args:
            participant_id: If provided, filter by visibility for this participant
            limit: Maximum number of messages to return
            since: Only return messages after this time
            role: Filter by message role

        Returns:
            List of messages
        """
        if participant_id:
            messages = self.state.get_messages_for_participant(
                participant_id, self.config
            )
        else:
            messages = list(self.state.messages)

        # Filter by time
        if since:
            messages = [m for m in messages if m.timestamp >= since]

        # Filter by role
        if role:
            messages = [m for m in messages if m.role == role]

        # Apply limit
        if limit:
            messages = messages[-limit:]

        return messages

    def get_context(self, key: Optional[str] = None) -> Any:
        """
        Get shared context.

        Args:
            key: Specific key to get, or None for all context

        Returns:
            Context value or full context dict
        """
        if key:
            return self.state.context.get(key)
        return dict(self.state.context.data)

    def set_context(
        self,
        key: str,
        value: Any,
        updater_id: Optional[str] = None
    ) -> None:
        """
        Set a shared context value.

        Args:
            key: Context key
            value: Context value
            updater_id: ID of the updater
        """
        self.state.context.set(key, value, updater_id)

    def update_context(
        self,
        updates: Dict[str, Any],
        updater_id: Optional[str] = None
    ) -> None:
        """
        Update multiple context values.

        Args:
            updates: Dictionary of updates
            updater_id: ID of the updater
        """
        self.state.context.update(updates, updater_id)

    async def execute(
        self,
        agent: AgentProtocol,
        message: Any,
        broadcast_input: bool = True,
        broadcast_output: bool = True,
        update_context_keys: Optional[List[str]] = None
    ) -> Any:
        """
        Execute an agent with shared context.

        Args:
            agent: Agent to execute
            message: Message to send to agent
            broadcast_input: Whether to broadcast the input message
            broadcast_output: Whether to broadcast the agent's reply
            update_context_keys: Keys to extract from agent and add to context

        Returns:
            Agent's reply
        """
        agent_id = getattr(agent, 'agent_id', str(id(agent)))
        agent_type = getattr(agent, 'agent_type', type(agent).__name__)

        # Ensure agent is a participant
        if agent_id not in self.state.participants:
            await self.add_participant(agent)

        # Broadcast input if requested
        if broadcast_input and isinstance(message, str):
            await self.broadcast_user_message(message)

        # Inject shared context into agent if it has collected_fields
        if hasattr(agent, 'collected_fields') and isinstance(agent.collected_fields, dict):
            # Add visible messages as context
            visible_messages = self.get_messages(participant_id=agent_id)
            agent.collected_fields['_hub_messages'] = [m.to_dict() for m in visible_messages]
            agent.collected_fields['_hub_context'] = dict(self.state.context.data)

        # Execute agent
        reply = await agent.reply(message)

        # Extract and update context
        if update_context_keys and hasattr(agent, 'collected_fields'):
            for key in update_context_keys:
                if key in agent.collected_fields:
                    self.set_context(key, agent.collected_fields[key], agent_id)

        # Broadcast output if requested
        if broadcast_output:
            reply_content = getattr(reply, 'raw_message', str(reply)) if reply else ""
            reply_data = getattr(reply, 'data', None)
            if isinstance(reply_data, dict):
                data = reply_data
            else:
                data = {"result": reply_data} if reply_data else {}

            await self.broadcast(
                content=reply_content,
                sender_id=agent_id,
                sender_type=agent_type,
                role=MessageRole.AGENT,
                message_type=MessageType.RESULT,
                data=data
            )

        # Update last seen message
        if self.state.messages:
            self.state.participants[agent_id].last_seen_message_id = self.state.messages[-1].id

        return reply

    async def execute_sequential(
        self,
        agents: List[AgentProtocol],
        initial_message: Any,
        broadcast_all: bool = True
    ) -> HubExecutionResult:
        """
        Execute agents sequentially, each seeing previous agents' messages.

        Args:
            agents: Agents to execute in order
            initial_message: Initial message to first agent
            broadcast_all: Whether to broadcast all messages

        Returns:
            HubExecutionResult with all agent results
        """
        result = HubExecutionResult(
            hub_id=self.state.hub_id,
            status="completed",
            started_at=datetime.now()
        )

        current_message = initial_message

        for agent in agents:
            agent_id = getattr(agent, 'agent_id', str(id(agent)))
            result.total_agents += 1

            try:
                reply = await self.execute(
                    agent,
                    current_message,
                    broadcast_input=(agent == agents[0]),  # Only broadcast first input
                    broadcast_output=broadcast_all
                )

                result.agent_results.append({
                    "agent_id": agent_id,
                    "status": "completed",
                    "output": getattr(reply, 'data', reply)
                })
                result.completed_agents += 1

                # Use reply as next message
                if hasattr(reply, 'raw_message'):
                    current_message = reply.raw_message
                elif isinstance(reply, str):
                    current_message = reply

            except Exception as e:
                result.agent_results.append({
                    "agent_id": agent_id,
                    "status": "failed",
                    "error": str(e)
                })
                result.failed_agents += 1
                result.errors.append(f"{agent_id}: {str(e)}")

        # Set final status
        if result.failed_agents > 0:
            if result.completed_agents > 0:
                result.status = "partial"
            else:
                result.status = "failed"

        result.final_messages = list(self.state.messages)
        result.final_context = dict(self.state.context.data)
        result.completed_at = datetime.now()

        return result

    async def execute_parallel(
        self,
        agents: List[AgentProtocol],
        message: Any,
        broadcast_all: bool = True
    ) -> HubExecutionResult:
        """
        Execute agents in parallel, all receiving the same message.

        Args:
            agents: Agents to execute
            message: Message to send to all agents
            broadcast_all: Whether to broadcast all messages

        Returns:
            HubExecutionResult with all agent results
        """
        result = HubExecutionResult(
            hub_id=self.state.hub_id,
            status="completed",
            started_at=datetime.now(),
            total_agents=len(agents)
        )

        # Broadcast initial message once
        if isinstance(message, str):
            await self.broadcast_user_message(message)

        async def execute_one(agent: AgentProtocol) -> Dict[str, Any]:
            agent_id = getattr(agent, 'agent_id', str(id(agent)))
            try:
                reply = await self.execute(
                    agent,
                    message,
                    broadcast_input=False,  # Already broadcast
                    broadcast_output=broadcast_all
                )
                return {
                    "agent_id": agent_id,
                    "status": "completed",
                    "output": getattr(reply, 'data', reply)
                }
            except Exception as e:
                return {
                    "agent_id": agent_id,
                    "status": "failed",
                    "error": str(e)
                }

        # Execute all in parallel
        tasks = [execute_one(agent) for agent in agents]
        agent_results = await asyncio.gather(*tasks)

        for r in agent_results:
            result.agent_results.append(r)
            if r["status"] == "completed":
                result.completed_agents += 1
            else:
                result.failed_agents += 1
                if "error" in r:
                    result.errors.append(f"{r['agent_id']}: {r['error']}")

        # Set final status
        if result.failed_agents > 0:
            if result.completed_agents > 0:
                result.status = "partial"
            else:
                result.status = "failed"

        result.final_messages = list(self.state.messages)
        result.final_context = dict(self.state.context.data)
        result.completed_at = datetime.now()

        return result

    def on_message(self, callback: Callable[[Message], Any]) -> None:
        """
        Register a callback for new messages.

        Args:
            callback: Function called with each new message
        """
        self._message_callbacks.append(callback)

    def format_context_for_agent(
        self,
        agent_id: str,
        include_messages: bool = True,
        include_context: bool = True,
        max_messages: int = 10
    ) -> str:
        """
        Format shared context as a string for agent consumption.

        Args:
            agent_id: Agent to format context for
            include_messages: Include message history
            include_context: Include shared context data
            max_messages: Maximum messages to include

        Returns:
            Formatted context string
        """
        parts = []

        if include_messages:
            messages = self.get_messages(participant_id=agent_id, limit=max_messages)
            if messages:
                parts.append("=== Conversation History ===")
                for msg in messages:
                    role_label = msg.role.value.upper()
                    sender = msg.sender_type or msg.sender_id
                    parts.append(f"[{role_label}] {sender}: {msg.content}")

        if include_context and self.state.context.data:
            parts.append("\n=== Shared Context ===")
            for key, value in self.state.context.data.items():
                parts.append(f"{key}: {value}")

        return "\n".join(parts)

    async def close(self) -> None:
        """Close the hub"""
        self.state.is_active = False
        self.state.closed_at = datetime.now()

        # Mark all participants as inactive
        for participant in self.state.participants.values():
            participant.is_active = False

    def get_state(self) -> MsgHubState:
        """Get current hub state"""
        return self.state

    def get_participant(self, agent_id: str) -> Optional[ParticipantInfo]:
        """Get participant info"""
        return self.state.participants.get(agent_id)

    @property
    def hub_id(self) -> str:
        """Get hub ID"""
        return self.state.hub_id

    @property
    def is_active(self) -> bool:
        """Check if hub is active"""
        return self.state.is_active

    @property
    def message_count(self) -> int:
        """Get message count"""
        return self.state.message_count

    @property
    def participant_count(self) -> int:
        """Get active participant count"""
        return self.state.participant_count


@asynccontextmanager
async def msghub(
    participants: Optional[List[AgentProtocol]] = None,
    config: Optional[MsgHubConfig] = None,
    hub_id: Optional[str] = None
) -> AsyncIterator[MsgHub]:
    """
    Context manager for creating a MsgHub.

    Example:
        async with msghub(participants=[agent1, agent2]) as hub:
            await hub.execute(agent1, "Hello")
    """
    hub = MsgHub(participants=participants, config=config, hub_id=hub_id)
    try:
        await hub.__aenter__()
        yield hub
    finally:
        await hub.__aexit__(None, None, None)

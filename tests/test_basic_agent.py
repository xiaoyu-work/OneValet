"""
Test basic agent functionality after refactoring.
"""

import pytest
from flowagents import (
    StandardAgent,
    RequiredField,
    Message,
    AgentResult,
    AgentStatus,
)


class GreetingAgent(StandardAgent):
    """Simple test agent that greets users"""

    def define_required_fields(self):
        return [
            RequiredField(
                name="name",
                description="User's name",
                prompt="What's your name?"
            )
        ]

    async def extract_fields(self, user_input: str):
        # Simple extraction - treat input as name if not a greeting
        greetings = ["hello", "hi", "hey", "start"]
        if user_input.strip().lower() in greetings:
            return {}
        return {"name": user_input.strip()}

    async def on_running(self, msg):
        name = self.collected_fields["name"]
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Hello, {name}!"
        )


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initializes correctly"""
    agent = GreetingAgent(tenant_id="test_user")

    assert agent.tenant_id == "test_user"
    assert agent.status == AgentStatus.INITIALIZING
    assert len(agent.required_fields) == 1
    assert agent.required_fields[0].name == "name"


@pytest.mark.asyncio
async def test_agent_field_collection():
    """Test agent collects fields correctly"""
    agent = GreetingAgent(tenant_id="test_user")

    # First message - trigger field collection
    result = await agent.reply(Message(
        name="user",
        content="Hello",
        role="user"
    ))

    assert result.status == AgentStatus.WAITING_FOR_INPUT
    assert "name" in result.missing_fields


@pytest.mark.asyncio
async def test_agent_execution():
    """Test agent executes after field collection"""
    agent = GreetingAgent(tenant_id="test_user")

    # First message
    await agent.reply(Message(name="user", content="Hello", role="user"))

    # Provide name
    result = await agent.reply(Message(name="user", content="Alice", role="user"))

    assert result.status == AgentStatus.COMPLETED
    assert "Hello, Alice!" in result.raw_message


@pytest.mark.asyncio
async def test_agent_context_hints():
    """Test agent uses context hints for pre-populated fields"""
    agent = GreetingAgent(
        tenant_id="test_user",
        context_hints={"name": "Bob"}
    )

    # With context hints, name is already collected
    result = await agent.reply(Message(name="user", content="Hello", role="user"))

    assert result.status == AgentStatus.COMPLETED
    assert "Hello, Bob!" in result.raw_message


@pytest.mark.asyncio
async def test_message_get_text():
    """Test Message.get_text() works correctly"""
    msg = Message(name="user", content="Hello world", role="user")
    assert msg.get_text() == "Hello world"


@pytest.mark.asyncio
async def test_agent_result_status_checks():
    """Test AgentResult status check methods"""
    completed_result = AgentResult(
        agent_type="TestAgent",
        status=AgentStatus.COMPLETED,
        raw_message="Done"
    )
    assert completed_result.is_completed() == True
    assert completed_result.is_waiting() == False
    assert completed_result.is_error() == False

    waiting_result = AgentResult(
        agent_type="TestAgent",
        status=AgentStatus.WAITING_FOR_INPUT,
        raw_message="Need input"
    )
    assert waiting_result.is_completed() == False
    assert waiting_result.is_waiting() == True
    assert waiting_result.is_error() == False

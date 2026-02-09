"""
Test that all FlowAgent imports work correctly after refactoring.
"""

import pytest


def test_core_imports():
    """Test core module imports"""
    from flowagents import (
        BaseAgent,
        StandardAgent,
        RequiredField,
        AgentState,
        Message,
        AgentResult,
        AgentStatus,
    )

    assert BaseAgent is not None
    assert StandardAgent is not None
    assert RequiredField is not None
    assert AgentState is not None
    assert Message is not None
    assert AgentResult is not None
    assert AgentStatus is not None


def test_validator_imports():
    """Test validator imports - validators are optional custom implementations"""
    # Note: Built-in validators were removed from flowagents core.
    # Users should define their own validators using register_validator()
    from flowagents.config.registry import register_validator, VALIDATORS

    # Register a custom validator
    def custom_email_validator(value: str) -> bool:
        return "@" in value and "." in value

    register_validator("email", custom_email_validator)

    # Test custom validator was registered
    assert "email" in VALIDATORS
    assert VALIDATORS["email"]("test@example.com") == True
    assert VALIDATORS["email"]("invalid") == False


def test_message_imports():
    """Test message system imports"""
    from flowagents import (
        Message,
        TextBlock,
        ImageBlock,
        AudioBlock,
        VideoBlock,
        ToolUseBlock,
        ToolResultBlock,
    )

    # Test Message creation
    msg = Message(name="user", content="Hello", role="user")
    assert msg.get_text() == "Hello"

    # Test TextBlock
    block = TextBlock(text="Test content")
    assert block.text == "Test content"
    assert block.type == "text"


def test_protocol_imports():
    """Test protocol imports"""
    from flowagents import (
        LLMClientProtocol,
        ConfigLoaderProtocol,
    )

    assert LLMClientProtocol is not None
    assert ConfigLoaderProtocol is not None


def test_tool_imports():
    """Test tool system imports"""
    from flowagents import (
        ToolRegistry,
        ToolExecutor,
        ToolDefinition,
        ToolCall,
        ToolResult,
        ToolCategory,
        ToolExecutionContext,
    )

    assert ToolRegistry is not None
    assert ToolExecutor is not None
    assert ToolDefinition is not None
    assert ToolCall is not None
    assert ToolResult is not None
    assert ToolCategory is not None
    assert ToolExecutionContext is not None


def test_mcp_imports():
    """Test MCP integration imports"""
    from flowagents import (
        MCPClientProtocol,
        MCPClient,
        MCPToolProvider,
        MCPManager,
        MCPServerConfig,
        MCPTool,
        MCPResource,
    )

    assert MCPClientProtocol is not None
    assert MCPClient is not None
    assert MCPToolProvider is not None
    assert MCPManager is not None
    assert MCPServerConfig is not None
    assert MCPTool is not None
    assert MCPResource is not None


def test_config_imports():
    """Test config system imports"""
    from flowagents import (
        ConfigLoader,
        AgentConfig,
        ToolConfig,
        MCPConfig,
        AgentRegistry,
    )

    assert ConfigLoader is not None
    assert AgentConfig is not None
    assert ToolConfig is not None
    assert MCPConfig is not None
    assert AgentRegistry is not None


def test_version():
    """Test version is defined"""
    from flowagents import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_submodule_imports():
    """Test submodule imports"""
    from flowagents.tools import ToolRegistry, ToolDefinition
    from flowagents.mcp import MCPClient, MCPToolProvider
    from flowagents.config import ConfigLoader, AgentRegistry
    from flowagents.memory import MemoryManager, MemoryConfig

    assert ToolRegistry is not None
    assert ToolDefinition is not None
    assert MCPClient is not None
    assert MCPToolProvider is not None
    assert ConfigLoader is not None
    assert AgentRegistry is not None
    assert MemoryManager is not None
    assert MemoryConfig is not None

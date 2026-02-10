"""
Agent Registry - Runtime registry that manages agents/tools/MCP

Agents are registered via @valet decorator only.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Callable

from ..tools.registry import ToolRegistry
from ..mcp.provider import MCPToolProvider, MCPManager
from ..mcp.protocol import MCPClientProtocol
from ..base_agent import BaseAgent
from ..llm.registry import LLMRegistry

logger = logging.getLogger(__name__)


# Validator registry - users register their own validators
VALIDATORS: Dict[str, Callable[[str], bool]] = {}


def register_validator(name: str, func: Callable[[str], bool]) -> None:
    """Register a custom validator"""
    VALIDATORS[name] = func


class AgentRegistry:
    """
    Runtime registry for agents, tools, and MCP servers

    Agents are registered via @valet decorator.

    Example:
        registry = AgentRegistry()
        await registry.initialize()

        # Get agent class (from decorator registry)
        AgentClass = registry.get_agent_class("SendEmailAgent")
        agent = AgentClass(user_id="123", llm_client=llm)

        # Cleanup
        await registry.shutdown()
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        llm_registry: Optional[LLMRegistry] = None,
        mcp_client_factory: Optional[Callable[[Any], MCPClientProtocol]] = None
    ):
        """
        Initialize agent registry

        Args:
            tool_registry: ToolRegistry to use (defaults to singleton)
            llm_registry: LLMRegistry to use (defaults to singleton)
            mcp_client_factory: Factory function to create MCP clients
        """
        self.tool_registry = tool_registry or ToolRegistry.get_instance()
        self.llm_registry = llm_registry or LLMRegistry.get_instance()
        self.mcp_client_factory = mcp_client_factory
        self.mcp_manager = MCPManager(self.tool_registry)

        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the registry.
        """
        if self._initialized:
            logger.warning("AgentRegistry already initialized")
            return

        self._initialized = True
        logger.info("AgentRegistry initialized")

    async def shutdown(self) -> None:
        """Disconnect all MCP servers and cleanup"""
        await self.mcp_manager.disconnect_all()
        self.llm_registry.clear()
        self._initialized = False
        logger.info("AgentRegistry shutdown complete")

    # ===== Agent Access (from decorator registry) =====

    def _get_agent_registry(self) -> Dict[str, Any]:
        """Get the decorator-based agent registry"""
        from ..agents.decorator import AGENT_REGISTRY
        return AGENT_REGISTRY

    def get_agent_class(self, name: str) -> Optional[Type[BaseAgent]]:
        """Get agent class by name from decorator registry"""
        registry = self._get_agent_registry()
        metadata = registry.get(name)
        if metadata:
            return metadata.agent_class
        return None

    def get_agent_metadata(self, name: str) -> Optional[Any]:
        """Get agent metadata by name"""
        registry = self._get_agent_registry()
        return registry.get(name)

    def get_agent_config(self, name: str) -> Optional[Any]:
        """Get agent config by name (alias for get_agent_metadata for backward compatibility)"""
        return self.get_agent_metadata(name)

    def get_all_agent_names(self) -> List[str]:
        """Get all registered agent names"""
        return list(self._get_agent_registry().keys())

    def get_all_agent_metadata(self) -> Dict[str, Any]:
        """Get all agent metadata"""
        return self._get_agent_registry()

    def create_agent(
        self,
        name: str,
        tenant_id: str = "default",
        llm_client: Optional[Any] = None,
        **kwargs
    ) -> Optional[BaseAgent]:
        """
        Create an agent instance

        Automatically injects LLM client from LLMRegistry based on agent metadata.
        If llm_client is provided, it takes precedence.

        Args:
            name: Agent name
            tenant_id: Tenant ID for multi-tenant isolation (default: "default")
            llm_client: LLM client (optional, will use config if not provided)
            **kwargs: Additional arguments

        Returns:
            Agent instance or None if not found
        """
        agent_class = self.get_agent_class(name)
        if not agent_class:
            logger.error(f"Agent not found: {name}")
            return None

        # Get LLM client from registry if not provided
        if llm_client is None:
            metadata = self.get_agent_metadata(name)
            if metadata and metadata.llm:
                llm_client = self.llm_registry.get(metadata.llm)
                if llm_client:
                    logger.debug(f"Injected LLM client '{metadata.llm}' for agent '{name}'")
                else:
                    logger.warning(f"LLM provider '{metadata.llm}' not found for agent '{name}'")

            # Fallback to default LLM client
            if llm_client is None:
                llm_client = self.llm_registry.get_default()

        return agent_class(tenant_id=tenant_id, llm_client=llm_client, **kwargs)

    # ===== Tool Access =====

    def get_tool_schemas(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """Get OpenAI-format tool schemas"""
        return self.tool_registry.get_tools_schema(tool_names)

    # ===== Validators =====

    def get_validator(self, name: str) -> Optional[Callable[[str], bool]]:
        """Get a validator function by name"""
        return VALIDATORS.get(name)

    # ===== Routing =====

    def get_all_agent_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return enhanced tool schemas for all agents with expose_as_tool=True."""
        from ..agents.decorator import generate_tool_schema, enhance_agent_tool_schema

        schemas = []
        for name, metadata in self._get_agent_registry().items():
            if not getattr(metadata, 'expose_as_tool', True):
                continue
            schema = generate_tool_schema(metadata.agent_class)
            schema = enhance_agent_tool_schema(metadata.agent_class, schema)
            schemas.append(schema)
        return schemas

    def get_schema_version(self, agent_type: str) -> Optional[int]:
        """Return schema version for a registered agent type."""
        from ..agents.decorator import get_schema_version

        agent_class = self.get_agent_class(agent_type)
        if agent_class is None:
            return None
        return get_schema_version(agent_class)

    def get_agent_descriptions(self) -> str:
        """
        Get formatted agent descriptions for LLM routing prompt.

        Includes all available info: description, inputs, outputs.
        """
        lines = ["Available agents:"]

        for name, metadata in self._get_agent_registry().items():
            description = metadata.description or metadata.agent_class.__doc__ or ""
            lines.append(f"- **{name}**: {description}")

            # Inputs
            if metadata.inputs:
                input_strs = [f"{i.name}" for i in metadata.inputs]
                lines.append(f"  - Inputs: {', '.join(input_strs)}")

            # Outputs
            if metadata.outputs:
                output_strs = [f"{o.name}" for o in metadata.outputs]
                lines.append(f"  - Outputs: {', '.join(output_strs)}")

        return "\n".join(lines)

"""
Agent Registry - Runtime registry that manages agents/tools/MCP

Agents are registered via @flowagent decorator only (not YAML).
YAML is only used for LLM providers, tools, and MCP servers.
"""

import logging
import importlib
from typing import Dict, List, Any, Optional, Type, Callable, Union

from .loader import ConfigLoader, AgentConfig, ToolConfig, MCPConfig, LLMProviderConfig, InputOutputConfig
from ..tools.registry import ToolRegistry
from ..tools.models import ToolDefinition, ToolCategory, ToolExecutionContext
from ..mcp.models import MCPServerConfig, MCPTransportType
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

    Agents are registered via @flowagent decorator.
    YAML configuration is used for LLM providers, tools, and MCP servers only.

    Example:
        # Load config and initialize
        registry = AgentRegistry(config_dir="./config")
        await registry.initialize()

        # Get agent class (from decorator registry)
        AgentClass = registry.get_agent_class("SendEmailAgent")
        agent = AgentClass(user_id="123", llm_client=llm)

        # Cleanup
        await registry.shutdown()
    """

    def __init__(
        self,
        config_dir: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        llm_registry: Optional[LLMRegistry] = None,
        mcp_client_factory: Optional[Callable[[MCPConfig], MCPClientProtocol]] = None
    ):
        """
        Initialize agent registry

        Args:
            config_dir: Path to config directory
            tool_registry: ToolRegistry to use (defaults to singleton)
            llm_registry: LLMRegistry to use (defaults to singleton)
            mcp_client_factory: Factory function to create MCP clients
        """
        self.config_loader = ConfigLoader(config_dir)
        self.tool_registry = tool_registry or ToolRegistry.get_instance()
        self.llm_registry = llm_registry or LLMRegistry.get_instance()
        self.mcp_client_factory = mcp_client_factory
        self.mcp_manager = MCPManager(self.tool_registry)

        self._initialized = False

    async def initialize(self) -> None:
        """
        Load config and initialize all components

        1. Load YAML configurations (LLM providers, tools, MCP servers only)
        2. Register LLM providers with LLMRegistry
        3. Register tools with ToolRegistry
        4. Connect MCP servers and register their tools
        """
        if self._initialized:
            logger.warning("AgentRegistry already initialized")
            return

        # Load configurations
        self.config_loader.load()

        # Register LLM providers
        self._register_llm_providers()

        # Register tools
        self._register_tools()

        # Connect MCP servers
        await self._connect_mcp_servers()

        self._initialized = True
        logger.info("AgentRegistry initialized")

    async def shutdown(self) -> None:
        """Disconnect all MCP servers and cleanup"""
        await self.mcp_manager.disconnect_all()
        self.llm_registry.clear()
        self._initialized = False
        logger.info("AgentRegistry shutdown complete")

    def _register_llm_providers(self) -> None:
        """Register LLM providers from config with LLMRegistry"""
        from ..llm.registry import LLMProviderConfig as LLMRegProviderConfig

        for name, config in self.config_loader.get_all_llm_providers().items():
            try:
                # Convert loader config to registry config
                provider_config = LLMRegProviderConfig(
                    name=config.name,
                    provider=config.provider,
                    model=config.model,
                    api_key_env=config.api_key_env,
                    api_key=config.api_key,
                    base_url=config.base_url,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                    extra=config.extra
                )

                self.llm_registry.register_from_config(provider_config)
                logger.debug(f"Registered LLM provider: {name}")

            except Exception as e:
                logger.error(f"Failed to register LLM provider {name}: {e}")

        # Set default and routing LLM from config
        default_llm = self.config_loader.get_default_llm_name()
        if default_llm:
            self.llm_registry.set_default(default_llm)

        routing_llm = self.config_loader.get_routing_llm_name()
        if routing_llm:
            self.llm_registry.set_routing(routing_llm)

    def _register_tools(self) -> None:
        """Register tools from config with ToolRegistry"""
        for name, config in self.config_loader.get_all_tools().items():
            try:
                # Import the function
                module = importlib.import_module(config.module)
                func = getattr(module, config.function)

                # Create ToolDefinition
                tool_def = ToolDefinition(
                    name=config.name,
                    description=config.description,
                    parameters=config.parameters,
                    executor=func,
                    category=ToolCategory(config.category) if config.category in [c.value for c in ToolCategory] else ToolCategory.CUSTOM,
                )

                self.tool_registry.register(tool_def)
                logger.debug(f"Registered tool: {name}")

            except Exception as e:
                logger.error(f"Failed to register tool {name}: {e}")

    async def _connect_mcp_servers(self) -> None:
        """Connect to MCP servers from config"""
        if not self.mcp_client_factory:
            logger.debug("No MCP client factory provided, skipping MCP setup")
            return

        for config in self.config_loader.get_enabled_mcp_servers():
            try:
                # Create client using factory
                client = self.mcp_client_factory(config)

                # Add to manager (connects and registers tools)
                await self.mcp_manager.add_server(client)
                logger.info(f"Connected MCP server: {config.name}")

            except Exception as e:
                logger.error(f"Failed to connect MCP server {config.name}: {e}")

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

    def find_agent_by_trigger(self, message: str) -> Optional[str]:
        """
        Find agent that matches a trigger

        Args:
            message: User message

        Returns:
            Agent name or None
        """
        message_lower = message.lower()

        for name, metadata in self._get_agent_registry().items():
            for trigger in metadata.triggers:
                if trigger.lower() in message_lower:
                    return name

        return None

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

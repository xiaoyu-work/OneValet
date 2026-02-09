"""
Config Loader - Load agent, tool, and MCP configurations from YAML
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


@dataclass
class FieldConfig:
    """Configuration for a required field (used in agent's field collection)"""
    name: str
    description: str
    prompt: str
    required: bool = True
    validator: Optional[str] = None  # Validator name (e.g., "email", "phone")


@dataclass
class InputOutputConfig:
    """Configuration for an input or output parameter"""
    name: str
    type: str = "string"  # string, number, boolean, array, object
    description: str = ""
    required: bool = False


@dataclass
class ToolConfig:
    """Configuration for a tool"""
    name: str
    description: str
    module: str  # Python module path
    function: str  # Function name in module
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False  # DEPRECATED: Use requires_approval in AgentConfig instead
    category: str = "utility"


@dataclass
class MCPConfig:
    """Configuration for an MCP server"""
    name: str
    transport: str = "stdio"  # stdio, sse, websocket
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    url: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider"""
    name: str
    provider: str  # openai, anthropic, dashscope, gemini, ollama
    model: str
    api_key_env: Optional[str] = None  # Environment variable name for API key
    api_key: Optional[str] = None  # Direct API key (not recommended)
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 60.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorYAMLConfig:
    """Configuration for the orchestrator from YAML"""
    llm_provider: str = ""  # LLM provider for routing decisions
    fallback_response: str = "I'm not sure how to help with that."
    max_agents_per_tenant: int = 10
    default_timeout_seconds: int = 300
    enable_streaming: bool = True


class AgentConfig(BaseModel):
    """Configuration for an agent"""
    name: str
    description: str = ""
    module: str  # Python module path
    class_name: str  # Class name in module

    # LLM provider for this agent (references llm_providers config)
    llm_provider: str = "main"

    # Routing triggers (keywords/patterns that route to this agent)
    triggers: List[str] = Field(default_factory=list)

    # Memory - if enabled, orchestrator will auto recall/store memories
    enable_memory: bool = False

    # Capabilities - what this agent can do (for routing decisions)
    capabilities: List[str] = Field(default_factory=list)

    # What this agent does NOT handle (helps LLM exclude during routing)
    does_not_handle: List[str] = Field(default_factory=list)

    # Input parameters this agent accepts (for context extraction)
    inputs: List[InputOutputConfig] = Field(default_factory=list)

    # Output parameters this agent produces (for agent chaining)
    outputs: List[InputOutputConfig] = Field(default_factory=list)

    # Tools this agent can use
    tools: List[str] = Field(default_factory=list)

    # MCP servers this agent can use
    mcp_servers: List[str] = Field(default_factory=list)

    # Extra config (for app-specific extensions)
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")  # Ignore unknown fields in YAML


class ConfigLoader:
    """
    Load configurations from YAML files

    Expected directory structure:
        config/
        ├── agents.yaml       # Agent definitions
        ├── tools.yaml        # Tool definitions
        ├── mcp.yaml          # MCP server definitions
        └── settings.yaml     # Global settings

    Or single file:
        config/flowagents.yaml  # All-in-one config

    Example agents.yaml:
        agents:
          SendEmailAgent:
            description: "Send an email"
            module: myapp.agents.email
            class_name: SendEmailAgent

            # Routing - triggers for LLM to match
            triggers:
              - send email
              - compose email
            group: email

            # Capabilities - what this agent can do (for routing)
            capabilities:
              - send email
              - compose email
              - forward email

            # What this agent does NOT handle (helps exclude during routing)
            does_not_handle:
              - read emails
              - delete emails

            # Input parameters (for context extraction)
            inputs:
              - name: recipient
                type: string
                description: Email recipient address
                required: true
              - name: subject
                type: string
                description: Email subject line
              - name: body
                type: string
                description: Email content
                required: true

            # Output parameters (for agent chaining)
            outputs:
              - name: message_id
                type: string
                description: ID of sent message
              - name: success
                type: boolean

            # Fields for interactive collection (prompts user)
            fields:
              - name: recipient
                description: Email recipient
                prompt: Who should I send to?
                validator: email
              - name: subject
                description: Email subject
                prompt: What's the subject?

            # Tools and MCP servers
            tools:
              - send_email
              - search_contacts
            mcp_servers:
              - gmail

            # Behavior
            requires_approval: true

    Example tools.yaml:
        tools:
          send_email:
            description: Send an email via SMTP
            module: myapp.tools.email
            function: send_email
            parameters:
              type: object
              properties:
                to: {type: string}
                subject: {type: string}
                body: {type: string}
              required: [to, subject, body]
            requires_approval: true
            category: email

    Example mcp.yaml:
        mcp_servers:
          gmail:
            transport: stdio
            command: npx
            args: ["-y", "@anthropic/mcp-server-gmail"]
            env:
              GMAIL_CREDENTIALS: /path/to/creds.json

          filesystem:
            transport: stdio
            command: npx
            args: ["-y", "@anthropic/mcp-server-filesystem", "/allowed/path"]
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config loader

        Args:
            config_dir: Path to config directory (default: ./config)
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self._tools: Dict[str, ToolConfig] = {}
        self._mcp_servers: Dict[str, MCPConfig] = {}
        self._llm_providers: Dict[str, LLMProviderConfig] = {}
        self._default_llm: Optional[str] = None  # default LLM for agents
        self._routing_llm: Optional[str] = None  # LLM for orchestrator routing
        self._orchestrator: Optional[OrchestratorYAMLConfig] = None
        self._settings: Dict[str, Any] = {}

    def load(self) -> None:
        """Load all configurations"""
        # Try single file first
        single_file = self.config_dir / "flowagents.yaml"
        if single_file.exists():
            self._load_single_file(single_file)
            return

        # Load separate files
        self._load_llm()
        self._load_tools()
        self._load_mcp()
        self._load_settings()

    def _load_single_file(self, path: Path) -> None:
        """Load all-in-one config file"""
        logger.info(f"Loading config from {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        # Parse LLM config (format: llm: {default, routing, providers})
        self._parse_llm(data.get('llm', {}))
        self._parse_orchestrator(data.get('orchestrator', {}))
        self._parse_tools(data.get('tools', {}))
        self._parse_mcp(data.get('mcp_servers', {}))
        self._settings = data.get('settings', {})

    def _load_llm(self) -> None:
        """Load llm.yaml"""
        path = self.config_dir / "llm.yaml"
        if not path.exists():
            logger.debug(f"No llm.yaml found at {path}")
            return

        logger.info(f"Loading LLM config from {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        self._parse_llm(data.get('llm', {}))

    def _load_tools(self) -> None:
        """Load tools.yaml"""
        path = self.config_dir / "tools.yaml"
        if not path.exists():
            logger.debug(f"No tools.yaml found at {path}")
            return

        logger.info(f"Loading tools from {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        self._parse_tools(data.get('tools', {}))

    def _load_mcp(self) -> None:
        """Load mcp.yaml"""
        path = self.config_dir / "mcp.yaml"
        if not path.exists():
            logger.debug(f"No mcp.yaml found at {path}")
            return

        logger.info(f"Loading MCP servers from {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        self._parse_mcp(data.get('mcp_servers', {}))

    def _load_settings(self) -> None:
        """Load settings.yaml"""
        path = self.config_dir / "settings.yaml"
        if not path.exists():
            return

        logger.info(f"Loading settings from {path}")
        with open(path, 'r') as f:
            self._settings = yaml.safe_load(f) or {}

    def _parse_llm(self, llm_data: Dict[str, Any]) -> None:
        """Parse LLM configuration (new format: llm: {default, routing, providers})"""
        if not llm_data:
            return

        # Parse providers first
        providers_data = llm_data.get('providers', {})
        self._parse_llm_providers_dict(providers_data)

        # Parse and validate default LLM
        self._default_llm = llm_data.get('default')
        if self._default_llm and self._default_llm not in self._llm_providers:
            raise ValueError(f"LLM config error: default='{self._default_llm}' not found in providers. "
                           f"Available: {list(self._llm_providers.keys())}")

        # Parse and validate routing LLM
        self._routing_llm = llm_data.get('routing')
        if self._routing_llm and self._routing_llm not in self._llm_providers:
            raise ValueError(f"LLM config error: routing='{self._routing_llm}' not found in providers. "
                           f"Available: {list(self._llm_providers.keys())}")

        logger.info(f"LLM config: default={self._default_llm}, routing={self._routing_llm}")

    def _parse_llm_providers_dict(self, providers_data: Dict[str, Any]) -> None:
        """Parse LLM providers dict"""
        for name, config in providers_data.items():
            self._llm_providers[name] = LLMProviderConfig(
                name=name,
                provider=config.get('provider', 'openai'),
                model=config.get('model', ''),
                api_key_env=config.get('api_key_env'),
                api_key=config.get('api_key'),
                base_url=config.get('base_url'),
                temperature=config.get('temperature', 0.7),
                max_tokens=config.get('max_tokens'),
                timeout=config.get('timeout', 60.0),
                extra=config.get('extra', {})
            )

        logger.info(f"Loaded {len(self._llm_providers)} LLM provider configurations")

    def _parse_orchestrator(self, orch_data: Dict[str, Any]) -> None:
        """Parse orchestrator configuration"""
        if not orch_data:
            self._orchestrator = OrchestratorYAMLConfig()
            return

        self._orchestrator = OrchestratorYAMLConfig(
            llm_provider=orch_data.get('llm_provider', ''),
            fallback_response=orch_data.get('fallback_response', "I'm not sure how to help with that."),
            max_agents_per_tenant=orch_data.get('max_agents_per_tenant', 10),
            default_timeout_seconds=orch_data.get('default_timeout_seconds', 300),
            enable_streaming=orch_data.get('enable_streaming', True),
        )
        logger.info(f"Loaded orchestrator configuration (LLM: {self._orchestrator.llm_provider})")

    def _parse_tools(self, tools_data: Dict[str, Any]) -> None:
        """Parse tools configuration"""
        for name, config in tools_data.items():
            self._tools[name] = ToolConfig(
                name=name,
                description=config.get('description', ''),
                module=config.get('module', ''),
                function=config.get('function', name),
                parameters=config.get('parameters', {}),
                requires_approval=config.get('requires_approval', False),
                category=config.get('category', 'utility')
            )

        logger.info(f"Loaded {len(self._tools)} tool configurations")

    def _parse_mcp(self, mcp_data: Dict[str, Any]) -> None:
        """Parse MCP servers configuration"""
        for name, config in mcp_data.items():
            self._mcp_servers[name] = MCPConfig(
                name=name,
                transport=config.get('transport', 'stdio'),
                command=config.get('command'),
                args=config.get('args', []),
                url=config.get('url'),
                env=config.get('env', {}),
                enabled=config.get('enabled', True)
            )

        logger.info(f"Loaded {len(self._mcp_servers)} MCP server configurations")

    # ===== Accessors =====

    def get_tool(self, name: str) -> Optional[ToolConfig]:
        """Get tool configuration by name"""
        return self._tools.get(name)

    def get_all_tools(self) -> Dict[str, ToolConfig]:
        """Get all tool configurations"""
        return dict(self._tools)

    def get_mcp_server(self, name: str) -> Optional[MCPConfig]:
        """Get MCP server configuration by name"""
        return self._mcp_servers.get(name)

    def get_all_mcp_servers(self) -> Dict[str, MCPConfig]:
        """Get all MCP server configurations"""
        return dict(self._mcp_servers)

    def get_enabled_mcp_servers(self) -> List[MCPConfig]:
        """Get all enabled MCP servers"""
        return [m for m in self._mcp_servers.values() if m.enabled]

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        return self._settings.get(key, default)

    def get_llm_provider(self, name: str) -> Optional[LLMProviderConfig]:
        """Get LLM provider configuration by name"""
        return self._llm_providers.get(name)

    def get_all_llm_providers(self) -> Dict[str, LLMProviderConfig]:
        """Get all LLM provider configurations"""
        return dict(self._llm_providers)

    def get_default_llm_name(self) -> Optional[str]:
        """Get default LLM provider name for agents"""
        return self._default_llm

    def get_routing_llm_name(self) -> Optional[str]:
        """Get LLM provider name for orchestrator routing"""
        return self._routing_llm

    def get_orchestrator_config(self) -> OrchestratorYAMLConfig:
        """Get orchestrator configuration"""
        if self._orchestrator is None:
            return OrchestratorYAMLConfig()
        return self._orchestrator

    def get_orchestrator_llm_provider(self) -> Optional[LLMProviderConfig]:
        """Get LLM provider for orchestrator routing"""
        if self._orchestrator and self._orchestrator.llm_provider:
            return self._llm_providers.get(self._orchestrator.llm_provider)
        return None

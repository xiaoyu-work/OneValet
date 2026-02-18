"""
OneValet Agent Decorator - Auto-register agent classes with @valet

Usage:
    from onevalet import valet, StandardAgent, InputField, OutputField

    @valet
    class SendEmailAgent(StandardAgent):
        '''Send emails to users'''

        recipient = InputField("Who should I send to?")
        subject = InputField("Subject?", required=False)

        message_id = OutputField(str, "ID of sent message")

        async def on_running(self, msg):
            ...

    # With parameters
    @valet(capabilities=["email"], enable_memory=True)
    class HelloAgent(StandardAgent):
        '''Say hello'''

        name = InputField("What's your name?")

        async def on_running(self, msg):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Hello, {self.name}!"
            )
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Callable, Union

from ..fields import InputField, OutputField

logger = logging.getLogger(__name__)


@dataclass
class InputSpec:
    """Specification for an input field (extracted from InputField)"""
    name: str
    prompt: str
    description: str
    required: bool = True
    default: Any = None
    validator: Optional[Callable] = None
    validator_description: Optional[str] = None


@dataclass
class OutputSpec:
    """Specification for an output field (extracted from OutputField)"""
    name: str
    type: type = str
    description: str = ""


@dataclass
class AgentMetadata:
    """
    Metadata for a registered agent class.

    Automatically extracted from:
    - Class docstring -> description
    - InputField class variables -> inputs
    - OutputField class variables -> outputs
    - @valet parameters -> llm, capabilities, enable_memory, extra
    """
    name: str
    agent_class: Type
    description: str = ""

    # LLM provider name
    llm: Optional[str] = None

    # Capabilities - what this agent can do (for routing decisions)
    capabilities: List[str] = field(default_factory=list)

    # Input fields (extracted from InputField class variables)
    inputs: List[InputSpec] = field(default_factory=list)

    # Output fields (extracted from OutputField class variables)
    outputs: List[OutputSpec] = field(default_factory=list)

    # Module path (auto-populated)
    module: str = ""

    # Memory - if enabled, orchestrator will auto recall/store memories
    enable_memory: bool = False

    # Whether this agent should be exposed as a tool in the ReAct loop
    expose_as_tool: bool = True

    # Extra config (for app-specific extensions like required_tier)
    extra: Dict[str, Any] = field(default_factory=dict)


# Global registry for decorated agents
# Key: agent class name, Value: AgentMetadata
AGENT_REGISTRY: Dict[str, AgentMetadata] = {}


def _extract_fields(cls: Type) -> tuple[List[InputSpec], List[OutputSpec]]:
    """
    Extract InputField and OutputField from class variables.

    Returns:
        Tuple of (input_specs, output_specs)
    """
    inputs = []
    outputs = []

    # Scan class attributes (not instance attributes)
    for name, value in vars(cls).items():
        if name.startswith("_"):
            continue

        if isinstance(value, InputField):
            value.name = name
            inputs.append(InputSpec(
                name=name,
                prompt=value.prompt,
                description=value.description,
                required=value.required,
                default=value.default,
                validator=value.validator,
                validator_description=getattr(value, 'validator_description', None),
            ))

        elif isinstance(value, OutputField):
            value.name = name
            outputs.append(OutputSpec(
                name=name,
                type=value.type,
                description=value.description,
            ))

    return inputs, outputs


def valet(
    _cls: Optional[Type] = None,
    *,
    llm: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
    enable_memory: bool = False,
    expose_as_tool: bool = True,
    requires_service: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Union[Type, Callable[[Type], Type]]:
    """
    Decorator to register an agent class.

    Can be used with or without arguments:
        @valet
        class MyAgent(StandardAgent): ...

        @valet(capabilities=["email"], enable_memory=True)
        class MyAgent(StandardAgent): ...

    Args:
        llm: LLM provider name (optional, uses default if not specified)
        capabilities: What this agent can do (for routing decisions)
        enable_memory: If True, orchestrator will auto recall/store memories
        expose_as_tool: If True, agent is exposed as a tool in the ReAct loop (default: True)
        requires_service: Credential service names this agent depends on.
            The agent is only available to a tenant who has at least one of
            these services configured in CredentialStore.  When omitted the
            agent is always available.
        extra: App-specific extensions (e.g., required_tier)

    The decorator automatically extracts:
        - description: from class docstring
        - inputs: from InputField class variables
        - outputs: from OutputField class variables
    """

    def decorator(cls: Type) -> Type:
        # Extract description from docstring
        description = cls.__doc__ or f"{cls.__name__} agent"
        # Clean up docstring (remove extra whitespace)
        description = " ".join(description.split())

        # Extract InputField and OutputField from class
        inputs, outputs = _extract_fields(cls)

        # Build extra dict, merging caller-supplied extra with requires_service
        merged_extra = dict(extra) if extra else {}
        if requires_service:
            merged_extra["requires_service"] = list(requires_service)

        # Create metadata
        metadata = AgentMetadata(
            name=cls.__name__,
            agent_class=cls,
            description=description,
            llm=llm,
            capabilities=capabilities or [],
            inputs=inputs,
            outputs=outputs,
            module=cls.__module__,
            enable_memory=enable_memory,
            expose_as_tool=expose_as_tool,
            extra=merged_extra,
        )

        # Attach metadata to class
        cls._valet_metadata = metadata

        # Store input/output specs on class for StandardAgent to use
        cls._input_specs = inputs
        cls._output_specs = outputs

        # Register globally
        AGENT_REGISTRY[cls.__name__] = metadata
        logger.debug(f"Registered agent: {cls.__name__} (inputs={[i.name for i in inputs]}, outputs={[o.name for o in outputs]})")

        return cls

    # Support both @valet and @valet(...)
    if _cls is not None:
        return decorator(_cls)
    return decorator


def get_agent_metadata(cls: Type) -> Optional[AgentMetadata]:
    """
    Get the AgentMetadata attached to a decorated class.

    Args:
        cls: A class decorated with @valet

    Returns:
        AgentMetadata or None if not decorated
    """
    return getattr(cls, "_valet_metadata", None)


def is_valet(cls: Type) -> bool:
    """
    Check if a class is decorated with @valet.

    Args:
        cls: Class to check

    Returns:
        True if decorated with @valet
    """
    return hasattr(cls, "_valet_metadata")


# ===== Tool Schema Generation =====

# Map Python type names to JSON Schema types
_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def generate_tool_schema(agent_cls: Type) -> Dict[str, Any]:
    """Generate a lightweight OpenAI function-calling tool schema for agent routing.

    Only exposes name, description, and a single task_instruction parameter.
    The orchestrator ReAct loop uses this to pick the right agent; the agent's
    own internal ReAct loop handles parameter extraction and tool execution.
    """
    metadata: AgentMetadata = getattr(agent_cls, "_valet_metadata", None)
    if metadata is None:
        raise ValueError(f"{agent_cls.__name__} is not decorated with @valet")

    return {
        "type": "function",
        "function": {
            "name": metadata.name,
            "description": metadata.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "task_instruction": {
                        "type": "string",
                        "description": "What the user wants this agent to do.",
                    },
                },
                "required": ["task_instruction"],
            },
        },
    }


def enhance_agent_tool_schema(agent_cls: Type, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance auto-generated schema before exposing to LLM.

    With lightweight schemas (name + description + task_instruction only),
    the main enhancement is surfacing the approval requirement in the
    tool description so the orchestrator LLM is aware of it.
    """
    from ..standard_agent import StandardAgent

    metadata: AgentMetadata = getattr(agent_cls, "_valet_metadata", None)
    if metadata is None:
        return schema

    func = schema.get("function", {})

    # Surface approval requirement in tool description
    if (
        hasattr(agent_cls, "needs_approval")
        and agent_cls.needs_approval is not StandardAgent.needs_approval
    ):
        existing_desc = func.get("description", "")
        func["description"] = f"{existing_desc} [Requires user confirmation before execution]"

    return schema


def get_schema_version(agent_cls: Type) -> int:
    """Compute schema version from InputField definitions.

    Hash of InputField names, types, and required flags.
    Changes when fields are added/removed/retyped.
    """
    metadata: AgentMetadata = getattr(agent_cls, "_valet_metadata", None)
    if metadata is None:
        return 0

    parts: List[str] = []
    for inp in sorted(metadata.inputs, key=lambda i: i.name):
        type_name = type(inp.default).__name__ if inp.default is not None else "str"
        parts.append(f"{inp.name}:{type_name}:{inp.required}")

    hash_input = "|".join(parts)
    digest = hashlib.sha256(hash_input.encode()).hexdigest()
    # Return a stable integer from the first 8 hex chars
    return int(digest[:8], 16)

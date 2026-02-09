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
    - @valet parameters -> triggers, llm, capabilities, enable_memory, extra
    """
    name: str
    agent_class: Type
    description: str = ""

    # LLM provider name
    llm: Optional[str] = None

    # Routing triggers (keywords/patterns that route to this agent)
    triggers: List[str] = field(default_factory=list)

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
    triggers: Optional[List[str]] = None,
    llm: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
    enable_memory: bool = False,
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
        triggers: Keywords/patterns that route messages to this agent (optional)
        llm: LLM provider name (optional, uses default if not specified)
        capabilities: What this agent can do (for routing decisions)
        enable_memory: If True, orchestrator will auto recall/store memories
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

        # Create metadata
        metadata = AgentMetadata(
            name=cls.__name__,
            agent_class=cls,
            description=description,
            llm=llm,
            triggers=triggers or [],
            capabilities=capabilities or [],
            inputs=inputs,
            outputs=outputs,
            module=cls.__module__,
            enable_memory=enable_memory,
            extra=extra or {},
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

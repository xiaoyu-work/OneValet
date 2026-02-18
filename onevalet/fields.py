"""
OneValet Fields - InputField and OutputField for agent definition

These are Python descriptors that enable clean syntax for accessing
input/output values on agent instances.

Usage:
    from onevalet import InputField, OutputField, valet, StandardAgent

    @valet()
    class SendEmailAgent(StandardAgent):
        '''Send emails to users'''

        # Inputs - collected from user
        recipient = InputField(
            prompt="Who should I send to?",
            validator=lambda x: None if "@" in x else "Invalid email",
        )
        subject = InputField("What's the subject?", required=False)

        # Outputs - set by agent
        message_id = OutputField(str, "ID of sent message")
        success = OutputField(bool)

        async def on_running(self, msg):
            # Access inputs directly
            to = self.recipient

            # Set outputs directly
            self.message_id = "123"
            self.success = True

            return self.make_result(...)
"""

from typing import Any, Callable, Optional, Type


class InputField:
    """
    Descriptor for input fields collected from user.

    Args:
        prompt: Question to ask the user
        description: Description for LLM routing (defaults to prompt)
        required: Whether this field is required (default: True)
        default: Default value if not required
        validator: Validation function, returns None if valid, error message string if invalid

    Example:
        # Simple
        name = InputField("What's your name?")

        # Optional with default
        subject = InputField("Subject?", required=False, default="No Subject")

        # With validator
        def validate_email(value):
            if not value:
                return "Email cannot be empty"
            if "@" not in value:
                return "Email must contain @"
            return None  # Valid

        email = InputField(
            prompt="Your email?",
            description="User's email address",
            validator=validate_email,
        )
    """

    def __init__(
        self,
        prompt: str,
        description: Optional[str] = None,
        required: bool = True,
        default: Any = None,
        validator: Optional[Callable[[Any], Optional[str]]] = None,
        validator_description: Optional[str] = None,
    ):
        self.prompt = prompt
        self.description = description or prompt
        self.required = required
        self.default = default
        self.validator = validator
        self.validator_description = validator_description
        self.name: Optional[str] = None

    def __set_name__(self, owner: Type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self.name = name

    def __get__(self, obj: Any, objtype: Optional[Type] = None) -> Any:
        """Get the field value from collected_fields."""
        if obj is None:
            # Accessed on class, return descriptor itself
            return self
        collected = getattr(obj, 'collected_fields', None)
        if collected is None:
            return self.default
        return collected.get(self.name, self.default)

    def __set__(self, obj: Any, value: Any) -> None:
        """Set the field value in collected_fields."""
        collected = getattr(obj, 'collected_fields', None)
        if collected is not None:
            collected[self.name] = value

    def validate(self, value: Any) -> Optional[str]:
        """
        Validate a value.

        Args:
            value: The value to validate

        Returns:
            None if valid, error message string if invalid
        """
        if self.validator is None:
            return None
        return self.validator(value)

    def __repr__(self) -> str:
        return f"InputField(prompt={self.prompt!r}, required={self.required})"


class OutputField:
    """
    Descriptor for output fields set by agent.

    Output fields are used for:
    - Agent chaining (passing data between agents)
    - LLM routing (understanding what an agent produces)
    - Documentation

    Args:
        type: The type of the output (str, int, bool, etc.)
        description: Description of what this output represents

    Example:
        message_id = OutputField(str, "ID of the sent message")
        success = OutputField(bool, "Whether the operation succeeded")
        count = OutputField(int)
    """

    def __init__(
        self,
        field_type: Type = str,
        description: str = "",
    ):
        self.type = field_type
        self.description = description
        self.name: Optional[str] = None

    def __set_name__(self, owner: Type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self.name = name

    def __get__(self, obj: Any, objtype: Optional[Type] = None) -> Any:
        """Get the output value."""
        if obj is None:
            # Accessed on class, return descriptor itself
            return self
        output_values = getattr(obj, '_output_values', None)
        if output_values is None:
            return None
        return output_values.get(self.name)

    def __set__(self, obj: Any, value: Any) -> None:
        """Set the output value."""
        output_values = getattr(obj, '_output_values', None)
        if output_values is None:
            obj._output_values = {}
            output_values = obj._output_values
        output_values[self.name] = value

    def __repr__(self) -> str:
        return f"OutputField(type={self.type.__name__}, description={self.description!r})"

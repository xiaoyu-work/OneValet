# Agent Development Guide

## Basic Agent

```python
from onevalet import valet, StandardAgent, InputField, AgentStatus

@valet
class BookingAgent(StandardAgent):
    guests = InputField("How many guests?")
    date = InputField("What date?")

    async def on_running(self, msg):
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Booked for {self.guests} on {self.date}!"
        )
```

## The @valet Decorator

Register agent classes with the `@valet` decorator. It can be used bare or with parameters:

```python
# Bare (no parameters)
@valet
class MyAgent(StandardAgent): ...

# With parameters
@valet(capabilities=["email"], enable_memory=True)
class MyAgent(StandardAgent): ...
```

Available parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `triggers` | `list[str]` | `None` | Keywords/patterns that route messages to this agent |
| `llm` | `str` | `None` | LLM provider name (uses default if not specified) |
| `capabilities` | `list[str]` | `None` | What this agent can do (for routing decisions) |
| `enable_memory` | `bool` | `False` | Auto recall/store memories via the orchestrator |
| `expose_as_tool` | `bool` | `True` | Expose this agent as a tool in the ReAct loop |
| `extra` | `dict` | `None` | App-specific extensions (e.g., `required_tier`) |

## InputField Options

```python
InputField(
    prompt="Question to ask",       # Required
    description="For LLM context",  # Optional (defaults to prompt)
    validator=my_validator,          # Optional
    required=True,                   # Default: True
    default="value",                 # If not required
)
```

## Validation

Validators are functions that return `None` if the value is valid, or an error message string if invalid. They should **not** raise exceptions.

```python
def validate_guests(value):
    if not value.isdigit():
        return "Please enter a number"
    if int(value) < 1 or int(value) > 20:
        return "1-20 guests only"
    return None  # Valid

@valet
class BookingAgent(StandardAgent):
    guests = InputField("How many guests?", validator=validate_guests)
```

You can also use inline lambdas for simple checks:

```python
email = InputField(
    prompt="Your email?",
    validator=lambda x: None if "@" in x else "Invalid email",
)
```

## OutputField

Use `OutputField` to declare structured outputs from your agent:

```python
from onevalet import valet, StandardAgent, InputField, OutputField, AgentStatus

@valet
class BookingAgent(StandardAgent):
    guests = InputField("How many?")

    booking_id = OutputField(str, "Confirmation ID")

    async def on_running(self, msg):
        self.booking_id = "BK-12345"  # Set output
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Booked! ID: {self.booking_id}"
        )
```

## State Handlers

Override to customize behavior at each state:

```python
@valet
class MyAgent(StandardAgent):
    name = InputField("Name?")

    async def on_initializing(self, msg):
        # Called first. Default: extract fields, go to next state
        return await super().on_initializing(msg)

    async def on_waiting_for_input(self, msg):
        # Collecting fields. Default: extract, check completion
        return await super().on_waiting_for_input(msg)

    async def on_waiting_for_approval(self, msg):
        # Awaiting yes/no. Default: parse response
        return await super().on_waiting_for_approval(msg)

    async def on_running(self, msg):
        # YOUR BUSINESS LOGIC - must override
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Hello, {self.name}!"
        )

    async def on_error(self, msg):
        # Error recovery. Default: return error message
        return await super().on_error(msg)
```

Most handlers have good defaults. You usually only need to override `on_running`.

## Error Handling

```python
async def on_running(self, msg):
    try:
        result = await self.do_something()
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Done: {result}"
        )
    except Exception as e:
        return self.make_result(
            status=AgentStatus.ERROR,
            raw_message=f"Failed: {e}"
        )
```

## Best Practices

1. **One agent = one task** - Keep agents focused
2. **Validate early** - Use validators to catch bad input (return `None` if valid, error string if not)
3. **Use approval for sensitive actions** - Override `on_waiting_for_approval` for deletes, sends, payments
4. **Use `make_result`** - Always return via `make_result` with a clear `AgentStatus`
5. **Write docstrings** - The class docstring becomes the agent description used for LLM routing

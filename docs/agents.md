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

## InputField Options

```python
InputField(
    prompt="Question to ask",       # Required
    description="For LLM context",  # Optional
    validator=my_validator,         # Optional
    required=True,                  # Default: True
    default="value",                # If not required
)
```

## Validation

```python
def validate_guests(value):
    if not value.isdigit():
        raise ValueError("Please enter a number")
    if int(value) < 1 or int(value) > 20:
        raise ValueError("1-20 guests only")
    return True

@valet
class BookingAgent(StandardAgent):
    guests = InputField("How many guests?", validator=validate_guests)
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

## Approval Flow

```python
@valet(requires_approval=True)
class DeleteAgent(StandardAgent):
    item = InputField("What to delete?")

    def get_approval_prompt(self):
        return f"Delete '{self.item}'? (yes/no)"

    async def on_running(self, msg):
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Deleted {self.item}!"
        )
```

## Output Fields

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

## Multi-tenant

```python
agent = BookingAgent(tenant_id="company_a")
```

Data (checkpoints, memory) is isolated by `tenant_id`.

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
2. **Validate early** - Use validators to catch bad input
3. **Use approval for sensitive actions** - Deletes, sends, payments
4. **Use tenant_id** - For multi-user applications

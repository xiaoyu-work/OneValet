# State Machine

OneValet uses a state machine to manage agent lifecycle. This document explains the default states, transitions, and how to customize behavior.

## Default States

| State | Description |
|-------|-------------|
| `INITIALIZING` | Agent created, ready to start |
| `WAITING_FOR_INPUT` | Waiting for user to provide required fields |
| `WAITING_FOR_APPROVAL` | All fields collected, waiting for user confirmation |
| `RUNNING` | Executing the main task |
| `PAUSED` | Temporarily suspended |
| `COMPLETED` | Task finished successfully |
| `ERROR` | Task failed with error |
| `CANCELLED` | User cancelled the task |

## State Transition Diagram

```
                    ┌─────────────────┐
                    │  INITIALIZING   │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
┌──────────────────┐  ┌─────────────┐  ┌──────────────────────┐
│ WAITING_FOR_INPUT│  │   RUNNING   │  │ WAITING_FOR_APPROVAL │
└────────┬─────────┘  └──────┬──────┘  └──────────┬───────────┘
         │                   │                    │
         │    ┌──────────────┼────────────────────┘
         │    │              │
         │    ▼              ▼
         │  ┌───────────────────┐
         └─►│     COMPLETED     │◄── Terminal
            └───────────────────┘

            ┌───────────────────┐
            │       ERROR       │◄── Terminal
            └───────────────────┘

            ┌───────────────────┐
            │     CANCELLED     │◄── Terminal
            └───────────────────┘
```

## Default State Transitions

The following transitions are allowed by default:

```python
STATE_TRANSITIONS = {
    INITIALIZING: [RUNNING, WAITING_FOR_INPUT, WAITING_FOR_APPROVAL, PAUSED, COMPLETED, ERROR],
    RUNNING: [COMPLETED, ERROR, PAUSED, WAITING_FOR_INPUT, WAITING_FOR_APPROVAL],
    WAITING_FOR_INPUT: [RUNNING, WAITING_FOR_APPROVAL, PAUSED, COMPLETED, ERROR, WAITING_FOR_INPUT],
    WAITING_FOR_APPROVAL: [RUNNING, WAITING_FOR_INPUT, WAITING_FOR_APPROVAL, PAUSED, COMPLETED, CANCELLED, ERROR],
    PAUSED: [INITIALIZING, RUNNING, WAITING_FOR_INPUT, WAITING_FOR_APPROVAL, CANCELLED, ERROR],
    COMPLETED: [],  # Terminal - no transitions out
    ERROR: [CANCELLED],
    CANCELLED: []   # Terminal - no transitions out
}
```

## Typical Flow

### Simple Agent (no fields)

```
INITIALIZING → RUNNING → COMPLETED
```

### Agent with Fields

```
INITIALIZING → WAITING_FOR_INPUT → RUNNING → COMPLETED
                      ↑     │
                      └─────┘  (collect more fields)
```

### Agent with Approval

```
INITIALIZING → WAITING_FOR_INPUT → WAITING_FOR_APPROVAL → RUNNING → COMPLETED
                                          │
                                          └→ CANCELLED (user rejected)
```

## State Handlers

Each state has a corresponding handler method. Override these to customize behavior:

| State | Handler | When Called |
|-------|---------|-------------|
| `INITIALIZING` | `on_initializing(msg)` | Agent starts |
| `WAITING_FOR_INPUT` | `on_waiting_for_input(msg)` | Collecting fields |
| `WAITING_FOR_APPROVAL` | `on_waiting_for_approval(msg)` | Awaiting user confirmation |
| `RUNNING` | `on_running(msg)` | Executing main task |
| `PAUSED` | `on_paused(msg)` | Agent is paused |
| `ERROR` | `on_error(msg)` | Error occurred |
| `COMPLETED` | (none) | Terminal state |
| `CANCELLED` | (none) | Terminal state |

## Customizing Behavior

### Override State Handlers

The most common customization is overriding `on_running()`:

```python
@valet(triggers=["order"])
class OrderAgent(StandardAgent):
    item = InputField("What would you like to order?")

    async def on_running(self, msg):
        # Your custom logic here
        order_id = await create_order(self.item)

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Order created: {order_id}"
        )
```

### Override Approval Handling

Customize the approval prompt and logic:

```python
@valet(triggers=["delete"], requires_approval=True)
class DeleteAgent(StandardAgent):
    target = InputField("What to delete?")

    def get_approval_prompt(self) -> str:
        return f"Delete '{self.target}'? This cannot be undone. (yes/no)"

    async def on_waiting_for_approval(self, msg):
        # Custom approval parsing
        user_input = msg.get_text().lower()

        if user_input == "yes delete":
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

        return self.make_result(
            status=AgentStatus.CANCELLED,
            raw_message="Deletion cancelled."
        )
```

### Override Field Collection

Customize how fields are collected:

```python
@valet(triggers=["survey"])
class SurveyAgent(StandardAgent):
    rating = InputField("Rate 1-5")
    comment = InputField("Any comments?")

    async def on_waiting_for_input(self, msg):
        user_input = msg.get_text()

        # Custom extraction logic
        if "skip" in user_input.lower():
            # Allow skipping optional fields
            current_field = self._get_missing_fields()[0]
            self.collected_fields[current_field] = "N/A"

        # Call parent for default behavior
        return await super().on_waiting_for_input(msg)
```

### Override Transition Validation

Customize which transitions are allowed:

```python
class StrictAgent(StandardAgent):
    def can_transition(self, from_state: AgentStatus, to_state: AgentStatus) -> bool:
        # Prevent going back to WAITING_FOR_INPUT from RUNNING
        if from_state == AgentStatus.RUNNING and to_state == AgentStatus.WAITING_FOR_INPUT:
            return False

        return super().can_transition(from_state, to_state)
```

## Manual State Transitions

Use `transition_to()` and `make_result()` for manual control:

```python
async def on_running(self, msg):
    if needs_more_info:
        # Go back to collect more input
        self.transition_to(AgentStatus.WAITING_FOR_INPUT)
        return self.make_result(
            status=AgentStatus.WAITING_FOR_INPUT,
            raw_message="I need more details. What else?"
        )

    return self.make_result(
        status=AgentStatus.COMPLETED,
        raw_message="Done!"
    )
```

## Limitations

**Custom states are not supported.** The `AgentStatus` enum is fixed and cannot be extended at runtime. If you need additional states, consider:

1. **Use `metadata`** - Store custom state info in result metadata
2. **Use `PAUSED`** - Treat PAUSED as a generic "waiting" state
3. **Internal flags** - Use instance variables to track sub-states

Example using metadata for sub-states:

```python
async def on_running(self, msg):
    sub_state = self.metadata.get("sub_state", "step1")

    if sub_state == "step1":
        # Do step 1
        self.metadata["sub_state"] = "step2"
        return self.make_result(
            status=AgentStatus.WAITING_FOR_INPUT,
            raw_message="Step 1 complete. Ready for step 2?"
        )

    elif sub_state == "step2":
        # Do step 2
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message="All steps complete!"
        )
```

## Related

- [Agents](agents.md) - Agent types and creation
- [Workflow](workflow.md) - Multi-agent workflows

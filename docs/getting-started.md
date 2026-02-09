# Getting Started

Build your first AI agent with OneValet in 5 minutes.

## Installation

```bash
pip install onevalet

# With a specific LLM provider
pip install onevalet[openai]

# With all providers
pip install onevalet[all]
```

## Your First Agent

```python
import asyncio
from onevalet import valet, StandardAgent, InputField, AgentStatus, Orchestrator

@valet(triggers=["greet", "hello"])
class GreetingAgent(StandardAgent):
    name = InputField("What's your name?")

    async def on_running(self, msg):
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Hello, {self.name}!"
        )

async def main():
    from onevalet import OpenAIClient
    llm = OpenAIClient(api_key="sk-xxx", model="gpt-4o-mini")

    orchestrator = Orchestrator(llm_client=llm)
    await orchestrator.initialize()

    result = await orchestrator.handle_message("user_1", "Hello!")
    print(result.raw_message)  # "What's your name?"

    result = await orchestrator.handle_message("user_1", "Alice")
    print(result.raw_message)  # "Hello, Alice!"

asyncio.run(main())
```

The Orchestrator:
- Routes messages to agents based on triggers
- Tracks conversation state per tenant
- Handles field collection automatically

## Understanding Agent States

```
INITIALIZING → WAITING_FOR_INPUT → RUNNING → COMPLETED
                      ↓
              WAITING_FOR_APPROVAL
```

| State | Description |
|-------|-------------|
| `INITIALIZING` | Agent starts, extracts fields from first message |
| `WAITING_FOR_INPUT` | Collecting required fields from user |
| `WAITING_FOR_APPROVAL` | Waiting for user to confirm (if enabled) |
| `RUNNING` | Executes `on_running()` - your business logic |
| `COMPLETED` | Task finished |
| `ERROR` | An error occurred |

## Adding Validation

```python
from onevalet import valet, StandardAgent, InputField, AgentStatus

def validate_guests(value):
    if not value.isdigit():
        raise ValueError("Please enter a number")
    if int(value) < 1 or int(value) > 20:
        raise ValueError("We can accommodate 1-20 guests")
    return True

@valet
class BookingAgent(StandardAgent):
    guests = InputField("How many guests?", validator=validate_guests)
    date = InputField("What date?")
    name = InputField("Name for the reservation?")

    async def on_running(self, msg):
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Booked for {self.guests} on {self.date} under {self.name}!"
        )
```

## Adding Approval

For sensitive actions, require user confirmation:

```python
@valet(requires_approval=True)
class DeleteAgent(StandardAgent):
    item = InputField("What to delete?")

    def get_approval_prompt(self):
        return f"Delete '{self.item}'? (yes/no)"

    async def on_running(self, msg):
        # Only runs after user says "yes"
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Deleted {self.item}!"
        )
```

## Using with LLM

Add an LLM for smart field extraction:

```python
from onevalet import OpenAIClient

llm = OpenAIClient(api_key="sk-xxx", model="gpt-4o-mini")
agent = BookingAgent(llm_client=llm)
```

Or configure via YAML (`config/onevalet.yaml`):

```yaml
llm:
  default: main       # LLM for agents
  routing: quick      # LLM for Orchestrator routing (optional)

  providers:
    main:
      provider: openai          # openai, anthropic, azure, dashscope, gemini, ollama
      model: gpt-4o
      api_key: ${OPENAI_API_KEY}

    quick:
      provider: openai
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
```

```python
orchestrator = Orchestrator(config_dir="./config")
await orchestrator.initialize()  # Loads LLM from YAML
```

Now the agent can extract fields from natural language like "Book a table for 4 on Friday under John".

## Next Steps

- [Agents](agents.md) - Deep dive into agent development
- [Tools](tools.md) - Add tools for LLM function calling
- [LLM Providers](llm-providers.md) - Configure different LLM backends

# Orchestrator

The Orchestrator routes messages to the appropriate agent and manages multi-agent workflows.

## Quick Start

```python
from flowagents import Orchestrator, OpenAIClient, StandardAgent, flowagent, InputField, AgentStatus

@flowagent(triggers=["book table", "reservation"])
class BookingAgent(StandardAgent):
    guests = InputField("How many guests?")
    async def on_running(self, msg):
        return self.make_result(status=AgentStatus.COMPLETED, raw_message="Booked!")

@flowagent(triggers=["leave request", "time off"])
class LeaveAgent(StandardAgent):
    start_date = InputField("Start date?")
    async def on_running(self, msg):
        return self.make_result(status=AgentStatus.COMPLETED, raw_message="Submitted!")

# Create orchestrator
llm = OpenAIClient(api_key="sk-xxx", model="gpt-4o-mini")
orchestrator = Orchestrator(llm_client=llm)
await orchestrator.initialize()

# Routes automatically based on triggers
result = await orchestrator.handle_message("user_1", "I need to book a table")  # → BookingAgent
result = await orchestrator.handle_message("user_1", "Request time off")         # → LeaveAgent
```

## Routing Strategies

### Trigger-based (Default)

Routes based on `triggers` in `@flowagent`:

```python
@flowagent(triggers=["book", "reserve", "reservation"])
class BookingAgent(StandardAgent):
    ...
```

### LLM-based

Uses LLM to understand intent:

```python
orchestrator = Orchestrator(
    agents=[...],
    routing_strategy="llm",
)
```

### Custom Routing

```python
class MyOrchestrator(Orchestrator):
    async def route_message(self, message: str, context: dict):
        if "urgent" in message.lower():
            return RoutingDecision(agent_type="UrgentAgent", confidence=1.0)
        return await super().route_message(message, context)
```

## Context Hooks

```python
class MyOrchestrator(Orchestrator):

    async def prepare_context(self, message: str, context: dict) -> dict:
        """Add context before processing"""
        context["user_prefs"] = await get_user_prefs(context["tenant_id"])
        return context

    async def should_process(self, message: str, context: dict) -> bool:
        """Guard: rate limiting, content filtering"""
        return not await is_rate_limited(context["tenant_id"])

    async def post_process(self, result, context: dict):
        """After processing: logging, save to history"""
        await save_to_history(context["tenant_id"], result)
        return result
```

## Multi-tenant

```python
result = await orchestrator.process(
    message="Book a table for 4",
    tenant_id="company_a",
)
```

## Streaming

```python
async for event in orchestrator.process_stream(message, tenant_id="user_123"):
    if event.type == "message_chunk":
        print(event.content, end="")
    elif event.type == "state_change":
        print(f"\n[State: {event.new_state}]")
```

## Agent Pool

```python
# Get active agents
agents = orchestrator.get_active_agents(tenant_id="user_123")

# Cleanup old agents
orchestrator.cleanup_completed_agents(max_age_seconds=3600)
```

## Error Handling

```python
class MyOrchestrator(Orchestrator):

    async def handle_no_match(self, message: str, context: dict):
        """No agent matched"""
        return AgentResult(
            status=AgentStatus.COMPLETED,
            raw_message="I can help with bookings and leave requests."
        )
```

## Best Practices

1. **Use specific triggers** - Better routing accuracy
2. **Set timeouts** - Prevent runaway agents
3. **Implement guards** - Rate limiting, content filtering
4. **Clean up** - Remove completed agents from pool

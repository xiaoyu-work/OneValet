# Tool System

FlowAgents provides a tool system for LLM function calling.

## Quick Start

### Define a Tool

```python
from flowagents import tool, ToolCategory, ToolExecutionContext

@tool(category=ToolCategory.UTILITY)
async def check_availability(
    date: str,
    party_size: int,
    context: ToolExecutionContext = None
) -> dict:
    """
    Check restaurant availability.

    Args:
        date: Date to check (YYYY-MM-DD)
        party_size: Number of guests
    """
    # Your implementation
    available = await check_tables(date, party_size)
    return {"available": available, "date": date}
```

The `@tool` decorator automatically:
- Registers the tool in the global registry
- Generates JSON Schema from type hints
- Handles sync/async functions

### Execute Tools

```python
from flowagents import ToolExecutor, ToolExecutionContext

executor = ToolExecutor(llm_client=my_llm_client)

result = await executor.run_with_tools(
    messages=[{"role": "user", "content": "Check availability for Dec 25, 4 people"}],
    tool_names=["check_availability"],
    context=ToolExecutionContext(user_id="user_123")
)
```

## Tool Categories

```python
from flowagents import ToolCategory

@tool(category=ToolCategory.DATABASE)
async def query_reservations(...): ...

@tool(category=ToolCategory.EMAIL)
async def send_confirmation(...): ...

@tool(category=ToolCategory.CALENDAR)
async def create_reminder(...): ...

@tool(category=ToolCategory.UTILITY)
async def calculate_total(...): ...
```

## Manual Registration

```python
from flowagents import ToolRegistry, ToolDefinition

registry = ToolRegistry.get_instance()

tool_def = ToolDefinition(
    name="my_tool",
    description="Does something useful",
    parameters={
        "type": "object",
        "properties": {
            "input": {"type": "string"}
        },
        "required": ["input"]
    },
    executor=my_function
)

registry.register(tool_def)
```

## Tool Context

Access user info and metadata:

```python
@tool()
async def book_table(
    date: str,
    guests: int,
    context: ToolExecutionContext = None
) -> dict:
    user_id = context.user_id

    # Check user permissions
    if not await can_book(user_id):
        return {"error": "Booking limit reached"}

    return {"success": True, "booking_id": "RES-001"}
```

## Using Tools in Agents

```python
@flowagent(tools=["check_availability", "book_table"])
class BookingAgent(StandardAgent):
    """Agent with tool access"""

    date = InputField("What date?")
    guests = InputField("How many guests?")

    async def on_running(self, msg):
        # Check availability first
        avail = await self.execute_tool(
            "check_availability",
            date=self.date,
            party_size=int(self.guests)
        )

        if not avail["available"]:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Sorry, no tables available."
            )

        # Book it
        result = await self.execute_tool(
            "book_table",
            date=self.date,
            guests=int(self.guests)
        )

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Booked! ID: {result['booking_id']}"
        )
```

## Get Tool Schemas

```python
registry = ToolRegistry.get_instance()

# Get specific tools
schemas = registry.get_tools_schema(["check_availability", "book_table"])

# Get by category
utility_tools = registry.get_tools_by_category(ToolCategory.UTILITY)
```

## Error Handling

```python
@tool()
async def risky_operation(input: str, context: ToolExecutionContext = None) -> dict:
    try:
        result = await perform_operation(input)
        return {"success": True, "data": result}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": "Internal error"}
```

## Best Practices

1. **Always add docstrings** - They become tool descriptions for LLM
2. **Use type hints** - They generate JSON Schema
3. **Return dicts** - Structured data for LLM processing
4. **Handle errors** - Never raise unhandled exceptions
5. **Use context** - For user isolation and logging
6. **Keep tools focused** - One tool = one action

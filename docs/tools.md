# Tool Development Guide

OneValet provides a tool system for LLM function calling.

## Defining a Tool

Use the `@tool` decorator to define and register a tool:

```python
from onevalet import tool, ToolCategory, ToolExecutionContext

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
    available = await check_tables(date, party_size)
    return {"available": available, "date": date}
```

The `@tool` decorator automatically:
- Registers the tool in the global registry
- Generates JSON Schema from type hints and docstrings
- Handles both sync and async functions

### @tool Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | function name | Override the tool name |
| `description` | `str` | docstring first line | Override the tool description |
| `category` | `ToolCategory` or `str` | `ToolCategory.CUSTOM` | Tool category for organization |
| `auto_register` | `bool` | `True` | Whether to auto-register to the global registry |

## Tool Categories

The built-in categories are:

```python
from onevalet import ToolCategory

ToolCategory.UTILITY   # General-purpose utilities
ToolCategory.WEB       # Web-related tools (search, fetch, etc.)
ToolCategory.USER      # User-facing tools
ToolCategory.CUSTOM    # Custom / uncategorized (default)
```

Example usage:

```python
@tool(category=ToolCategory.WEB)
async def search_web(query: str, context: ToolExecutionContext = None) -> dict:
    """Search the web for information."""
    results = await web_search(query)
    return {"results": results}

@tool(category=ToolCategory.UTILITY)
async def calculate_total(prices: list, context: ToolExecutionContext = None) -> dict:
    """Calculate the total of a list of prices."""
    return {"total": sum(prices)}
```

## Tool Context

Access user info and metadata via `ToolExecutionContext`:

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

`ToolExecutionContext` fields:

| Field | Type | Description |
|---|---|---|
| `user_id` | `str` | ID of the current user |
| `account_spec` | `str` (optional) | Account specification (e.g., email) |
| `metadata` | `dict` | Arbitrary metadata dict |
| `credentials` | any (optional) | CredentialStore instance, if available |

Use `context.get(key, default)` to read from the metadata dict.

## Error Handling

Tools should catch their own exceptions and return structured error data. Never let unhandled exceptions propagate:

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

1. **Always add docstrings** - They become tool descriptions for the LLM
2. **Use type hints** - They are used to generate JSON Schema for function calling
3. **Return dicts** - Structured data is easier for the LLM to process
4. **Handle errors gracefully** - Never raise unhandled exceptions from a tool
5. **Use context** - For user isolation, permissions, and logging
6. **Keep tools focused** - One tool = one action

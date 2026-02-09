# MCP Integration

FlowAgents natively supports the Model Context Protocol (MCP) for tool extensibility.

## What is MCP?

MCP (Model Context Protocol) is an open protocol that enables AI models to securely access external tools and data sources. FlowAgents can connect to any MCP-compatible server.

## Quick Start

### Connect to MCP Server

```python
from flowagents import MCPClient

# Connect via stdio
client = MCPClient(
    transport="stdio",
    command="npx",
    args=["@anthropic/mcp-server-filesystem", "/path/to/allowed/dir"]
)

await client.connect()

# List available tools
tools = await client.list_tools()
print(tools)

# Call a tool
result = await client.call_tool("read_file", {"path": "/path/to/file.txt"})
```

### Use with Agent

```python
from flowagents import StandardAgent, flowagent

@flowagent(mcp_servers=["filesystem"])
class FileAgent(StandardAgent):
    """Agent that can read and write files via MCP"""

    async def on_running(self, msg):
        # MCP tools are available
        content = await self.call_mcp_tool(
            server="filesystem",
            tool="read_file",
            args={"path": "/some/file.txt"}
        )
        return self.make_result(...)
```

## Transport Types

### stdio (Recommended)

```python
client = MCPClient(
    transport="stdio",
    command="npx",
    args=["@anthropic/mcp-server-github"]
)
```

### SSE (Server-Sent Events)

```python
client = MCPClient(
    transport="sse",
    url="http://localhost:8080/sse"
)
```

### WebSocket

```python
client = MCPClient(
    transport="websocket",
    url="ws://localhost:8080/ws"
)
```

## Configuration via YAML

```yaml
# flowagents.yaml
mcp_servers:
  filesystem:
    transport: stdio
    command: npx
    args: ["@anthropic/mcp-server-filesystem", "/allowed/path"]

  github:
    transport: stdio
    command: npx
    args: ["@anthropic/mcp-server-github"]
    env:
      GITHUB_TOKEN: ${GITHUB_TOKEN}

  database:
    transport: sse
    url: http://localhost:8080/sse
```

## MCP Tool Provider

Register MCP servers as tool providers:

```python
from flowagents import MCPToolProvider, ToolRegistry

# Create provider from client
provider = MCPToolProvider(client)

# Register MCP tools with the tool registry
registry = ToolRegistry.get_instance()
await provider.register_tools(registry)

# Now MCP tools are available like regular tools
result = await executor.run_with_tools(
    messages=[...],
    tool_names=["mcp__read_file", "mcp__create_issue"]
)
```

## Available MCP Servers

Popular MCP servers you can use:

| Server | Description | Install |
|--------|-------------|---------|
| filesystem | File operations | `@anthropic/mcp-server-filesystem` |
| github | GitHub API | `@anthropic/mcp-server-github` |
| slack | Slack integration | `@anthropic/mcp-server-slack` |
| google-drive | Google Drive | `@anthropic/mcp-server-gdrive` |
| postgres | PostgreSQL | `@anthropic/mcp-server-postgres` |

## Resources and Prompts

MCP also supports resources and prompts:

### Resources

```python
# List resources
resources = await client.list_resources()

# Read a resource
content = await client.read_resource("file:///path/to/file")
```

### Prompts

```python
# List prompts
prompts = await client.list_prompts()

# Get a prompt
prompt = await client.get_prompt("summarize", {"text": "..."})
```

## Connection Management

```python
# Connect with timeout
await client.connect(timeout=30)

# Check connection status
if client.is_connected:
    ...

# Reconnect on failure
await client.reconnect()

# Disconnect
await client.disconnect()
```

## Best Practices

1. **Use stdio transport** - Most reliable for local servers
2. **Set timeouts** - MCP calls can be slow
3. **Handle disconnections** - Implement reconnection logic
4. **Limit permissions** - Only expose necessary directories/resources
5. **Use environment variables** - For sensitive config like tokens
6. **Test servers locally** - Before integrating with agents

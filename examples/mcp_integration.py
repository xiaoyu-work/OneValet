"""
MCP Integration Example - Using MCP tools with FlowAgents

This example shows how to:
1. Create a mock MCP client for testing
2. Register MCP tools with the ToolRegistry
3. Use MCP tools alongside regular tools
"""

import asyncio
from flowagents import (
    ToolRegistry,
    ToolExecutor,
    ToolDefinition,
    ToolExecutionContext,
    ToolCategory,
)
from flowagents.mcp import (
    MockMCPClient,
    MCPToolProvider,
    MCPManager,
    MCPTool,
)


# ============================================================
# Example 1: Using MockMCPClient for testing
# ============================================================

async def example_mock_mcp():
    print("\n" + "=" * 60)
    print("Example 1: Mock MCP Client")
    print("=" * 60 + "\n")

    # Define mock tools
    mock_tools = [
        MCPTool(
            name="read_file",
            description="Read contents of a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            },
            server_name="filesystem"
        ),
        MCPTool(
            name="list_directory",
            description="List files in a directory",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"]
            },
            server_name="filesystem"
        ),
    ]

    # Custom tool handler for mock
    async def mock_handler(name: str, arguments: dict):
        if name == "read_file":
            return {"content": f"Contents of {arguments['path']}", "size": 1234}
        elif name == "list_directory":
            return {"files": ["file1.txt", "file2.txt", "subdir/"]}
        return {"error": "Unknown tool"}

    # Create mock client
    client = MockMCPClient(
        name="filesystem",
        tools=mock_tools,
        tool_handler=mock_handler
    )

    # Connect and register tools
    await client.connect()
    print(f"Connected to: {client.server_name}")

    provider = MCPToolProvider(client)
    tool_names = await provider.register_tools()
    print(f"Registered tools: {tool_names}")

    # Check ToolRegistry
    registry = ToolRegistry.get_instance()
    print(f"\nAll registered tools: {registry.get_all_tool_names()}")

    # Execute a tool directly
    context = ToolExecutionContext(user_id="test-user")
    result = await registry.get_tool("mcp__filesystem__read_file").executor(
        context=context,
        path="/example/test.txt"
    )
    print(f"\nTool result: {result}")

    # Cleanup
    await provider.unregister_tools()
    await client.disconnect()
    print("\nCleanup complete")


# ============================================================
# Example 2: Using MCPManager for multiple servers
# ============================================================

async def example_mcp_manager():
    print("\n" + "=" * 60)
    print("Example 2: MCP Manager (Multiple Servers)")
    print("=" * 60 + "\n")

    # Create multiple mock servers
    filesystem_tools = [
        MCPTool(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            server_name="filesystem"
        ),
    ]

    database_tools = [
        MCPTool(
            name="query",
            description="Execute SQL query",
            input_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
            server_name="database"
        ),
        MCPTool(
            name="insert",
            description="Insert data",
            input_schema={"type": "object", "properties": {"table": {"type": "string"}}},
            server_name="database"
        ),
    ]

    fs_client = MockMCPClient(name="filesystem", tools=filesystem_tools)
    db_client = MockMCPClient(name="database", tools=database_tools)

    # Use MCPManager
    manager = MCPManager()

    await manager.add_server(fs_client)
    await manager.add_server(db_client)

    print(f"Manager: {manager}")
    print(f"All tool names: {manager.get_all_tool_names()}")
    print(f"Filesystem tools: {manager.get_server_tool_names('filesystem')}")
    print(f"Database tools: {manager.get_server_tool_names('database')}")

    # Cleanup
    await manager.disconnect_all()
    print("\nAll servers disconnected")


# ============================================================
# Example 3: Mixing MCP tools with regular tools
# ============================================================

async def example_mixed_tools():
    print("\n" + "=" * 60)
    print("Example 3: Mixed MCP and Regular Tools")
    print("=" * 60 + "\n")

    # Reset registry for clean state
    ToolRegistry.reset()
    registry = ToolRegistry.get_instance()

    # Register a regular FlowAgents tool
    async def get_weather(location: str, context: ToolExecutionContext):
        return {"location": location, "temperature": 72, "condition": "sunny"}

    regular_tool = ToolDefinition(
        name="get_weather",
        description="Get weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        },
        executor=get_weather,
        category=ToolCategory.UTILITY
    )
    registry.register(regular_tool)

    # Add MCP tools
    mcp_tools = [
        MCPTool(
            name="search",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            },
            server_name="web"
        ),
    ]

    async def web_handler(name: str, arguments: dict):
        return {"results": [f"Result for: {arguments.get('query', '')}"]}

    client = MockMCPClient(name="web", tools=mcp_tools, tool_handler=web_handler)
    await client.connect()

    provider = MCPToolProvider(client)
    await provider.register_tools()

    # Now we have both regular and MCP tools
    all_tools = registry.get_all_tool_names()
    print(f"All available tools: {all_tools}")

    # Get schemas for all tools (useful for LLM)
    schemas = registry.get_tools_schema(all_tools)
    print(f"\nTool schemas ({len(schemas)} tools):")
    for schema in schemas:
        print(f"  - {schema['function']['name']}: {schema['function']['description'][:50]}...")

    # Execute tools
    context = ToolExecutionContext(user_id="test-user")

    # Regular tool
    weather_result = await registry.get_tool("get_weather").executor(
        context=context,
        location="San Francisco"
    )
    print(f"\nWeather result: {weather_result}")

    # MCP tool
    search_result = await registry.get_tool("mcp__web__search").executor(
        context=context,
        query="FlowAgents framework"
    )
    print(f"Search result: {search_result}")

    # Cleanup
    await provider.unregister_tools()
    await client.disconnect()


# ============================================================
# Example 4: Custom MCP Client Implementation
# ============================================================

async def example_custom_client():
    print("\n" + "=" * 60)
    print("Example 4: Custom MCP Client (Skeleton)")
    print("=" * 60 + "\n")

    print("""
To implement a real MCP client, extend MCPClient or implement MCPClientProtocol:

    from flowagents.mcp import MCPClient, MCPServerConfig, MCPTool
    from mcp import ClientSession  # Official MCP SDK

    class RealMCPClient(MCPClient):
        def __init__(self, config: MCPServerConfig):
            super().__init__(config)
            self._session: ClientSession = None

        async def _connect_stdio(self):
            # Use official MCP SDK
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env
            )

            self._read, self._write = await stdio_client(params)
            self._session = ClientSession(self._read, self._write)
            await self._session.initialize()

        async def _fetch_tools(self):
            result = await self._session.list_tools()
            return [
                MCPTool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                    server_name=self.server_name
                )
                for tool in result.tools
            ]

        async def _execute_tool(self, name, arguments):
            result = await self._session.call_tool(name, arguments)
            return result.content
    """)


async def main():
    await example_mock_mcp()
    await example_mcp_manager()
    await example_mixed_tools()
    await example_custom_client()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

"""
Tests for FlowAgent Tool Decorator

Tests cover:
- @tool decorator basic usage
- Auto schema generation from type hints
- Docstring parsing for descriptions
- Tool registration
- ToolDiscovery for auto-discovery
"""

import pytest
from typing import List, Optional, Dict, Any

from flowagents.tools import (
    tool,
    get_tool_definition,
    ToolRegistry,
    ToolCategory,
    ToolDefinition,
    ToolDiscovery,
)
from flowagents.tools.decorator import (
    _parse_docstring,
    _get_json_type,
    _generate_parameters_schema,
)


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestDocstringParsing:
    """Tests for docstring parsing"""

    def test_simple_docstring(self):
        """Test parsing simple docstring"""
        docstring = "Send an email to someone."
        result = _parse_docstring(docstring)

        assert result["description"] == "Send an email to someone."
        assert result["params"] == {}
        assert result["returns"] == ""

    def test_google_style_docstring(self):
        """Test parsing Google-style docstring"""
        docstring = """
        Send an email via SMTP

        Args:
            to: Recipient email address
            subject: Email subject line
            body: Email body content

        Returns:
            Success message with delivery status
        """
        result = _parse_docstring(docstring)

        assert "Send an email via SMTP" in result["description"]
        assert result["params"]["to"] == "Recipient email address"
        assert result["params"]["subject"] == "Email subject line"
        assert result["params"]["body"] == "Email body content"
        assert "Success message" in result["returns"]

    def test_docstring_with_type_annotations_in_args(self):
        """Test parsing docstring with types in Args"""
        docstring = """
        Search for items

        Args:
            query (str): Search query string
            limit (int): Maximum results to return
        """
        result = _parse_docstring(docstring)

        assert result["params"]["query"] == "Search query string"
        assert result["params"]["limit"] == "Maximum results to return"

    def test_multiline_param_description(self):
        """Test parsing multiline parameter description"""
        docstring = """
        Do something

        Args:
            config: Configuration object containing
                all the settings for the operation
        """
        result = _parse_docstring(docstring)

        assert "all the settings" in result["params"]["config"]

    def test_empty_docstring(self):
        """Test parsing empty docstring"""
        result = _parse_docstring("")
        assert result["description"] == ""
        assert result["params"] == {}


class TestTypeMapping:
    """Tests for type to JSON Schema mapping"""

    def test_basic_types(self):
        """Test basic type mappings"""
        assert _get_json_type(str) == "string"
        assert _get_json_type(int) == "integer"
        assert _get_json_type(float) == "number"
        assert _get_json_type(bool) == "boolean"
        assert _get_json_type(list) == "array"
        assert _get_json_type(dict) == "object"

    def test_optional_type(self):
        """Test Optional type unwrapping"""
        assert _get_json_type(Optional[str]) == "string"
        assert _get_json_type(Optional[int]) == "integer"

    def test_list_type(self):
        """Test List type"""
        assert _get_json_type(List[str]) == "array"
        assert _get_json_type(List[int]) == "array"

    def test_dict_type(self):
        """Test Dict type"""
        assert _get_json_type(Dict[str, Any]) == "object"


class TestParameterSchemaGeneration:
    """Tests for parameter schema generation"""

    def test_basic_function(self):
        """Test schema generation for basic function"""
        def test_func(name: str, count: int) -> str:
            pass

        schema = _generate_parameters_schema(test_func, {})

        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert "count" in schema["properties"]
        assert schema["properties"]["count"]["type"] == "integer"
        assert "name" in schema["required"]
        assert "count" in schema["required"]

    def test_optional_params(self):
        """Test schema with optional parameters"""
        def test_func(name: str, count: int = 10) -> str:
            pass

        schema = _generate_parameters_schema(test_func, {})

        assert "name" in schema["required"]
        assert "count" not in schema["required"]
        assert schema["properties"]["count"].get("default") == 10

    def test_with_param_descriptions(self):
        """Test schema with parameter descriptions from docstring"""
        def test_func(query: str, limit: int = 20) -> str:
            pass

        param_docs = {
            "query": "Search query string",
            "limit": "Maximum results"
        }
        schema = _generate_parameters_schema(test_func, param_docs)

        assert schema["properties"]["query"]["description"] == "Search query string"
        assert schema["properties"]["limit"]["description"] == "Maximum results"


# =============================================================================
# Test @tool Decorator
# =============================================================================

class TestToolDecorator:
    """Tests for @tool decorator"""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test"""
        ToolRegistry.reset()
        yield
        ToolRegistry.reset()

    def test_basic_decorator(self):
        """Test basic @tool decorator"""
        @tool()
        async def greet(name: str) -> str:
            """Greet someone by name"""
            return f"Hello, {name}!"

        # Check tool definition attached
        tool_def = get_tool_definition(greet)
        assert tool_def is not None
        assert tool_def.name == "greet"
        assert "Greet someone" in tool_def.description

        # Check registered
        registry = ToolRegistry.get_instance()
        assert registry.has_tool("greet")

    def test_custom_name(self):
        """Test @tool with custom name"""
        @tool(name="custom_greet")
        async def greet(name: str) -> str:
            """Greet someone"""
            return f"Hello, {name}!"

        tool_def = get_tool_definition(greet)
        assert tool_def.name == "custom_greet"

        registry = ToolRegistry.get_instance()
        assert registry.has_tool("custom_greet")

    def test_custom_description(self):
        """Test @tool with custom description"""
        @tool(description="Custom greeting function")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        tool_def = get_tool_definition(greet)
        assert tool_def.description == "Custom greeting function"

    def test_category(self):
        """Test @tool with category"""
        @tool(category=ToolCategory.WEB)
        async def search(query: str) -> str:
            """Search the web"""
            return "results"

        tool_def = get_tool_definition(search)
        assert tool_def.category == ToolCategory.WEB

    def test_category_string(self):
        """Test @tool with category as string"""
        @tool(category="web")
        async def search(query: str) -> str:
            """Search the web"""
            return "results"

        tool_def = get_tool_definition(search)
        assert tool_def.category == ToolCategory.WEB

    def test_auto_register_false(self):
        """Test @tool with auto_register=False"""
        @tool(auto_register=False)
        async def private_tool(x: int) -> int:
            """Private tool"""
            return x * 2

        tool_def = get_tool_definition(private_tool)
        assert tool_def is not None

        # Should not be in registry
        registry = ToolRegistry.get_instance()
        assert not registry.has_tool("private_tool")

    def test_parameters_from_type_hints(self):
        """Test parameter schema from type hints"""
        @tool()
        async def complex_func(
            name: str,
            count: int,
            enabled: bool = True,
            tags: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """Complex function with many params"""
            return {}

        tool_def = get_tool_definition(complex_func)
        params = tool_def.parameters

        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["count"]["type"] == "integer"
        assert params["properties"]["enabled"]["type"] == "boolean"
        assert "name" in params["required"]
        assert "count" in params["required"]
        assert "enabled" not in params["required"]

    def test_docstring_param_descriptions(self):
        """Test parameter descriptions from docstring"""
        @tool()
        async def search_emails(
            query: str,
            limit: int = 20
        ) -> List[dict]:
            """
            Search emails by query

            Args:
                query: Search query string
                limit: Maximum number of results

            Returns:
                List of matching emails
            """
            return []

        tool_def = get_tool_definition(search_emails)
        params = tool_def.parameters

        assert params["properties"]["query"]["description"] == "Search query string"
        assert params["properties"]["limit"]["description"] == "Maximum number of results"

    def test_sync_function(self):
        """Test @tool with sync function"""
        @tool()
        def sync_tool(x: int) -> int:
            """Sync tool"""
            return x * 2

        tool_def = get_tool_definition(sync_tool)
        assert tool_def is not None
        assert tool_def.name == "sync_tool"

    def test_openai_schema(self):
        """Test tool generates valid OpenAI schema"""
        @tool()
        async def get_weather(location: str, unit: str = "celsius") -> str:
            """
            Get weather for a location

            Args:
                location: City name or zip code
                unit: Temperature unit (celsius or fahrenheit)
            """
            return "sunny"

        tool_def = get_tool_definition(get_weather)
        schema = tool_def.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "Get weather" in schema["function"]["description"]
        assert "location" in schema["function"]["parameters"]["properties"]


# =============================================================================
# Test ToolDiscovery
# =============================================================================

class TestToolDiscovery:
    """Tests for ToolDiscovery"""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test"""
        ToolRegistry.reset()
        yield
        ToolRegistry.reset()

    def test_discovery_initialization(self):
        """Test ToolDiscovery initialization"""
        discovery = ToolDiscovery()
        assert discovery.registry is not None
        assert discovery.get_discovered_tools() == []

    def test_scan_nonexistent_module(self):
        """Test scanning non-existent module"""
        discovery = ToolDiscovery()
        count = discovery.scan_module("nonexistent.module.path")
        assert count == 0

    def test_get_discovered_tools(self):
        """Test getting discovered tools list"""
        # Register a tool first
        @tool()
        async def test_tool(x: int) -> int:
            """Test"""
            return x

        discovery = ToolDiscovery()
        # Since tool is already registered, discovery won't find new ones
        # but we can test the interface
        tools = discovery.get_discovered_tools()
        assert isinstance(tools, list)


# =============================================================================
# Test Integration with ToolRegistry
# =============================================================================

class TestToolRegistryIntegration:
    """Tests for integration with ToolRegistry"""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test"""
        ToolRegistry.reset()
        yield
        ToolRegistry.reset()

    def test_multiple_tools_registration(self):
        """Test registering multiple tools"""
        @tool()
        async def tool_a(x: int) -> int:
            """Tool A"""
            return x

        @tool()
        async def tool_b(y: str) -> str:
            """Tool B"""
            return y

        @tool()
        async def tool_c(z: bool) -> bool:
            """Tool C"""
            return z

        registry = ToolRegistry.get_instance()
        assert registry.has_tool("tool_a")
        assert registry.has_tool("tool_b")
        assert registry.has_tool("tool_c")
        assert len(registry) >= 3

    def test_get_tools_schema(self):
        """Test getting schemas for multiple tools"""
        @tool()
        async def tool_x(a: str) -> str:
            """Tool X"""
            return a

        @tool()
        async def tool_y(b: int) -> int:
            """Tool Y"""
            return b

        registry = ToolRegistry.get_instance()
        schemas = registry.get_tools_schema(["tool_x", "tool_y"])

        assert len(schemas) == 2
        names = [s["function"]["name"] for s in schemas]
        assert "tool_x" in names
        assert "tool_y" in names

    def test_get_tools_by_category(self):
        """Test getting tools by category"""
        @tool(category=ToolCategory.WEB)
        async def web_tool(url: str) -> str:
            """Web tool"""
            return url

        @tool(category=ToolCategory.UTILITY)
        async def util_tool(data: str) -> str:
            """Utility tool"""
            return data

        registry = ToolRegistry.get_instance()
        web_tools = registry.get_tools_by_category(ToolCategory.WEB)

        assert any(t.name == "web_tool" for t in web_tools)

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test that decorated tools can be executed"""
        @tool()
        async def add_numbers(a: int, b: int) -> int:
            """Add two numbers"""
            return a + b

        # Get tool and execute
        registry = ToolRegistry.get_instance()
        tool_def = registry.get_tool("add_numbers")

        result = await tool_def.executor(a=5, b=3)
        assert result == 8


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

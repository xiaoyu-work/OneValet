"""
FlowAgent Tool Registry - Central registry for all available tools
"""

import logging
import threading
from typing import Dict, List, Optional, Union

from .models import ToolDefinition, ToolCategory

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Singleton registry for managing all available tools

    Usage:
        registry = ToolRegistry.get_instance()
        registry.register(tool_definition)
        tools = registry.get_tools_schema(["search_web", "get_weather"])

    Example:
        # Register a tool
        registry = ToolRegistry.get_instance()
        registry.register(ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={...},
            executor=get_weather_func
        ))

        # Get tool schemas for LLM
        schemas = registry.get_tools_schema(["get_weather"])
    """

    _instance: Optional["ToolRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls):
        # Note: Lock is handled in get_instance(), not here
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._tools = {}
            cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset registry (for testing)"""
        with cls._lock:
            cls._instance = None

    def register(self, tool: ToolDefinition) -> None:
        """
        Register a tool definition

        Args:
            tool: ToolDefinition to register

        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} ({tool.category.value})")

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """
        Get a tool definition by name

        Args:
            name: Tool name

        Returns:
            ToolDefinition or None if not found
        """
        return self._tools.get(name)

    def get_tools(self, names: List[str]) -> List[ToolDefinition]:
        """
        Get multiple tool definitions by name

        Args:
            names: List of tool names

        Returns:
            List of ToolDefinitions (skips unknown tools)
        """
        tools = []
        for name in names:
            tool = self._tools.get(name)
            if tool:
                tools.append(tool)
            else:
                logger.warning(f"Unknown tool requested: {name}")
        return tools

    def get_tools_schema(self, names: List[str]) -> List[Dict]:
        """
        Get OpenAI-format tool schemas for specified tools

        Args:
            names: List of tool names

        Returns:
            List of tool schemas in OpenAI format
        """
        schemas = []
        for name in names:
            tool = self._tools.get(name)
            if tool:
                schemas.append(tool.to_openai_schema())
            else:
                logger.warning(f"Unknown tool requested for schema: {name}")
        return schemas

    def get_all_tools(self) -> List[ToolDefinition]:
        """Get all registered tools"""
        return list(self._tools.values())

    def get_all_tool_names(self) -> List[str]:
        """Get all registered tool names"""
        return list(self._tools.keys())

    def get_tools_by_category(
        self,
        category: Union[ToolCategory, str]
    ) -> List[ToolDefinition]:
        """
        Get all tools in a specific category

        Args:
            category: ToolCategory enum or string value

        Returns:
            List of ToolDefinitions in that category
        """
        if isinstance(category, str):
            return [tool for tool in self._tools.values()
                    if tool.category.value == category]
        return [tool for tool in self._tools.values()
                if tool.category == category]

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered"""
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={len(self._tools)}>"

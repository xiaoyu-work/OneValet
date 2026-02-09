"""
FlowAgent Tool Decorator - Auto-register tools with @tool decorator

This module provides:
- @tool decorator for auto-registration
- Auto schema generation from type hints and docstrings
- Support for sync and async functions
- Auto-discovery of tool modules

Usage:
    from flowagents import tool

    @tool()
    async def send_email(to: str, subject: str, body: str) -> str:
        '''
        Send an email via SMTP

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body content

        Returns:
            Success message
        '''
        # Implementation
        return f"Email sent to {to}"

    # Tool is automatically registered to ToolRegistry
"""

import inspect
import logging
import importlib
import pkgutil
import re
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_type_hints,
    get_origin,
    get_args,
)

from .models import ToolDefinition, ToolCategory, ToolExecutionContext
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


# Type mapping from Python types to JSON Schema types
TYPE_MAPPING = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _get_json_type(python_type: Type) -> str:
    """Convert Python type to JSON Schema type"""
    # Handle Optional, List, Dict etc.
    origin = get_origin(python_type)

    if origin is Union:
        # Optional[X] is Union[X, None]
        args = get_args(python_type)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _get_json_type(non_none[0])
        return "string"  # Fallback

    if origin is list or origin is List:
        return "array"

    if origin is dict or origin is Dict:
        return "object"

    return TYPE_MAPPING.get(python_type, "string")


def _get_json_schema_for_type(python_type: Type) -> Dict[str, Any]:
    """Generate JSON Schema for a Python type"""
    origin = get_origin(python_type)

    if origin is Union:
        args = get_args(python_type)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _get_json_schema_for_type(non_none[0])

    if origin is list or origin is List:
        args = get_args(python_type)
        if args:
            return {
                "type": "array",
                "items": _get_json_schema_for_type(args[0])
            }
        return {"type": "array"}

    if origin is dict or origin is Dict:
        return {"type": "object"}

    return {"type": _get_json_type(python_type)}


def _parse_docstring(docstring: str) -> Dict[str, Any]:
    """
    Parse Google-style docstring to extract description and parameter docs.

    Returns:
        {
            "description": "Main description",
            "params": {"param_name": "param description", ...},
            "returns": "Return description"
        }
    """
    if not docstring:
        return {"description": "", "params": {}, "returns": ""}

    lines = docstring.strip().split("\n")
    result = {
        "description": "",
        "params": {},
        "returns": ""
    }

    # States: description, args, returns
    state = "description"
    current_param = None
    description_lines = []

    for line in lines:
        line_stripped = line.strip()

        # Check for section headers
        if line_stripped.lower() in ("args:", "arguments:", "parameters:"):
            state = "args"
            continue
        elif line_stripped.lower() in ("returns:", "return:"):
            state = "returns"
            continue
        elif line_stripped.lower() in ("raises:", "example:", "examples:", "note:", "notes:"):
            state = "other"
            continue

        if state == "description":
            if line_stripped:
                description_lines.append(line_stripped)
            elif description_lines:
                # Empty line ends description
                result["description"] = " ".join(description_lines)
                state = "post_description"

        elif state == "args":
            # Match "param_name: description" or "param_name (type): description"
            match = re.match(r"^\s*(\w+)(?:\s*\([^)]+\))?\s*:\s*(.*)$", line)
            if match:
                current_param = match.group(1)
                result["params"][current_param] = match.group(2).strip()
            elif current_param and line_stripped:
                # Continuation of previous param description
                result["params"][current_param] += " " + line_stripped

        elif state == "returns":
            if line_stripped:
                if result["returns"]:
                    result["returns"] += " " + line_stripped
                else:
                    result["returns"] = line_stripped

    # If we never hit a section, use all lines as description
    if not result["description"] and description_lines:
        result["description"] = " ".join(description_lines)

    return result


def _generate_parameters_schema(
    func: Callable,
    param_docs: Dict[str, str]
) -> Dict[str, Any]:
    """
    Generate JSON Schema for function parameters from type hints.
    """
    sig = inspect.signature(func)
    hints = {}

    try:
        hints = get_type_hints(func)
    except Exception:
        pass  # Type hints may not be available

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        # Skip special parameters
        if name in ("self", "cls", "context"):
            continue
        if isinstance(param.annotation, type) and issubclass(param.annotation, ToolExecutionContext):
            continue

        # Get type from hints or annotation
        param_type = hints.get(name, param.annotation)
        if param_type is inspect.Parameter.empty:
            param_type = str  # Default to string

        # Build property schema
        prop_schema = _get_json_schema_for_type(param_type)

        # Add description from docstring
        if name in param_docs:
            prop_schema["description"] = param_docs[name]

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif param.default is not None:
            # Add default value
            prop_schema["default"] = param.default

        properties[name] = prop_schema

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: Union[ToolCategory, str] = ToolCategory.CUSTOM,
    auto_register: bool = True
) -> Callable:
    """
    Decorator to define and register a tool function.

    The decorator extracts:
    - Tool name from function name (or override with `name`)
    - Description from docstring (or override with `description`)
    - Parameters schema from type hints and docstring

    Args:
        name: Tool name (default: function name)
        description: Tool description (default: docstring first line)
        category: Tool category for organization
        auto_register: Whether to auto-register to global ToolRegistry

    Returns:
        Decorated function

    Note:
        Approval logic is handled at the Agent level, not Tool level.
        Use `requires_approval` in agent config (flowagents.yaml) instead.

    Example:
        @tool(category="email")
        async def send_email(to: str, subject: str, body: str) -> str:
            '''
            Send an email via SMTP

            Args:
                to: Recipient email address
                subject: Email subject
                body: Email body content

            Returns:
                Success message
            '''
            await smtp_client.send(to=to, subject=subject, body=body)
            return f"Email sent to {to}"
    """

    def decorator(func: Callable) -> Callable:
        # Get tool name
        tool_name = name or func.__name__

        # Parse docstring
        docstring_info = _parse_docstring(func.__doc__ or "")

        # Get description
        tool_description = description or docstring_info["description"] or f"Execute {tool_name}"

        # Generate parameters schema
        parameters_schema = _generate_parameters_schema(
            func,
            docstring_info["params"]
        )

        # Get category
        if isinstance(category, str):
            try:
                tool_category = ToolCategory(category)
            except ValueError:
                tool_category = ToolCategory.CUSTOM
        else:
            tool_category = category

        # Create tool definition
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=parameters_schema,
            executor=func,
            category=tool_category,
        )

        # Auto-register if enabled
        if auto_register:
            registry = ToolRegistry.get_instance()
            registry.register(tool_def)

        # Attach tool definition to function for introspection
        func._tool_definition = tool_def

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        # Copy tool definition to wrapper
        async_wrapper._tool_definition = tool_def

        # Return appropriate wrapper
        if inspect.iscoroutinefunction(func):
            return func  # Already async
        return async_wrapper

    return decorator


def get_tool_definition(func: Callable) -> Optional[ToolDefinition]:
    """Get the ToolDefinition attached to a decorated function"""
    return getattr(func, "_tool_definition", None)


class ToolDiscovery:
    """
    Auto-discover and register tools from Python modules.

    Usage:
        discovery = ToolDiscovery()
        discovery.scan_module("myapp.tools")
        discovery.scan_paths(["myapp.tools", "myapp.custom_tools"])
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or ToolRegistry.get_instance()
        self._discovered_tools: List[str] = []

    def scan_module(self, module_path: str) -> int:
        """
        Scan a module and register all @tool decorated functions.

        Args:
            module_path: Dot-separated module path (e.g., "myapp.tools")

        Returns:
            Number of tools discovered
        """
        count = 0

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.warning(f"Failed to import module {module_path}: {e}")
            return 0

        # Scan all attributes
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and hasattr(obj, "_tool_definition"):
                tool_def = obj._tool_definition
                if tool_def.name not in self._discovered_tools:
                    # Tool may already be registered by decorator
                    if not self.registry.has_tool(tool_def.name):
                        self.registry.register(tool_def)
                    self._discovered_tools.append(tool_def.name)
                    count += 1
                    logger.debug(f"Discovered tool: {tool_def.name} from {module_path}")

        return count

    def scan_package(self, package_path: str, recursive: bool = True) -> int:
        """
        Scan a package and all submodules for tools.

        Args:
            package_path: Dot-separated package path (e.g., "myapp.tools")
            recursive: Whether to scan subpackages

        Returns:
            Number of tools discovered
        """
        count = 0

        try:
            package = importlib.import_module(package_path)
        except ImportError as e:
            logger.warning(f"Failed to import package {package_path}: {e}")
            return 0

        # Scan the package itself
        count += self.scan_module(package_path)

        # Scan submodules
        if hasattr(package, "__path__"):
            for importer, modname, ispkg in pkgutil.walk_packages(
                package.__path__,
                prefix=package_path + ".",
                onerror=lambda x: None
            ):
                if recursive or not ispkg:
                    count += self.scan_module(modname)

        return count

    def scan_paths(self, paths: List[str]) -> int:
        """
        Scan multiple module/package paths.

        Args:
            paths: List of module paths to scan

        Returns:
            Total number of tools discovered
        """
        count = 0
        for path in paths:
            count += self.scan_package(path)
        return count

    def get_discovered_tools(self) -> List[str]:
        """Get list of discovered tool names"""
        return self._discovered_tools.copy()


# Convenience function for module-level tool registration
def register_tools_from_module(module_path: str) -> int:
    """
    Convenience function to register all tools from a module.

    Args:
        module_path: Dot-separated module path

    Returns:
        Number of tools registered
    """
    discovery = ToolDiscovery()
    return discovery.scan_module(module_path)


def register_tools_from_paths(paths: List[str]) -> int:
    """
    Convenience function to register tools from multiple paths.

    Args:
        paths: List of module paths

    Returns:
        Number of tools registered
    """
    discovery = ToolDiscovery()
    return discovery.scan_paths(paths)

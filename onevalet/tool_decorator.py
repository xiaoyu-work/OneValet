"""
@tool decorator — auto-generate AgentTool instances from typed async functions.

Inspects the function signature and type hints to build JSON Schema for
parameters, then wraps the function into the executor signature expected
by AgentTool (``async def executor(args: dict, context: AgentToolContext) -> str``).

Usage::

    from typing import Annotated
    from onevalet.tool_decorator import tool
    from onevalet.models import AgentToolContext

    @tool
    async def search_emails(
        query: Annotated[str, "Search keywords"] = "",
        sender: Annotated[str | None, "Filter by sender"] = None,
        max_results: Annotated[int, "Max results to return"] = 15,
        *,
        context: AgentToolContext,
    ) -> str:
        \"\"\"Search emails across connected accounts.\"\"\"
        ...

    # search_emails is now an AgentTool instance
    # search_emails.name == "search_emails"
    # search_emails.parameters == {"type": "object", "properties": {...}, "required": []}

    @tool(needs_approval=True, risk_level="write")
    async def delete_email(
        email_id: Annotated[str, "ID of the email to delete"],
        *,
        context: AgentToolContext,
    ) -> str:
        \"\"\"Delete an email by ID.\"\"\"
        ...
"""

from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .models import AgentTool, AgentToolContext

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NoneType = type(None)


def _is_optional(annotation: Any) -> bool:
    """Return True if *annotation* is ``Optional[X]`` (i.e. ``Union[X, None]``)."""
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        return _NoneType in args
    return False


def _unwrap_optional(annotation: Any) -> Any:
    """Given ``Optional[X]``, return ``X``."""
    args = get_args(annotation)
    non_none = [a for a in args if a is not _NoneType]
    return non_none[0] if len(non_none) == 1 else annotation


def _extract_base_type(annotation: Any) -> Any:
    """Unwrap ``Annotated[T, ...]`` to get ``T``."""
    # typing.Annotated has origin typing.Annotated (3.11+) or
    # typing_extensions.Annotated
    origin = get_origin(annotation)
    if origin is not None:
        try:
            from typing import Annotated as _Annotated  # 3.9+
        except ImportError:
            _Annotated = None
        if _Annotated is not None and origin is _Annotated:
            return get_args(annotation)[0]
        # typing_extensions fallback
        if getattr(origin, "__name__", None) == "Annotated":
            return get_args(annotation)[0]
    return annotation


def _extract_annotated_description(annotation: Any) -> Optional[str]:
    """If *annotation* is ``Annotated[T, "desc"]``, return ``"desc"``."""
    origin = get_origin(annotation)
    if origin is None:
        return None
    try:
        from typing import Annotated as _Annotated
    except ImportError:
        _Annotated = None
    is_annotated = (_Annotated is not None and origin is _Annotated) or getattr(
        origin, "__name__", None
    ) == "Annotated"
    if not is_annotated:
        return None
    args = get_args(annotation)
    for a in args[1:]:
        if isinstance(a, str):
            return a
    return None


def _python_type_to_json_schema(annotation: Any) -> Dict[str, Any]:
    """Map a Python type annotation to a JSON Schema dict."""
    # Unwrap Annotated
    base = _extract_base_type(annotation)

    # Handle Optional[X] / Union[X, None]
    if _is_optional(base):
        inner = _unwrap_optional(base)
        return _python_type_to_json_schema(inner)

    # Also handle X | None (Python 3.10+ union syntax via types.UnionType)
    origin = get_origin(base)
    if origin is Union:
        args = [a for a in get_args(base) if a is not _NoneType]
        if len(args) == 1:
            return _python_type_to_json_schema(args[0])

    # Primitives
    if base is str:
        return {"type": "string"}
    if base is int:
        return {"type": "integer"}
    if base is float:
        return {"type": "number"}
    if base is bool:
        return {"type": "boolean"}

    # list / List / List[X]
    if base is list or origin is list:
        schema: Dict[str, Any] = {"type": "array"}
        args = get_args(base)
        if args:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema

    # dict / Dict / Dict[K, V]
    if base is dict or origin is dict:
        return {"type": "object"}

    # Fallback
    return {"type": "string"}


def _build_json_schema(func: Callable) -> Dict[str, Any]:
    """Build a full JSON Schema ``{"type": "object", ...}`` from *func*'s signature."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        hints = {}

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        # Skip injected context
        if name == "context":
            continue
        # Skip *args / **kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = str  # default to string

        prop_schema = _python_type_to_json_schema(annotation)

        # Extract Annotated description
        desc = _extract_annotated_description(annotation)
        if desc:
            prop_schema["description"] = desc

        properties[name] = prop_schema

        # Determine if required: no default AND not Optional
        has_default = param.default is not inspect.Parameter.empty
        is_opt = _is_optional(_extract_base_type(annotation))
        if not has_default and not is_opt:
            required.append(name)

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _build_wrapper(func: Callable) -> Callable:
    """Create an executor wrapper with the AgentTool-expected signature.

    Returns an ``async def wrapper(args: dict, context: AgentToolContext) -> str``
    that unpacks *args* into keyword arguments for *func*.
    """
    sig = inspect.signature(func)

    # Collect parameter defaults for filling in missing args
    defaults: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "context":
            continue
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default

    async def wrapper(args: Dict[str, Any], context: AgentToolContext) -> str:
        kwargs: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "context":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name in args:
                kwargs[name] = args[name]
            elif name in defaults:
                kwargs[name] = defaults[name]
        return await func(**kwargs, context=context)

    return wrapper


# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------


def tool(
    func: Optional[Callable] = None,
    *,
    needs_approval: bool = False,
    risk_level: str = "read",
    category: str = "utility",
    get_preview: Optional[Callable] = None,
    name: Optional[str] = None,
) -> Any:
    """Decorator that converts a typed async function into an :class:`AgentTool`.

    Supports both bare ``@tool`` and parameterised ``@tool(needs_approval=True)``
    usage.  The decorated name is replaced by an ``AgentTool`` instance.
    """

    def _make_tool(fn: Callable) -> AgentTool:
        tool_name = name or fn.__name__
        # First line of docstring as description
        doc = inspect.getdoc(fn) or ""
        description = doc.split("\n")[0].strip() if doc else tool_name

        return AgentTool(
            name=tool_name,
            description=description,
            parameters=_build_json_schema(fn),
            executor=_build_wrapper(fn),
            needs_approval=needs_approval,
            risk_level=risk_level,
            category=category,
            get_preview=get_preview,
        )

    if func is not None:
        # Called as @tool (no parentheses)
        return _make_tool(func)

    # Called as @tool(...) — return the actual decorator
    return _make_tool

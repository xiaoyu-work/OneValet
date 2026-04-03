"""Tool management mixin for the Orchestrator.

Handles building tool schemas, executing tool calls, and managing
builtin tools.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..models import AgentTool, AgentToolContext
from .agent_tool import execute_agent_tool
from .constants import TOOL_RESULT_HARD_CAP_CHARS
from .react_config import COMPLETE_TASK_SCHEMA

logger = logging.getLogger(__name__)


class ToolManagerMixin:
    """Mixin providing tool schema building, execution, and management.

    Expects the following attributes on ``self`` (provided by Orchestrator):
    - ``_agent_registry``
    - ``credential_store``
    - ``database``
    - ``_tool_policy_filter``
    - ``_execution_policy``
    - ``builtin_tools``
    """

    async def _build_tool_schemas(
        self,
        tenant_id: str,
        domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build combined tool schemas: builtin tools + domain-filtered agent-tools.

        Args:
            tenant_id: Tenant identifier for credential filtering.
            domains: List of domains to load agent tools for.
                If ``None`` or contains ``"all"``, loads all agent tools.
        """
        schemas: List[Dict[str, Any]] = []

        if not self._agent_registry:
            return schemas

        # Builtin tools (orchestrator-level, always included)
        for tool in getattr(self, 'builtin_tools', []):
            schemas.append(tool.to_openai_schema())

        # Agent-tools: domain-filtered
        if domains and "all" not in domains:
            agent_tool_schemas = await self._agent_registry.get_domain_agent_tool_schemas(
                domains=domains,
                tenant_id=tenant_id,
                credential_store=self.credential_store,
            )
        else:
            agent_tool_schemas = await self._agent_registry.get_all_agent_tool_schemas(
                tenant_id=tenant_id,
                credential_store=self.credential_store,
            )

        schemas.extend(agent_tool_schemas)

        # Always inject complete_task as the last tool
        schemas.append(COMPLETE_TASK_SCHEMA)

        # Apply tool policy filter if configured
        if self._tool_policy_filter:
            schemas = self._tool_policy_filter.filter_tools(schemas, tenant_id=tenant_id)

        agent_tool_names = [
            s.get("function", {}).get("name", s.get("name", "?"))
            for s in agent_tool_schemas
        ]
        logger.info(
            f"[Tools] {len(schemas)} total available "
            f"(agents={len(agent_tool_schemas)}, domains={domains or ['all']}, "
            f"agent_tools={agent_tool_names})"
        )

        return schemas

    def _build_builtin_tools(self) -> List[AgentTool]:
        """Build the orchestrator's builtin tools as AgentTool instances.

        These are lightweight tools the orchestrator calls directly
        (not via an agent).
        """
        from ..builtin_agents.tools.google_search import (
            google_search_executor, GOOGLE_SEARCH_SCHEMA,
        )
        from ..builtin_agents.tools.web_fetch import (
            web_fetch_executor, WEB_FETCH_SCHEMA,
        )
        from ..builtin_agents.tools.download_image import (
            download_image_executor, DOWNLOAD_IMAGE_SCHEMA,
        )
        from ..builtin_agents.tools.important_dates import IMPORTANT_DATES_TOOL_DEFS
        from ..builtin_agents.tools.user_tools import (
            get_user_accounts_executor, get_user_profile_executor,
            GET_USER_ACCOUNTS_SCHEMA, GET_USER_PROFILE_SCHEMA,
        )
        from ..builtin_agents.location import (
            get_user_location_executor, GET_USER_LOCATION_SCHEMA,
            set_location_reminder_executor, SET_LOCATION_REMINDER_SCHEMA,
        )
        from ..builtin_agents.trip_planner.travel_tools import (
            get_weather,
        )

        tools: List[AgentTool] = []

        # Google search
        tools.append(AgentTool(
            name="google_search",
            description="Search the web using Google. Returns titles, URLs, and snippets of top results. Set search_type to 'image' to search for images — thumbnails will be shown for you to review and pick the best one.",
            parameters=GOOGLE_SEARCH_SCHEMA,
            executor=google_search_executor,
            category="web",
            renderer="table",
        ))

        # Web fetch
        tools.append(AgentTool(
            name="web_fetch",
            description="Fetch a URL and extract its readable content as text. Use this to read articles, documentation, or any web page. Returns the main content with boilerplate removed.",
            parameters=WEB_FETCH_SCHEMA,
            executor=web_fetch_executor,
            category="web",
            renderer="markdown",
        ))

        # Download image (for storing selected images locally)
        tools.append(AgentTool(
            name="download_image",
            description="Download a full-resolution image from a URL and return it as base64 data for local storage. Use this AFTER selecting the best image from search results to deliver the actual image data to the user's device.",
            parameters=DOWNLOAD_IMAGE_SCHEMA,
            executor=download_image_executor,
            category="web",
            renderer="image",
        ))

        # Important dates (6 tools)
        for td in IMPORTANT_DATES_TOOL_DEFS:
            tools.append(AgentTool(
                name=td["name"],
                description=td["description"],
                parameters=td["parameters"],
                executor=td["executor"],
                category="user",
            ))

        # User tools
        tools.append(AgentTool(
            name="get_user_accounts",
            description="Get the user's connected accounts (email, calendar). Use this when user asks about their connected accounts.",
            parameters=GET_USER_ACCOUNTS_SCHEMA,
            executor=get_user_accounts_executor,
            category="user",
        ))
        tools.append(AgentTool(
            name="get_user_profile",
            description="Get the user's profile information (name, email, phone, timezone). Use this when you need to know about the user.",
            parameters=GET_USER_PROFILE_SCHEMA,
            executor=get_user_profile_executor,
            category="user",
        ))

        # Location tools
        tools.append(AgentTool(
            name="get_user_location",
            description="Get the user's current location (latitude, longitude, and place name). Use when you need the user's precise current location for finding nearby places, calculating distances, or checking proximity.",
            parameters=GET_USER_LOCATION_SCHEMA,
            executor=get_user_location_executor,
            category="location",
        ))
        tools.append(AgentTool(
            name="set_location_reminder",
            description="Set a location-based reminder that notifies the user when they arrive near a specific place. Use when the user says things like 'remind me to X when I'm near Y' or 'notify me when I get to Z'.",
            parameters=SET_LOCATION_REMINDER_SCHEMA,
            executor=set_location_reminder_executor,
            category="location",
            read_only=False,
            mutates_user_data=True,
            idempotent=False,
        ))

        # Weather (reuses the @tool-decorated AgentTool from trip_planner)
        tools.append(get_weather)

        # Action history - lets LLM recall recent user actions
        from ..builtin_agents.tools.action_history import (
            recall_recent_actions_executor, RECALL_RECENT_ACTIONS_SCHEMA,
        )
        tools.append(AgentTool(
            name="recall_recent_actions",
            description="Look up the user's recent actions and tool executions. Use this when the user asks about what they did recently, past activity, or references a previous action (e.g. 'what did I do yesterday?', 'did my email send?', 'what was that flight I searched?').",
            parameters=RECALL_RECENT_ACTIONS_SCHEMA,
            executor=recall_recent_actions_executor,
            category="user",
        ))

        return tools

    def _build_tool_metadata(self, metadata: Optional[dict] = None) -> dict:
        """Build metadata dict for regular tool execution context."""
        # Start with request-level metadata (contains location, timezone, etc.)
        meta = dict(metadata or {})
        if self.database:
            from onevalet.builtin_agents.digest.important_dates_repo import ImportantDatesRepository
            meta["important_dates_store"] = ImportantDatesRepository(self.database)
            meta["database"] = self.database
        return meta

    async def _execute_with_timeout(
        self,
        tool_call: Any,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        request_tools: Optional[List] = None,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a single tool/agent-tool with timeout."""
        import asyncio

        tool_name = tool_call.name
        is_agent = self._is_agent_tool(tool_name)
        timeout = (
            self._react_config.agent_tool_execution_timeout
            if is_agent
            else self._react_config.tool_execution_timeout
        )

        try:
            return await asyncio.wait_for(
                self._execute_single(
                    tool_call, tenant_id,
                    metadata=metadata, request_tools=request_tools,
                    request_context=request_context,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            kind = "Agent-Tool" if is_agent else "Tool"
            raise TimeoutError(f"{kind} '{tool_name}' timed out after {timeout}s")

    async def _execute_single(
        self,
        tool_call: Any,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        request_tools: Optional[List] = None,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Dispatch to agent-tool or regular tool execution."""
        tool_name = tool_call.name
        try:
            args = tool_call.arguments if isinstance(tool_call.arguments, dict) else json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            return (
                f"Error: Failed to parse arguments for tool '{tool_name}': {e}. "
                "Please retry with valid JSON arguments."
            )

        blocked = self._check_schema_policy(tool_name, tenant_id=tenant_id)
        if blocked is not None:
            return blocked

        if self._is_agent_tool(tool_name):
            # Agent-Tool execution
            task_instruction = args.pop("task_instruction", "")
            return await execute_agent_tool(
                self,
                agent_type=tool_name,
                tenant_id=tenant_id,
                tool_call_args=args,
                task_instruction=task_instruction,
                request_context=request_context,
            )
        else:
            # Builtin tool execution — use request_tools (local copy) if available
            tools = request_tools if request_tools is not None else getattr(self, 'builtin_tools', [])
            tool = next((t for t in tools if t.name == tool_name), None)
            if not tool:
                return f"Error: Tool '{tool_name}' not found"

            runtime_block = self._check_runtime_policy(
                tool,
                args=args,
                tenant_id=tenant_id,
                metadata=metadata,
                request_context=request_context,
            )
            if runtime_block is not None:
                return runtime_block

            context = AgentToolContext(
                tenant_id=tenant_id,
                credentials=self.credential_store,
                metadata=self._build_tool_metadata(metadata),
            )
            return await tool.executor(args, context)

    def _is_agent_tool(self, tool_name: str) -> bool:
        """Check if tool_name corresponds to a registered agent."""
        if not self._agent_registry:
            return False
        return self._agent_registry.get_agent_class(tool_name) is not None

    def _cap_tool_result(self, result_text: str) -> str:
        """Hard cap on tool result size to prevent context window overflow."""
        if len(result_text) <= TOOL_RESULT_HARD_CAP_CHARS:
            return result_text
        cut = TOOL_RESULT_HARD_CAP_CHARS
        newline_pos = result_text.rfind("\n", int(cut * 0.8), cut)
        if newline_pos > 0:
            cut = newline_pos
        logger.warning(
            f"[ReAct] Tool result truncated: {len(result_text)} -> {cut} chars"
        )
        return result_text[:cut] + "\n\n[truncated - result exceeded size limit]"

    def _check_schema_policy(self, tool_name: str, *, tenant_id: str) -> Optional[str]:
        """Reapply schema-time policy at execution time."""
        if not self._tool_policy_filter:
            return None
        if self._tool_policy_filter.is_tool_allowed(tool_name, tenant_id=tenant_id):
            return None
        reason = self._tool_policy_filter.get_filter_reason(tool_name, tenant_id=tenant_id)
        detail = f": {reason}" if reason else ""
        return f"Error: Tool '{tool_name}' is blocked by policy{detail}."

    def _check_runtime_policy(
        self,
        tool: AgentTool,
        *,
        args: Dict[str, Any],
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Apply runtime execution policy to a concrete tool call."""
        if not getattr(self, "_execution_policy", None):
            return None

        decision = self._execution_policy.evaluate(
            tool,
            tenant_id=tenant_id,
            args=args,
            metadata=metadata,
            request_context=request_context,
        )
        if not decision.allowed:
            return f"Error: Permission denied for tool '{tool.name}': {decision.reason}"
        if decision.require_approval:
            return (
                f"Error: Tool '{tool.name}' requires approval before execution. "
                "Ask the user for confirmation or pre-approve the tool in request permissions."
            )
        return None

    def _build_notify_user_tool(self, context: Dict[str, Any]):
        """Build a notify_user tool for conditional cron delivery.

        When a cron job has conditional delivery, this tool is injected into
        the ReAct loop. The agent calls it ONLY when the monitored condition
        is met. If the agent never calls it, no notification is sent.

        Returns:
            (AgentTool, schema_dict) tuple
        """
        def make_executor(ctx):
            async def notify_user_executor(args: dict, tool_context=None) -> str:
                message = args.get("message", "")
                if not message:
                    return "Error: message is required."
                ctx["cron_notification"] = message
                return f"Notification queued: {message}"
            return notify_user_executor

        tool = AgentTool(
            name="notify_user",
            description=(
                "Send a notification to the user. ONLY call this when a monitored "
                "condition is actually met (e.g. price dropped below threshold, "
                "weather changed, etc.). If the condition is NOT met, do NOT call "
                "this tool — just respond normally without notifying."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The notification message to send to the user",
                    }
                },
                "required": ["message"],
            },
            executor=make_executor(context),
            category="cron",
            read_only=False,
            mutates_user_data=True,
            idempotent=False,
        )

        return tool, tool.to_openai_schema()

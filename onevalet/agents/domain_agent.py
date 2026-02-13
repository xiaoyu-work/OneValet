"""
DomainAgent - Base class for domain-specific agents with internal ReAct loops.

Each DomainAgent groups related API tools under a single agent that the orchestrator
sees as one tool. Internally, the DomainAgent runs its own mini ReAct loop with
a small set of domain-specific tools (3-5 typically).

This solves the "too many tools" problem: instead of exposing 50+ fine-grained
agent-tools to the orchestrator, we expose ~8 domain agents, each handling
its own tool selection internally.

Usage:
    from onevalet import valet
    from onevalet.agents.domain_agent import DomainAgent, DomainTool

    @valet(capabilities=["travel"])
    class TravelAgent(DomainAgent):
        '''Plan travel, search flights, hotels, and weather.'''

        domain_system_prompt = "You are a travel planning assistant..."
        domain_tools = [
            DomainTool(name="search_flights", ...),
            DomainTool(name="search_hotels", ...),
        ]

        async def on_running(self, msg):
            return await self.run_domain_react(msg)
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..llm.base import LLMResponse, ToolCall as LLMToolCall
from ..message import Message
from ..result import AgentResult, AgentStatus, ApprovalResult
from ..standard_agent import StandardAgent

logger = logging.getLogger(__name__)


@dataclass
class DomainToolContext:
    """Context passed to domain tool executors.

    Provides access to shared resources that tool functions need.
    """

    llm_client: Any = None
    tenant_id: str = ""
    user_profile: Optional[Dict[str, Any]] = None
    context_hints: Optional[Dict[str, Any]] = None


@dataclass
class DomainTool:
    """A tool available inside a DomainAgent's mini ReAct loop.

    Attributes:
        name: Tool function name (used in LLM tool_calls).
        description: What this tool does (shown to domain LLM).
        parameters: JSON Schema for tool arguments.
        executor: Async function(args: dict, context: DomainToolContext) -> str.
        needs_approval: If True, pause execution for user confirmation before running.
        get_preview: Async function to generate human-readable preview for approval.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    executor: Callable
    needs_approval: bool = False
    get_preview: Optional[Callable] = None

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function-calling tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class DomainAgent(StandardAgent):
    """Base class for domain agents with internal mini ReAct loops."""

    domain_system_prompt: str = ""
    domain_tools: List[DomainTool] = []
    max_domain_turns: int = 5
    domain_router_prompt: str = (
        "You are a domain tool routing policy engine. Return strict JSON only: "
        "{\"must_use_tools\": boolean, \"force_first_tool\": string|null, \"reason_code\": string}. "
        "Set must_use_tools=true when user intent needs tool data/action. "
        "Set must_use_tools=false ONLY for clear non-tool scenarios: casual_chat, creative_writing, "
        "language_translation, text_rewrite, pure_math."
    )

    def get_system_prompt(self) -> str:
        """Return the system prompt for the mini ReAct loop."""
        return self.domain_system_prompt

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._react_messages: List[Dict[str, Any]] = []
        self._react_turn: int = 0
        self._pending_tool_call: Optional[Tuple[LLMToolCall, DomainTool, Dict[str, Any]]] = None
        self._remaining_tool_calls: List[LLMToolCall] = []
        self._tool_trace: List[Dict[str, Any]] = []

    def needs_approval(self) -> bool:
        return False

    async def on_running(self, msg: Message) -> AgentResult:
        """Run or resume the domain mini ReAct loop."""
        if self._pending_tool_call:
            return await self._resume_after_approval()

        instruction = self.collected_fields.get("task_instruction", "")
        if not instruction and msg:
            instruction = msg.get_text()

        if not instruction:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No task instruction provided.",
                metadata={
                    "tool_trace": list(self._tool_trace),
                    "tool_calls_count": len(self._tool_trace),
                },
            )

        self._react_messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": instruction},
        ]
        self._react_turn = 0
        self._tool_trace = []
        return await self._run_react()

    async def _run_react(self) -> AgentResult:
        """Core mini ReAct loop with domain tools."""
        tool_schemas = [t.to_openai_schema() for t in self.domain_tools]
        messages = self._react_messages
        instruction = ""
        for m in messages:
            if m.get("role") == "user":
                instruction = str(m.get("content", "") or "")
                break
        policy = await self._plan_first_turn_tool_policy(instruction, tool_schemas)

        if self._remaining_tool_calls:
            result = await self._execute_tool_calls(self._remaining_tool_calls, messages)
            self._remaining_tool_calls = []
            if result is not None:
                return result

        for turn in range(self._react_turn, self.max_domain_turns):
            self._react_turn = turn + 1
            tool_choice = policy["first_turn_tool_choice"] if turn == 0 else "auto"
            response: LLMResponse = await self.llm_client.chat_completion(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice=tool_choice,
            )

            if not response.has_tool_calls:
                if (
                    turn == 0
                    and policy["retry_with_required_on_empty"]
                    and tool_schemas
                    and tool_choice != "required"
                ):
                    response = await self.llm_client.chat_completion(
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice="required",
                    )
                if not response.has_tool_calls:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=response.content or "",
                        metadata={
                            "tool_trace": list(self._tool_trace),
                            "tool_calls_count": len(self._tool_trace),
                        },
                    )

            messages.append(self._format_assistant_msg(response))
            result = await self._execute_tool_calls(response.tool_calls, messages)
            if result is not None:
                return result

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=(
                "I wasn't able to complete the task within the allowed steps. "
                "Please try again with more specific information."
            ),
            metadata={
                "tool_trace": list(self._tool_trace),
                "tool_calls_count": len(self._tool_trace),
            },
        )

    async def _execute_tool_calls(
        self,
        tool_calls: List[LLMToolCall],
        messages: List[Dict[str, Any]],
    ) -> Optional[AgentResult]:
        """Execute tool calls. Returns AgentResult if paused for approval, None otherwise."""
        for i, tc in enumerate(tool_calls):
            tool = self._find_domain_tool(tc.name)
            if tool is None:
                error_text = f"Error: Unknown tool '{tc.name}'"
                self._tool_trace.append(
                    {"tool": tc.name, "status": "error", "summary": error_text}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": error_text,
                    }
                )
                continue

            args = tc.arguments if isinstance(tc.arguments, dict) else {}

            if tool.needs_approval:
                if tool.get_preview:
                    try:
                        preview = await tool.get_preview(args, self._build_tool_context())
                    except Exception as e:
                        logger.error(f"Preview generation failed for {tc.name}: {e}")
                        preview = f"About to execute: {tc.name}({json.dumps(args, ensure_ascii=False)})"
                else:
                    preview = f"About to execute: {tc.name}({json.dumps(args, ensure_ascii=False)})"

                self._pending_tool_call = (tc, tool, args)
                self._remaining_tool_calls = list(tool_calls[i + 1 :])
                self._react_messages = messages
                self._tool_trace.append(
                    {
                        "tool": tc.name,
                        "status": "waiting_for_approval",
                        "summary": preview[:240],
                    }
                )
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_APPROVAL,
                    raw_message=preview,
                    metadata={
                        "tool_trace": list(self._tool_trace),
                        "tool_calls_count": len(self._tool_trace),
                    },
                )

            try:
                result_text = await tool.executor(args, self._build_tool_context())
                self._tool_trace.append(
                    {
                        "tool": tc.name,
                        "status": "ok",
                        "summary": str(result_text)[:240],
                    }
                )
            except Exception as e:
                logger.error(f"Domain tool {tc.name} failed: {e}", exc_info=True)
                result_text = f"Error executing {tc.name}: {e}"
                self._tool_trace.append(
                    {
                        "tool": tc.name,
                        "status": "error",
                        "summary": str(result_text)[:240],
                    }
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result_text),
                }
            )

        return None

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        """Handle user's approval response (approve/reject/modify)."""
        user_input = msg.get_text() if msg else ""
        approval = self.parse_approval(user_input)

        if approval == ApprovalResult.APPROVED:
            return await self._resume_after_approval()
        if approval == ApprovalResult.REJECTED:
            self._pending_tool_call = None
            self._remaining_tool_calls = []
            self._tool_trace.append(
                {"tool": "approval", "status": "rejected", "summary": "User rejected approval."}
            )
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message="Operation cancelled.",
                metadata={
                    "tool_trace": list(self._tool_trace),
                    "tool_calls_count": len(self._tool_trace),
                },
            )

        self._pending_tool_call = None
        self._remaining_tool_calls = []
        self._tool_trace.append(
            {
                "tool": "approval",
                "status": "modified",
                "summary": f"User requested modification: {user_input[:180]}",
            }
        )
        return self.make_result(
            status=AgentStatus.CANCELLED,
            raw_message=f"Operation cancelled. User said: {user_input}",
            metadata={
                "tool_trace": list(self._tool_trace),
                "tool_calls_count": len(self._tool_trace),
            },
        )

    async def _resume_after_approval(self) -> AgentResult:
        """Execute approved tool and continue mini ReAct loop."""
        if not self._pending_tool_call:
            return self.make_result(
                status=AgentStatus.ERROR,
                raw_message="No pending tool call to resume.",
                metadata={
                    "tool_trace": list(self._tool_trace),
                    "tool_calls_count": len(self._tool_trace),
                },
            )

        tc, tool, args = self._pending_tool_call
        self._pending_tool_call = None

        try:
            result_text = await tool.executor(args, self._build_tool_context())
            self._tool_trace.append(
                {
                    "tool": tc.name,
                    "status": "ok",
                    "summary": str(result_text)[:240],
                }
            )
        except Exception as e:
            logger.error(f"Approved tool {tc.name} failed: {e}", exc_info=True)
            result_text = f"Error executing {tc.name}: {e}"
            self._tool_trace.append(
                {
                    "tool": tc.name,
                    "status": "error",
                    "summary": str(result_text)[:240],
                }
            )

        self._react_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result_text),
            }
        )
        return await self._run_react()

    def _find_domain_tool(self, name: str) -> Optional[DomainTool]:
        for tool in self.domain_tools:
            if tool.name == name:
                return tool
        return None

    def _build_tool_context(self) -> DomainToolContext:
        return DomainToolContext(
            llm_client=self.llm_client,
            tenant_id=self.tenant_id,
            user_profile=self.context_hints.get("user_profile") if self.context_hints else None,
            context_hints=self.context_hints,
        )

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    async def _plan_first_turn_tool_policy(
        self,
        instruction: str,
        tool_schemas: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Plan first-turn tool policy with a lightweight router call."""
        if not tool_schemas:
            return {
                "first_turn_tool_choice": "auto",
                "retry_with_required_on_empty": False,
            }

        tool_names: List[str] = []
        tool_lines: List[str] = []
        for schema in tool_schemas:
            function_data = schema.get("function", {}) if isinstance(schema, dict) else {}
            name = function_data.get("name")
            if not isinstance(name, str):
                continue
            desc = function_data.get("description", "")
            tool_names.append(name)
            tool_lines.append(f"- {name}: {desc}")

        if not tool_names:
            return {
                "first_turn_tool_choice": "auto",
                "retry_with_required_on_empty": False,
            }

        route_messages = [
            {"role": "system", "content": self.domain_router_prompt},
            {
                "role": "user",
                "content": (
                    f"User instruction:\n{instruction}\n\n"
                    f"Available tools:\n" + "\n".join(tool_lines) + "\n\n"
                    "Return JSON only."
                ),
            },
        ]
        try:
            route_response: LLMResponse = await self.llm_client.chat_completion(messages=route_messages)
            route_json = self._extract_json_object(route_response.content or "")
        except Exception:
            route_json = None

        if not route_json:
            return {
                "first_turn_tool_choice": "auto",
                "retry_with_required_on_empty": False,
            }

        must_use_tools = bool(route_json.get("must_use_tools"))
        reason_code = str(route_json.get("reason_code", "") or "").strip().lower()
        no_tool_reasons = {
            "casual_chat",
            "creative_writing",
            "language_translation",
            "text_rewrite",
            "pure_math",
        }
        if not must_use_tools and reason_code not in no_tool_reasons:
            must_use_tools = True

        force_first_tool = route_json.get("force_first_tool")
        if isinstance(force_first_tool, str) and force_first_tool in tool_names:
            return {
                "first_turn_tool_choice": {
                    "type": "function",
                    "function": {"name": force_first_tool},
                },
                "retry_with_required_on_empty": True,
            }
        if must_use_tools:
            return {
                "first_turn_tool_choice": "required",
                "retry_with_required_on_empty": True,
            }
        return {
            "first_turn_tool_choice": "auto",
            "retry_with_required_on_empty": False,
        }

    @staticmethod
    def _format_assistant_msg(response: LLMResponse) -> Dict[str, Any]:
        """Convert LLMResponse to OpenAI-format assistant message."""
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": response.content or None,
        }
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        if isinstance(tc.arguments, dict)
                        else tc.arguments,
                    },
                }
                for tc in response.tool_calls
            ]
        return msg

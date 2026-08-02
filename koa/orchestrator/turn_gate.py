"""What happens to the model's tool calls before any of them run.

Two gates stand between "the model asked for these calls" and "we ran
them". complete_task never runs -- it is the model saying it is done, and
is answered here. Everything else is checked against the schema it was
offered, because a model can invent a tool name or an argument shape, and
a rejected call has to come back as a tool result the model can read
rather than as an exception that ends the run.

Both gates share one rule: every call the model made gets an answer in
the transcript, whether it ran, was rejected, or was never going to run.
A provider rejects the next request otherwise.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm.tool_validator import ToolSchemaValidator
from .react_config import COMPLETE_TASK_TOOL_NAME, CompleteTaskResult
from .run_state import RunState

logger = logging.getLogger(__name__)

_MISSING_RESULT_ERROR = 'Error: "result" argument is required for complete_task.'


@dataclass
class CompleteTaskIntercept:
    """The outcome of pulling complete_task out of a turn's calls."""

    result: Optional[CompleteTaskResult] = None
    call_id: Optional[str] = None

    #: Calls that still need executing.
    remaining: List[Any] = field(default_factory=list)

    @property
    def ends_turn_alone(self) -> bool:
        """The model only said it was done, so nothing needs to run."""
        return self.result is not None and not self.remaining


class TurnGateMixin:
    """Screens a turn's tool calls before execution."""

    def _intercept_complete_task(
        self,
        tool_calls: List[Any],
        messages: List[Dict[str, Any]],
    ) -> CompleteTaskIntercept:
        """Separate the model's "I'm done" from calls that need running.

        A complete_task without a result is not a completion, so it is
        answered with an error and left in the queue for the schema gate to
        reject -- the model gets told what was wrong either way.
        """
        intercept = CompleteTaskIntercept()
        for tc in tool_calls:
            if tc.name != COMPLETE_TASK_TOOL_NAME:
                intercept.remaining.append(tc)
                continue

            try:
                args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            text = args.get("result", "") if isinstance(args, dict) else ""

            if not text:
                messages.append(
                    self._build_tool_result_message(tc.id, _MISSING_RESULT_ERROR, is_error=True)
                )
                intercept.remaining.append(tc)
                continue

            intercept.result = CompleteTaskResult(result=text)
            intercept.call_id = tc.id
            logger.info(f"[ReAct] complete_task called ({len(text)} chars)")
        return intercept

    def _settle_complete_task(
        self,
        intercept: CompleteTaskIntercept,
        messages: List[Dict[str, Any]],
        state: RunState,
        also_called: Optional[List[str]] = None,
    ) -> None:
        """Answer the completion call and make its text the run's answer."""
        text = intercept.result.result if intercept.result else ""
        messages.append(self._build_tool_result_message(intercept.call_id, "Task completed."))
        state.record_tool_call(
            name=COMPLETE_TASK_TOOL_NAME,
            args_summary={"result": text[:100]},
            duration_ms=0,
            success=True,
            result_status="COMPLETED",
            result_chars=len(text),
        )
        state.final_response = text
        self._audit.log_react_turn(
            turn=state.turn,
            tool_calls=(also_called or []) + [COMPLETE_TASK_TOOL_NAME],
            final_answer=True,
            tenant_id=state.tenant_id,
        )

    def _validate_tool_calls(
        self,
        tool_calls: List[Any],
        tool_schemas: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        state: RunState,
    ) -> List[Any]:
        """Drop calls that do not match the schemas the model was given.

        A rejected call becomes an error tool result naming the problem and
        the tools that do exist, so the model can correct itself on the next
        turn instead of the run failing on a name it invented.
        """
        validator = ToolSchemaValidator.from_openai_tools(tool_schemas)
        accepted = []
        for tc in tool_calls:
            # complete_task is synthetic; it is never in the schema list.
            if tc.name == COMPLETE_TASK_TOOL_NAME:
                accepted.append(tc)
                continue

            reason, details = _check(validator, tc)
            if reason is None:
                accepted.append(tc)
                continue

            logger.warning("[ReAct] Rejecting tool call %r: %s %s", tc.name, reason, details)
            self._audit.log_tool_execution(
                tool_name=tc.name,
                args_summary={"rejected_reason": reason},
                success=False,
                duration_ms=0,
                error=f"schema_validation:{reason}",
                tenant_id=state.tenant_id,
            )
            messages.append(
                self._build_tool_result_message(
                    tc.id,
                    f"Error: tool call rejected by schema validation "
                    f"({reason}). details={json.dumps(details, default=str)[:256]}. "
                    f"Allowed tools: {', '.join(validator.known_names[:20])}",
                    is_error=True,
                )
            )
            state.record_tool_call(
                name=tc.name,
                args_summary={"rejected_reason": reason},
                duration_ms=0,
                success=False,
                result_status="REJECTED",
                result_chars=0,
            )
        return accepted


def _check(validator: ToolSchemaValidator, tc: Any) -> tuple:
    """Return (reason, details) for a bad call, or (None, {}) if it passes."""
    try:
        args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return "arguments_not_json", {"name": tc.name}

    outcome = validator.validate(tc.name, args)
    if outcome.ok:
        return None, {}
    return outcome.reason, outcome.details or {}

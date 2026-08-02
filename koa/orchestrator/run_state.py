"""Mutable state carried through a single ReAct run.

The loop accumulates a dozen values -- token totals, tool records, watchdog
fingerprints, the media to hand back -- and every one of them is read or
written at several points hundreds of lines apart. Holding them as locals
inside one function is what forced that function to stay one function: any
extraction had to take them all as parameters and give them all back.

Collecting them here lets a phase be extracted as a method that takes the
run and mutates it, which is what the loop was already doing to its locals.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .react_config import TokenUsage, ToolCallRecord

#: How many recent tool calls the watchdog keeps to detect repetition.
_WATCHDOG_WINDOW = 12


@dataclass
class RunState:
    """Everything a ReAct run accumulates from start to EXECUTION_END."""

    tenant_id: str
    start_time: float = field(default_factory=time.monotonic)

    #: Turn number currently executing; 0 until the loop body runs.
    turn: int = 0

    final_response: str = ""
    result_status: Optional[str] = None
    interrupted: bool = False

    tool_records: List[ToolCallRecord] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    pending_approvals: List[Any] = field(default_factory=list)

    #: Media (images, inline cards) to return alongside the final response.
    response_media: List[Dict[str, Any]] = field(default_factory=list)

    #: Decided once before the loop and reused every turn.
    llm_client: Optional[Any] = None
    routing_score: int = -1
    enable_reasoning: bool = False

    #: Watchdog history, parallel lists appended once per executed tool call.
    recent_names: List[str] = field(default_factory=list)
    recent_fingerprints: List[str] = field(default_factory=list)
    recent_result_hashes: List[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return int((time.monotonic() - self.start_time) * 1000)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def add_usage(self, usage: Any) -> int:
        """Fold one response's usage into the run total.

        Returns the prompt-side token count, which is the number the context
        manager needs: it is what actually occupied the window this round.
        """
        if not usage:
            return 0
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        self.usage.input_tokens += prompt_tokens
        self.usage.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.usage.cost_usd += getattr(usage, "cost", 0) or 0
        return prompt_tokens

    def turn_tokens(self, usage: Any) -> Optional[TokenUsage]:
        """Tokens attributable to this turn, for per-tool-call attribution."""
        if not usage:
            return None
        return TokenUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def record_tool_call(self, **kwargs: Any) -> None:
        self.tool_records.append(ToolCallRecord(**kwargs))

    def observe_for_watchdog(self, name: str, arguments: Any, result: Any) -> None:
        """Fingerprint one executed call so repetition can be detected.

        A repeat is only interesting when the same tool is called with the
        same arguments and gets the same answer back, so all three are
        recorded rather than the name alone.
        """
        self.recent_names.append(name)

        try:
            parsed = arguments if isinstance(arguments, dict) else json.loads(arguments)
            args_str = json.dumps(parsed, sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            args_str = str(arguments)
        digest = hashlib.md5(args_str.encode(), usedforsecurity=False).hexdigest()[:8]
        self.recent_fingerprints.append(f"{name}:{digest}")

        result_str = str(result) if result is not None else ""
        self.recent_result_hashes.append(
            hashlib.md5(result_str[:2000].encode(), usedforsecurity=False).hexdigest()[:8]
        )

        # Only the tail matters; without this the lists grow with the run.
        for history in (
            self.recent_names,
            self.recent_fingerprints,
            self.recent_result_hashes,
        ):
            del history[:-_WATCHDOG_WINDOW]

    def execution_end_payload(self) -> Dict[str, Any]:
        """The data block of the terminal EXECUTION_END event."""
        import dataclasses

        return {
            "duration_ms": self.duration_ms,
            "turns": self.turn,
            "tool_calls_count": len(self.tool_records),
            "final_response": self.final_response,
            "result_status": self.result_status,
            "interrupted": self.interrupted,
            "pending_approvals": self.pending_approvals,
            "media": self.response_media or None,
            "token_usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cost_usd": round(self.usage.cost_usd, 6),
            },
            "tool_calls": [dataclasses.asdict(r) for r in self.tool_records],
        }

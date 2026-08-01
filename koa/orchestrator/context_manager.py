"""Lightweight context management with a three-line-of-defense system.

Defense 1 -- Single tool-result truncation (after each tool execution).
Defense 2 -- History message trimming (before each loop iteration).
Defense 2b -- Context summarization (summarize old messages via LLM before dropping).
Defense 3 -- Force trim to safe range (after a context overflow error).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    JSON_CHARS_PER_TOKEN,
    JSON_DETECTION_RATIO,
    JSON_DETECTION_SAMPLE_SIZE,
    TEXT_CHARS_PER_TOKEN,
    TOOL_CALL_STRUCTURE_OVERHEAD_TOKENS,
)
from .react_config import ReactLoopConfig

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages conversation context size using three lines of defense.

    The trigger signal is the prompt-side token total the provider actually
    reported for the most recent call (``observe_usage``). Character-based
    estimation is only a fallback for the first call of a request, before any
    usage has been reported.
    """

    def __init__(self, config: ReactLoopConfig) -> None:
        self.config = config
        self._last_prompt_tokens: Optional[int] = None
        self._context_window: Optional[int] = None

    # ------------------------------------------------------------------
    # Live signals
    # ------------------------------------------------------------------

    def set_context_window(self, context_window: Optional[int]) -> None:
        """Bind the real window of the model in use, overriding the config default."""
        if context_window and context_window > 0:
            self._context_window = int(context_window)

    @property
    def context_window(self) -> int:
        return self._context_window or self.config.context_token_limit

    def observe_usage(self, prompt_tokens: Optional[int]) -> None:
        """Record the prompt-side total the provider reported for the last call.

        This is what actually occupied the window on that round-trip, so it is a
        far better trigger than counting characters.
        """
        if prompt_tokens and prompt_tokens > 0:
            self._last_prompt_tokens = int(prompt_tokens)

    def invalidate_usage(self) -> None:
        """Drop the recorded measurement after the message list is rewritten.

        A trim or summarization changes what will be sent, so the previous
        prompt-token count no longer describes it; fall back to estimation until
        the provider reports again.
        """
        self._last_prompt_tokens = None

    def current_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Best available measure of what the next call will cost."""
        if self._last_prompt_tokens is not None:
            return self._last_prompt_tokens
        return self.estimate_tokens(messages)

    def _trim_threshold(self) -> int:
        return int(self.context_window * self.config.context_trim_threshold)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count from messages.

        Uses ~4 chars/token for English text, ~2 chars/token for code/JSON,
        plus overhead per message (role, formatting).
        """
        total = 0
        for msg in messages:
            content = msg.get("content")
            if content is None:
                continue
            if isinstance(content, str):
                total += self._estimate_string_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "")
                        if part_type in ("image_url", "image"):
                            continue
                        text = part.get("text") or part.get("content", "")
                        if isinstance(text, str):
                            total += self._estimate_string_tokens(text)
                    elif isinstance(part, str):
                        total += self._estimate_string_tokens(part)
            # Tool calls in assistant messages
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    total += TOOL_CALL_STRUCTURE_OVERHEAD_TOKENS
                    args = tc.get("arguments") or tc.get("function", {}).get("arguments", "")
                    if isinstance(args, str):
                        total += len(args) // JSON_CHARS_PER_TOKEN
                    elif isinstance(args, dict):
                        import json

                        total += len(json.dumps(args)) // JSON_CHARS_PER_TOKEN
        return total

    @staticmethod
    def _estimate_string_tokens(text: str) -> int:
        """Estimate tokens for a string. JSON/code averages ~3 chars/token,
        natural language ~4 chars/token."""
        if not text:
            return 0
        # Heuristic: if special-char fraction exceeds threshold, treat as JSON/code
        sample_len = min(len(text), JSON_DETECTION_SAMPLE_SIZE)
        special = sum(1 for c in text[:sample_len] if c in '{}[]":,')
        ratio = special / sample_len if text else 0
        chars_per_token = (
            JSON_CHARS_PER_TOKEN if ratio > JSON_DETECTION_RATIO else TEXT_CHARS_PER_TOKEN
        )
        return len(text) // chars_per_token

    # ------------------------------------------------------------------
    # Defense 1: Single tool-result truncation
    # ------------------------------------------------------------------

    def truncate_tool_result(self, result: str) -> str:
        """Truncate a single tool result to stay within budget.

        The budget is the smaller of:
          - context_token_limit * max_tool_result_share * 4  (chars)
          - max_tool_result_chars
        Truncation prefers a newline boundary when possible.
        """
        max_chars = int(
            min(
                self.config.context_token_limit * self.config.max_tool_result_share * 4,
                self.config.max_tool_result_chars,
            )
        )
        if len(result) <= max_chars:
            return result

        # Try to cut at the last newline within the budget
        cut = result[:max_chars]
        newline_pos = cut.rfind("\n")
        if newline_pos > max_chars // 2:
            cut = cut[: newline_pos + 1]

        return cut + "\n[...truncated]"

    # ------------------------------------------------------------------
    # Defense 2: History message trimming
    # ------------------------------------------------------------------

    def trim_if_needed(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trim history when the context in use exceeds the trim threshold.

        Keeps the system prompt (first message if role=='system') plus the
        most recent ``max_history_messages`` messages.
        """
        if self.current_tokens(messages) <= self._trim_threshold():
            return messages

        self.invalidate_usage()
        return self._keep_recent(messages, self.config.max_history_messages)

    def trim_now(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trim unconditionally, without consulting the threshold.

        Used on a confirmed context-overflow error: the provider has already
        told us the payload was too big, so the trigger check would only be a
        chance to disagree with it and retry the same failing request.
        """
        self.invalidate_usage()
        return self._keep_recent(messages, self.config.max_history_messages)

    # ------------------------------------------------------------------
    # Step 2 of overflow recovery: truncate all tool results in-place
    # ------------------------------------------------------------------

    def truncate_all_tool_results(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Walk all tool-result messages and apply truncation to each."""
        out: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "tool":
                msg = dict(msg)  # shallow copy to avoid mutating caller
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = self.truncate_tool_result(content)
                elif isinstance(content, list):
                    new_parts = []
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            part = {**part, "text": self.truncate_tool_result(part["text"])}
                        new_parts.append(part)
                    msg["content"] = new_parts
            out.append(msg)
        return out

    # ------------------------------------------------------------------
    # Defense 3: Force trim to safe range
    # ------------------------------------------------------------------

    def force_trim(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggressively trim to system prompt + most recent 5 messages."""
        self.invalidate_usage()
        return self._keep_recent(messages, keep=5)

    # ------------------------------------------------------------------
    # Defense 2b: Context summarization
    # ------------------------------------------------------------------

    def split_for_summarization(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """Check if context needs trimming and split messages for summarization.

        Returns None if no trimming is needed.
        Otherwise returns (system_msgs, old_msgs_to_summarize, recent_msgs_to_keep).
        """
        if self.current_tokens(messages) <= self._trim_threshold():
            return None

        if not messages:
            return None

        if messages[0].get("role") == "system":
            system = [messages[0]]
            rest = messages[1:]
        else:
            system = []
            rest = messages

        keep = self.config.max_history_messages
        if len(rest) <= keep:
            return None

        old = rest[:-keep]
        recent = rest[-keep:]
        return system, old, recent

    @staticmethod
    def build_summarized_messages(
        system_msgs: List[Dict[str, Any]],
        summary_text: str,
        recent_msgs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rebuild the message list with a summary replacing old messages."""
        result = list(system_msgs)
        result.append(
            {
                "role": "user",
                "content": f"[Conversation summary of earlier messages]\n{summary_text}",
            }
        )
        result.extend(recent_msgs)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _keep_recent(messages: List[Dict[str, Any]], keep: int) -> List[Dict[str, Any]]:
        """Return the system prompt (if present) plus the last *keep* messages."""
        if not messages:
            return messages

        if messages[0].get("role") == "system":
            system = [messages[0]]
            rest = messages[1:]
        else:
            system = []
            rest = messages

        trimmed = rest[-keep:] if len(rest) > keep else rest
        return system + trimmed

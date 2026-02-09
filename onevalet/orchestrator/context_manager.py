"""Lightweight context management with a three-line-of-defense system.

Defense 1 -- Single tool-result truncation (after each tool execution).
Defense 2 -- History message trimming (before each loop iteration).
Defense 3 -- Force trim to safe range (after a context overflow error).
"""

from typing import Any, Dict, List

from .react_config import ReactLoopConfig


class ContextManager:
    """Manages conversation context size using three lines of defense."""

    def __init__(self, config: ReactLoopConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count from messages using ~4 chars per token."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content")
            if content is None:
                continue
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Structured content blocks (e.g. tool results with parts)
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content", "")
                        if isinstance(text, str):
                            total_chars += len(text)
                    elif isinstance(part, str):
                        total_chars += len(part)
        return total_chars // 4

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
        """Trim history when estimated tokens exceed the trim threshold.

        Keeps the system prompt (first message if role=='system') plus the
        most recent ``max_history_messages`` messages.
        """
        threshold = int(self.config.context_token_limit * self.config.context_trim_threshold)
        if self.estimate_tokens(messages) <= threshold:
            return messages

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
        return self._keep_recent(messages, keep=5)

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

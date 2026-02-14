"""
Transcript Repair - Fix malformed message transcripts before LLM calls

Ported from OpenClaw's transcript repair logic. Handles two categories of issues:

1. Tool call inputs: Drops tool_calls with missing arguments from assistant messages.
2. Tool use result pairing: Ensures every tool_call has a matching tool result
   immediately after the assistant message, handling displaced/missing/duplicate/orphan
   results.

All functions operate on OpenAI message format:
- Assistant: {"role": "assistant", "content": "...", "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]}
- Tool result: {"role": "tool", "tool_call_id": "...", "content": "..."}
"""

import logging
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

SYNTHETIC_TOOL_RESULT = "[synthetic] missing tool result - inserted for transcript repair"


def repair_tool_call_inputs(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Drop tool_calls that have no arguments from assistant messages.

    If all tool_calls in an assistant message are dropped, the entire message
    is removed.

    Args:
        messages: List of OpenAI-format messages.

    Returns:
        Tuple of (repaired_messages, dropped_tool_calls_count, dropped_assistant_messages_count).
        If no changes are needed, returns the original list reference.
    """
    dropped_tool_calls = 0
    dropped_assistant_messages = 0
    repaired: List[Dict[str, Any]] = []
    changed = False

    for msg in messages:
        if msg.get("role") != "assistant" or "tool_calls" not in msg:
            repaired.append(msg)
            continue

        tool_calls = msg["tool_calls"]
        valid_tool_calls = []

        for tc in tool_calls:
            func = tc.get("function", {})
            arguments = func.get("arguments")
            # Drop if arguments is missing, None, or empty string
            if not arguments:
                dropped_tool_calls += 1
                tc_id = tc.get("id", "unknown")
                tc_name = func.get("name", "unknown")
                logger.warning(
                    "transcript_repair: dropped tool_call %s (%s) - missing arguments",
                    tc_id, tc_name,
                )
                changed = True
            else:
                valid_tool_calls.append(tc)

        if not valid_tool_calls:
            # All tool_calls were invalid - drop the entire assistant message
            dropped_assistant_messages += 1
            logger.warning(
                "transcript_repair: dropped assistant message - all tool_calls invalid"
            )
            changed = True
        elif len(valid_tool_calls) < len(tool_calls):
            # Some tool_calls were dropped - rebuild message
            new_msg = dict(msg)
            new_msg["tool_calls"] = valid_tool_calls
            repaired.append(new_msg)
        else:
            repaired.append(msg)

    if not changed:
        return messages, 0, 0

    return repaired, dropped_tool_calls, dropped_assistant_messages


def repair_tool_use_result_pairing(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int, int, int]:
    """
    Ensure every tool_call in an assistant message has a matching tool result
    immediately after it.

    Handles:
    - Displaced tool results: moved to the correct position after the assistant message
    - Missing tool results: synthetic error results inserted
    - Duplicate tool results: only the first is kept
    - Orphaned tool results: results with no matching tool_call are dropped

    Assistant messages with stop_reason "error" or "aborted" are skipped.

    Args:
        messages: List of OpenAI-format messages.

    Returns:
        Tuple of (repaired_messages, added_synthetic, dropped_duplicates,
        dropped_orphans, moved).
        If no changes are needed, returns the original list reference.
    """
    added_synthetic = 0
    dropped_duplicates = 0
    dropped_orphans = 0
    moved = 0

    # Build a lookup of all tool results by tool_call_id
    # Map: tool_call_id -> list of (index, message)
    tool_result_map: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id:
                tool_result_map.setdefault(tc_id, []).append((i, msg))

    # Collect all tool_call_ids that are expected by assistant messages
    expected_tool_call_ids: Set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            stop_reason = msg.get("stop_reason")
            if stop_reason in ("error", "aborted"):
                continue
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                if tc_id:
                    expected_tool_call_ids.add(tc_id)

    # Build repaired list
    repaired: List[Dict[str, Any]] = []
    consumed_indices: Set[int] = set()
    changed = False

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.get("role") == "assistant" and "tool_calls" in msg:
            stop_reason = msg.get("stop_reason")
            if stop_reason in ("error", "aborted"):
                # Skip processing - keep as-is
                repaired.append(msg)
                i += 1
                continue

            repaired.append(msg)
            i += 1

            # Collect the tool_call_ids from this assistant message
            tc_ids = [tc.get("id", "") for tc in msg["tool_calls"] if tc.get("id")]

            seen_ids: Set[str] = set()
            for tc_id in tc_ids:
                if tc_id in seen_ids:
                    continue
                seen_ids.add(tc_id)

                results = tool_result_map.get(tc_id, [])
                if not results:
                    # No result found - insert synthetic
                    synthetic = {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": SYNTHETIC_TOOL_RESULT,
                    }
                    repaired.append(synthetic)
                    added_synthetic += 1
                    changed = True
                    logger.warning(
                        "transcript_repair: inserted synthetic result for tool_call %s",
                        tc_id,
                    )
                else:
                    # Use the first result
                    first_idx, first_result = results[0]
                    consumed_indices.add(first_idx)
                    repaired.append(first_result)

                    # Check if the result was displaced (not immediately after assistant)
                    # We consider it "moved" if the original index isn't the next
                    # position after the assistant message
                    if first_idx != i + len(seen_ids) - 1:
                        moved += 1
                        changed = True
                        logger.info(
                            "transcript_repair: moved tool result for %s from index %d",
                            tc_id, first_idx,
                        )

                    # Drop duplicates
                    for dup_idx, _ in results[1:]:
                        consumed_indices.add(dup_idx)
                        dropped_duplicates += 1
                        changed = True
                        logger.warning(
                            "transcript_repair: dropped duplicate result for tool_call %s at index %d",
                            tc_id, dup_idx,
                        )
        elif msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if i in consumed_indices:
                # Already placed after its assistant message - skip original position
                i += 1
                continue
            if tc_id not in expected_tool_call_ids:
                # Orphaned tool result - no matching tool_call exists
                dropped_orphans += 1
                changed = True
                logger.warning(
                    "transcript_repair: dropped orphaned tool result for tool_call_id %s at index %d",
                    tc_id, i,
                )
                i += 1
                continue
            # This tool result has a matching tool_call but was already placed
            # by the assistant message handler above - skip to avoid duplication
            i += 1
            continue
        else:
            repaired.append(msg)
            i += 1
            continue

        # Do not increment i here - it was already incremented above

    if not changed:
        return messages, 0, 0, 0, 0

    return repaired, added_synthetic, dropped_duplicates, dropped_orphans, moved


def repair_transcript(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function that runs all transcript repairs in sequence.

    Calls repair_tool_call_inputs first, then repair_tool_use_result_pairing.

    Args:
        messages: List of OpenAI-format messages.

    Returns:
        Repaired messages list. Returns the original list reference if no
        changes were needed.
    """
    messages, dropped_tc, dropped_asst = repair_tool_call_inputs(messages)
    if dropped_tc or dropped_asst:
        logger.info(
            "transcript_repair: tool_call_inputs phase - dropped %d tool_calls, %d assistant messages",
            dropped_tc, dropped_asst,
        )

    messages, synthetic, duplicates, orphans, moved_count = repair_tool_use_result_pairing(messages)
    if synthetic or duplicates or orphans or moved_count:
        logger.info(
            "transcript_repair: result_pairing phase - %d synthetic, %d duplicates dropped, %d orphans dropped, %d moved",
            synthetic, duplicates, orphans, moved_count,
        )

    return messages

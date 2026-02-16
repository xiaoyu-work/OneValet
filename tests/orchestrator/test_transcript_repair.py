"""Tests for onevalet.orchestrator.transcript_repair"""

import pytest

from onevalet.orchestrator.transcript_repair import (
    SYNTHETIC_TOOL_RESULT,
    repair_tool_call_inputs,
    repair_tool_use_result_pairing,
    repair_transcript,
)


# ── helpers ──


def _assistant(tool_calls, content=None, **extra):
    msg = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    msg.update(extra)
    return msg


def _tc(tc_id, name, arguments):
    return {"id": tc_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _tool_result(tc_id, content="ok"):
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def _user(text):
    return {"role": "user", "content": text}


def _system(text):
    return {"role": "system", "content": text}


# =========================================================================
# repair_tool_call_inputs
# =========================================================================


class TestRepairToolCallInputs:

    def test_no_changes_returns_original_ref(self):
        msgs = [
            _user("hi"),
            _assistant([_tc("1", "get_weather", '{"city":"tokyo"}')]),
            _tool_result("1"),
        ]
        result, dropped_tc, dropped_asst = repair_tool_call_inputs(msgs)
        assert result is msgs
        assert dropped_tc == 0
        assert dropped_asst == 0

    def test_drop_tool_call_with_none_arguments(self):
        msgs = [
            _assistant([_tc("1", "bad_tool", None)]),
        ]
        result, dropped_tc, dropped_asst = repair_tool_call_inputs(msgs)
        assert dropped_tc == 1
        assert dropped_asst == 1
        assert len(result) == 0

    def test_drop_tool_call_with_empty_string_arguments(self):
        msgs = [
            _assistant([_tc("1", "bad_tool", "")]),
        ]
        result, dropped_tc, dropped_asst = repair_tool_call_inputs(msgs)
        assert dropped_tc == 1
        assert dropped_asst == 1

    def test_drop_tool_call_with_missing_arguments_key(self):
        msgs = [
            _assistant([{"id": "1", "type": "function", "function": {"name": "x"}}]),
        ]
        result, dropped_tc, dropped_asst = repair_tool_call_inputs(msgs)
        assert dropped_tc == 1
        assert dropped_asst == 1

    def test_mixed_valid_and_invalid(self):
        msgs = [
            _assistant([
                _tc("1", "good_tool", '{"a":1}'),
                _tc("2", "bad_tool", None),
                _tc("3", "another_good", '{"b":2}'),
            ]),
        ]
        result, dropped_tc, dropped_asst = repair_tool_call_inputs(msgs)
        assert dropped_tc == 1
        assert dropped_asst == 0
        assert len(result) == 1
        assert len(result[0]["tool_calls"]) == 2
        ids = [tc["id"] for tc in result[0]["tool_calls"]]
        assert ids == ["1", "3"]

    def test_all_invalid_drops_entire_message(self):
        msgs = [
            _user("hi"),
            _assistant([_tc("1", "a", None), _tc("2", "b", "")]),
            _tool_result("1"),
        ]
        result, dropped_tc, dropped_asst = repair_tool_call_inputs(msgs)
        assert dropped_tc == 2
        assert dropped_asst == 1
        # user + tool_result remain
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "tool"

    def test_non_assistant_messages_pass_through(self):
        msgs = [_system("you are helpful"), _user("hi")]
        result, dropped_tc, dropped_asst = repair_tool_call_inputs(msgs)
        assert result is msgs
        assert dropped_tc == 0

    def test_assistant_without_tool_calls_passes_through(self):
        msgs = [_user("hi"), {"role": "assistant", "content": "hello"}]
        result, _, _ = repair_tool_call_inputs(msgs)
        assert result is msgs


# =========================================================================
# repair_tool_use_result_pairing
# =========================================================================


class TestRepairToolUseResultPairing:

    def test_no_changes_returns_original_ref(self):
        msgs = [
            _assistant([_tc("1", "weather", '{"c":"tokyo"}')]),
            _tool_result("1"),
        ]
        result, synth, dup, orphan, moved = repair_tool_use_result_pairing(msgs)
        assert result is msgs
        assert (synth, dup, orphan, moved) == (0, 0, 0, 0)

    def test_missing_result_inserts_synthetic(self):
        msgs = [
            _assistant([_tc("1", "weather", '{"c":"tokyo"}')]),
            # no tool result for "1"
        ]
        result, synth, dup, orphan, moved = repair_tool_use_result_pairing(msgs)
        assert synth == 1
        assert len(result) == 2
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "1"
        assert SYNTHETIC_TOOL_RESULT in result[1]["content"]

    def test_duplicate_results_keeps_first(self):
        msgs = [
            _assistant([_tc("1", "weather", '{"c":"tokyo"}')]),
            _tool_result("1", "first"),
            _tool_result("1", "second"),
        ]
        result, synth, dup, orphan, moved = repair_tool_use_result_pairing(msgs)
        assert dup == 1
        assert synth == 0
        # assistant + one tool result
        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0]["content"] == "first"

    def test_orphaned_result_dropped(self):
        msgs = [
            _user("hi"),
            _tool_result("orphan_id", "i have no parent"),
        ]
        result, synth, dup, orphan, moved = repair_tool_use_result_pairing(msgs)
        assert orphan == 1
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_stop_reason_error_skipped(self):
        msgs = [
            _assistant([_tc("1", "weather", '{"c":"x"}')], stop_reason="error"),
            # no tool result — but should NOT get synthetic because stop_reason=error
        ]
        result, synth, dup, orphan, moved = repair_tool_use_result_pairing(msgs)
        assert synth == 0
        assert len(result) == 1
        assert result[0]["role"] == "assistant"

    def test_stop_reason_aborted_skipped(self):
        msgs = [
            _assistant([_tc("1", "weather", '{"c":"x"}')], stop_reason="aborted"),
        ]
        result, synth, _, _, _ = repair_tool_use_result_pairing(msgs)
        assert synth == 0

    def test_displaced_result_moved(self):
        msgs = [
            _assistant([_tc("1", "a", '{}'), _tc("2", "b", '{}')]),
            _tool_result("2"),  # result for "2" before "1"
            _tool_result("1"),
        ]
        result, synth, dup, orphan, moved = repair_tool_use_result_pairing(msgs)
        # Both results should be placed after assistant
        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results) == 2
        # Order should match tool_calls order: "1" first, then "2"
        assert tool_results[0]["tool_call_id"] == "1"
        assert tool_results[1]["tool_call_id"] == "2"

    def test_multiple_assistant_messages(self):
        msgs = [
            _assistant([_tc("1", "a", '{}')]),
            _tool_result("1"),
            _user("thanks"),
            _assistant([_tc("2", "b", '{}')]),
            # missing result for "2"
        ]
        result, synth, dup, orphan, moved = repair_tool_use_result_pairing(msgs)
        assert synth == 1
        assert result[-1]["tool_call_id"] == "2"
        assert SYNTHETIC_TOOL_RESULT in result[-1]["content"]

    def test_empty_messages(self):
        result, synth, dup, orphan, moved = repair_tool_use_result_pairing([])
        assert result == []
        assert (synth, dup, orphan, moved) == (0, 0, 0, 0)


# =========================================================================
# repair_transcript (integration)
# =========================================================================


class TestRepairTranscript:

    def test_clean_transcript_returns_original_ref(self):
        msgs = [
            _user("hi"),
            _assistant([_tc("1", "x", '{"a":1}')]),
            _tool_result("1"),
        ]
        result = repair_transcript(msgs)
        assert result is msgs

    def test_both_phases_applied(self):
        msgs = [
            _assistant([_tc("1", "good", '{"a":1}'), _tc("2", "bad", None)]),
            _tool_result("1"),
            _tool_result("orphan_id"),
        ]
        result = repair_transcript(msgs)
        # Phase 1: "bad" tool_call dropped, assistant rebuilt with 1 tc
        # Phase 2: orphan result dropped
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 1
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "1"

    def test_synthetic_inserted_after_invalid_dropped(self):
        msgs = [
            _assistant([_tc("1", "ok", '{"a":1}'), _tc("2", "bad", "")]),
            # result for "1" present, "2" was dropped so no result needed
        ]
        result = repair_transcript(msgs)
        # Phase 1: "bad" dropped. Phase 2: "1" missing result → synthetic
        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_call_id"] == "1"

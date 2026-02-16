"""Tests for onevalet.orchestrator.context_manager"""

import pytest

from onevalet.orchestrator.context_manager import ContextManager
from onevalet.orchestrator.react_config import ReactLoopConfig


@pytest.fixture
def default_cm():
    return ContextManager(ReactLoopConfig())


@pytest.fixture
def small_cm():
    """Context manager with a tiny context window for easy threshold testing."""
    return ContextManager(ReactLoopConfig(
        context_token_limit=100,      # 100 tokens = ~400 chars
        context_trim_threshold=0.5,   # trim at 50 tokens = ~200 chars
        max_history_messages=3,
        max_tool_result_share=0.3,
        max_tool_result_chars=200,
    ))


# =========================================================================
# estimate_tokens
# =========================================================================


class TestEstimateTokens:

    def test_string_content(self, default_cm):
        msgs = [{"role": "user", "content": "a" * 400}]
        assert default_cm.estimate_tokens(msgs) == 100

    def test_none_content(self, default_cm):
        msgs = [{"role": "assistant", "content": None}]
        assert default_cm.estimate_tokens(msgs) == 0

    def test_empty_messages(self, default_cm):
        assert default_cm.estimate_tokens([]) == 0

    def test_list_content_with_text_parts(self, default_cm):
        msgs = [{"role": "tool", "content": [
            {"text": "a" * 80},
            {"text": "b" * 120},
        ]}]
        assert default_cm.estimate_tokens(msgs) == 50  # 200 / 4

    def test_list_content_with_string_parts(self, default_cm):
        msgs = [{"role": "tool", "content": ["hello", "world"]}]
        # 5 + 5 = 10 chars / 4 = 2 tokens
        assert default_cm.estimate_tokens(msgs) == 2

    def test_list_content_with_content_key(self, default_cm):
        msgs = [{"role": "tool", "content": [{"content": "a" * 40}]}]
        assert default_cm.estimate_tokens(msgs) == 10

    def test_mixed_message_types(self, default_cm):
        msgs = [
            {"role": "system", "content": "a" * 100},
            {"role": "user", "content": "b" * 100},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": "c" * 200},
        ]
        assert default_cm.estimate_tokens(msgs) == 100  # 400 / 4

    def test_image_url_parts_ignored(self, default_cm):
        msgs = [{"role": "tool", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]}]
        assert default_cm.estimate_tokens(msgs) == 0


# =========================================================================
# truncate_tool_result
# =========================================================================


class TestTruncateToolResult:

    def test_under_limit_unchanged(self, small_cm):
        result = "short text"
        assert small_cm.truncate_tool_result(result) == result

    def test_over_limit_truncated_with_marker(self, small_cm):
        # small_cm: limit = min(100*0.3*4, 200) = min(120, 200) = 120
        result = "x" * 200
        truncated = small_cm.truncate_tool_result(result)
        assert truncated.endswith("\n[...truncated]")
        assert len(truncated) < 200

    def test_prefers_newline_boundary(self, small_cm):
        # max_chars = 120
        line1 = "a" * 80 + "\n"   # 81 chars
        line2 = "b" * 100          # 100 chars
        result = line1 + line2     # 181 chars, over 120 limit
        truncated = small_cm.truncate_tool_result(result)
        # Should cut at the newline (pos 80), not at char 120
        assert truncated == line1 + "\n[...truncated]"

    def test_no_newline_hard_cut(self, small_cm):
        # max_chars = 120, no newlines anywhere
        result = "x" * 200
        truncated = small_cm.truncate_tool_result(result)
        # Hard cut at 120 chars + marker
        assert truncated == "x" * 120 + "\n[...truncated]"

    def test_newline_too_early_uses_hard_cut(self, small_cm):
        # max_chars = 120, newline at position 10 (< 120//2=60)
        result = "a" * 10 + "\n" + "b" * 200
        truncated = small_cm.truncate_tool_result(result)
        # Newline at 10 is < 60, so hard cut at 120
        assert truncated == result[:120] + "\n[...truncated]"

    def test_exact_limit_unchanged(self, small_cm):
        result = "x" * 120
        assert small_cm.truncate_tool_result(result) == result

    def test_empty_string(self, small_cm):
        assert small_cm.truncate_tool_result("") == ""


# =========================================================================
# trim_if_needed
# =========================================================================


class TestTrimIfNeeded:

    def test_under_threshold_returns_original(self, small_cm):
        msgs = [{"role": "user", "content": "hi"}]
        result = small_cm.trim_if_needed(msgs)
        assert result is msgs

    def test_over_threshold_trims(self, small_cm):
        # threshold = 100 * 0.5 = 50 tokens = 200 chars
        msgs = [
            {"role": "system", "content": "sys"},
            *[{"role": "user", "content": f"msg{i}" + "x" * 50} for i in range(10)],
        ]
        result = small_cm.trim_if_needed(msgs)
        # Should keep system + last 3
        assert result[0]["role"] == "system"
        assert len(result) == 4  # system + 3 recent


# =========================================================================
# truncate_all_tool_results
# =========================================================================


class TestTruncateAllToolResults:

    def test_truncates_string_tool_content(self, small_cm):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "1", "content": "x" * 200},
        ]
        result = small_cm.truncate_all_tool_results(msgs)
        assert result[0]["content"] == "hi"  # user untouched
        assert result[1]["content"].endswith("[...truncated]")

    def test_truncates_list_content_text_parts(self, small_cm):
        msgs = [
            {"role": "tool", "tool_call_id": "1", "content": [
                {"text": "y" * 200},
                {"type": "image_url"},  # non-text part untouched
            ]},
        ]
        result = small_cm.truncate_all_tool_results(msgs)
        parts = result[0]["content"]
        assert parts[0]["text"].endswith("[...truncated]")
        assert parts[1] == {"type": "image_url"}

    def test_does_not_mutate_original(self, small_cm):
        original = {"role": "tool", "tool_call_id": "1", "content": "x" * 200}
        msgs = [original]
        result = small_cm.truncate_all_tool_results(msgs)
        assert original["content"] == "x" * 200  # original unchanged


# =========================================================================
# force_trim
# =========================================================================


class TestForceTrim:

    def test_keeps_system_plus_last_5(self, default_cm):
        msgs = [
            {"role": "system", "content": "sys"},
            *[{"role": "user", "content": f"msg{i}"} for i in range(20)],
        ]
        result = default_cm.force_trim(msgs)
        assert result[0]["role"] == "system"
        assert len(result) == 6  # system + 5

    def test_fewer_than_5_keeps_all(self, default_cm):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        result = default_cm.force_trim(msgs)
        assert len(result) == 3

    def test_no_system_message(self, default_cm):
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
        result = default_cm.force_trim(msgs)
        assert len(result) == 5
        assert result[0]["role"] == "user"

    def test_empty_messages(self, default_cm):
        assert default_cm.force_trim([]) == []


# =========================================================================
# split_for_summarization
# =========================================================================


class TestSplitForSummarization:

    def test_under_threshold_returns_none(self, small_cm):
        msgs = [{"role": "user", "content": "hi"}]
        assert small_cm.split_for_summarization(msgs) is None

    def test_empty_returns_none(self, small_cm):
        assert small_cm.split_for_summarization([]) is None

    def test_over_threshold_but_few_messages_returns_none(self, small_cm):
        # Over threshold but only 2 rest messages, keep=3 → rest <= keep
        msgs = [
            {"role": "system", "content": "x" * 400},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        assert small_cm.split_for_summarization(msgs) is None

    def test_over_threshold_splits_correctly(self, small_cm):
        msgs = [
            {"role": "system", "content": "sys"},
            *[{"role": "user", "content": "x" * 50} for _ in range(10)],
        ]
        result = small_cm.split_for_summarization(msgs)
        assert result is not None
        system, old, recent = result
        assert len(system) == 1
        assert system[0]["content"] == "sys"
        assert len(recent) == 3  # max_history_messages
        assert len(old) == 7    # 10 - 3

    def test_no_system_message(self, small_cm):
        msgs = [{"role": "user", "content": "x" * 50} for _ in range(10)]
        result = small_cm.split_for_summarization(msgs)
        assert result is not None
        system, old, recent = result
        assert system == []
        assert len(recent) == 3
        assert len(old) == 7


# =========================================================================
# build_summarized_messages
# =========================================================================


class TestBuildSummarizedMessages:

    def test_basic_reconstruction(self):
        system = [{"role": "system", "content": "sys"}]
        recent = [{"role": "user", "content": "recent"}]
        result = ContextManager.build_summarized_messages(system, "summary text", recent)
        assert len(result) == 3
        assert result[0] == system[0]
        assert result[1]["role"] == "user"
        assert "summary text" in result[1]["content"]
        assert result[2] == recent[0]

    def test_empty_system(self):
        result = ContextManager.build_summarized_messages([], "summary", [{"role": "user", "content": "x"}])
        assert len(result) == 2
        assert result[0]["role"] == "user"  # summary
        assert result[1]["content"] == "x"


# =========================================================================
# _keep_recent (internal, tested via public methods but also directly)
# =========================================================================


class TestKeepRecent:

    def test_with_system_prompt(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "1"},
            {"role": "user", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "user", "content": "4"},
        ]
        result = ContextManager._keep_recent(msgs, keep=2)
        assert len(result) == 3  # system + last 2
        assert result[0]["content"] == "sys"
        assert result[1]["content"] == "3"
        assert result[2]["content"] == "4"

    def test_without_system_prompt(self):
        msgs = [{"role": "user", "content": str(i)} for i in range(5)]
        result = ContextManager._keep_recent(msgs, keep=2)
        assert len(result) == 2
        assert result[0]["content"] == "3"

    def test_fewer_than_keep(self):
        msgs = [{"role": "user", "content": "a"}]
        result = ContextManager._keep_recent(msgs, keep=10)
        assert result == msgs

    def test_empty(self):
        result = ContextManager._keep_recent([], keep=5)
        assert result == []

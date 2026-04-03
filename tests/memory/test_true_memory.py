"""Tests for true-memory proposal extraction helpers."""

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from onevalet.memory.true_memory import (
    extract_true_memory_proposals,
    format_true_memory_for_prompt,
    looks_like_true_memory_candidate,
)


@dataclass
class MockLLMResponse:
    content: str


class TestLooksLikeTrueMemoryCandidate:
    def test_accepts_direct_preference_statement(self):
        assert looks_like_true_memory_candidate(
            "Remember that I prefer aisle seats when I fly.",
        )

    def test_accepts_identity_statement(self):
        assert looks_like_true_memory_candidate("My name is Alice Johnson.")

    def test_accepts_self_description(self):
        assert looks_like_true_memory_candidate("I am a software engineer in Seattle.")

    def test_rejects_question(self):
        assert not looks_like_true_memory_candidate("What should I eat for lunch?")

    def test_rejects_task_request(self):
        assert not looks_like_true_memory_candidate("Send an email to Bob about the meeting.")

    def test_rejects_short_messages(self):
        assert not looks_like_true_memory_candidate("hi")

    def test_rejects_empty(self):
        assert not looks_like_true_memory_candidate("")

    def test_rejects_none(self):
        assert not looks_like_true_memory_candidate(None)


class TestFormatTrueMemoryForPrompt:
    def test_formats_facts(self):
        facts = [
            {"summary": "User prefers aisle seats."},
            {"summary": "User lives in Seattle."},
        ]
        result = format_true_memory_for_prompt(facts)
        assert "- User prefers aisle seats." in result
        assert "- User lives in Seattle." in result

    def test_empty_returns_empty(self):
        assert format_true_memory_for_prompt([]) == ""
        assert format_true_memory_for_prompt(None) == ""

    def test_skips_facts_without_summary(self):
        facts = [{"namespace": "travel", "fact_key": "seat", "value": "aisle"}]
        result = format_true_memory_for_prompt(facts)
        assert result == ""  # No summary → skipped, no internal IDs leak


class TestExtractTrueMemoryProposals:
    @pytest.mark.asyncio
    async def test_extracts_structured_llm_proposals(self):
        llm_client = AsyncMock()
        llm_client.chat_completion.return_value = MockLLMResponse(
            content="""{
              "should_store": true,
              "proposals": [
                {
                  "operation": "upsert",
                  "namespace": "travel",
                  "fact_key": "flight_seat",
                  "value": {"seat": "aisle"},
                  "summary": "User prefers aisle seats on flights.",
                  "confidence": 0.97,
                  "source_type": "user_direct",
                  "reason": "Directly stated travel preference."
                }
              ]
            }""",
        )

        proposals = await extract_true_memory_proposals(
            llm_client,
            user_message="Remember that I prefer aisle seats when I fly.",
        )

        assert len(proposals) == 1
        assert proposals[0]["namespace"] == "travel"
        assert proposals[0]["fact_key"] == "flight_seat"
        assert proposals[0]["confidence"] == 0.97
        assert proposals[0]["source_type"] == "user_direct"

    @pytest.mark.asyncio
    async def test_falls_back_to_rules_when_llm_fails(self):
        llm_client = AsyncMock()
        llm_client.chat_completion.side_effect = RuntimeError("boom")

        proposals = await extract_true_memory_proposals(
            llm_client,
            user_message="My name is Alice Johnson.",
        )

        assert len(proposals) == 1
        assert proposals[0]["namespace"] == "identity"
        assert proposals[0]["fact_key"] == "full_name"
        assert proposals[0]["value"] == "Alice Johnson"

    @pytest.mark.asyncio
    async def test_skips_non_candidates_without_calling_llm(self):
        llm_client = AsyncMock()

        proposals = await extract_true_memory_proposals(
            llm_client,
            user_message="What time is my meeting tomorrow?",
        )

        assert proposals == []
        llm_client.chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_extracts_location(self):
        llm_client = AsyncMock()
        llm_client.chat_completion.side_effect = RuntimeError("boom")

        proposals = await extract_true_memory_proposals(
            llm_client,
            user_message="I live in Seattle",
        )

        assert len(proposals) == 1
        assert proposals[0]["namespace"] == "identity"
        assert proposals[0]["fact_key"] == "home_location"

    @pytest.mark.asyncio
    async def test_handles_empty_llm_response(self):
        llm_client = AsyncMock()
        llm_client.chat_completion.return_value = MockLLMResponse(
            content='{"should_store": false, "proposals": []}',
        )

        proposals = await extract_true_memory_proposals(
            llm_client,
            user_message="Remember that I prefer tea.",
        )
        # LLM says nothing to store, fallback also has nothing → empty
        assert proposals == []

    @pytest.mark.asyncio
    async def test_fallback_extracts_feedback_correction(self):
        llm_client = AsyncMock()
        llm_client.chat_completion.side_effect = RuntimeError("boom")

        proposals = await extract_true_memory_proposals(
            llm_client,
            user_message="Stop asking me to confirm every email before sending",
        )

        assert len(proposals) == 1
        assert proposals[0]["namespace"] == "feedback"
        assert proposals[0]["source_type"] == "user_correction"
        assert "stop" in proposals[0]["summary"].lower()
        assert proposals[0]["why"] is not None
        assert proposals[0]["how_to_apply"] is not None

    @pytest.mark.asyncio
    async def test_fallback_extracts_feedback_confirmation(self):
        llm_client = AsyncMock()
        llm_client.chat_completion.side_effect = RuntimeError("boom")

        proposals = await extract_true_memory_proposals(
            llm_client,
            user_message="Yes exactly, that's the right approach",
        )

        assert len(proposals) == 1
        assert proposals[0]["namespace"] == "feedback"
        assert proposals[0]["source_type"] == "user_confirmation"

    @pytest.mark.asyncio
    async def test_llm_extracts_feedback_with_why_and_how(self):
        llm_client = AsyncMock()
        llm_client.chat_completion.return_value = MockLLMResponse(
            content="""{
              "should_store": true,
              "proposals": [
                {
                  "operation": "upsert",
                  "namespace": "feedback",
                  "fact_key": "no_email_confirmation",
                  "value": {"rule": "skip confirmation for routine emails"},
                  "summary": "User wants assistant to skip email confirmation prompts.",
                  "confidence": 0.92,
                  "source_type": "user_correction",
                  "reason": "User explicitly told the assistant to stop asking.",
                  "why": "User finds confirmation prompts slow and unnecessary for routine emails.",
                  "how_to_apply": "Send routine emails directly without asking for confirmation. Still confirm for emails to new recipients or with attachments."
                }
              ]
            }""",
        )

        proposals = await extract_true_memory_proposals(
            llm_client,
            user_message="Stop asking me to confirm every email before sending, just send it",
        )

        assert len(proposals) == 1
        assert proposals[0]["namespace"] == "feedback"
        assert proposals[0]["why"] is not None
        assert "confirmation" in proposals[0]["why"].lower()
        assert proposals[0]["how_to_apply"] is not None
        assert "routine" in proposals[0]["how_to_apply"].lower()


class TestFormatTrueMemoryWithFeedback:
    def test_feedback_memory_shows_why_and_apply(self):
        facts = [
            {
                "summary": "User wants assistant to skip email confirmation.",
                "namespace": "feedback",
                "why": "User finds confirmations slow for routine emails.",
                "how_to_apply": "Send routine emails directly without confirmation.",
            },
            {
                "summary": "User prefers aisle seats.",
                "namespace": "travel",
            },
        ]
        result = format_true_memory_for_prompt(facts)
        assert "Why:" in result
        assert "Apply:" in result
        # Non-feedback memory should NOT have Why/Apply
        lines = result.strip().split("\n")
        assert "Why:" not in lines[1]

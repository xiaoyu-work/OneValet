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

    def test_falls_back_to_namespace_key(self):
        facts = [{"namespace": "travel", "fact_key": "seat", "value": "aisle"}]
        result = format_true_memory_for_prompt(facts)
        assert "travel.seat" in result


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

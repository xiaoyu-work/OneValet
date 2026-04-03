"""Integration-style unit tests for orchestrator memory wiring."""

import asyncio
from unittest.mock import MagicMock

import pytest

from onevalet.orchestrator.orchestrator import Orchestrator
from onevalet.result import AgentResult, AgentStatus


class DummyMomex:
    def __init__(self, search_results=None):
        self.search_results = search_results or []
        self.search_calls = []
        self.add_calls = []

    async def search(self, tenant_id: str, query: str, limit: int = 5):
        self.search_calls.append((tenant_id, query, limit))
        return list(self.search_results)

    async def add(self, tenant_id: str, messages, infer: bool = True):
        self.add_calls.append(
            {
                "tenant_id": tenant_id,
                "messages": list(messages),
                "infer": infer,
            }
        )


@pytest.mark.asyncio
async def test_build_llm_messages_includes_session_memory_and_recall():
    momex = DummyMomex(
        search_results=[
            {"text": "User prefers aisle seats", "type": "preference", "score": 0.9},
        ]
    )
    orchestrator = Orchestrator(momex=momex, llm_client=MagicMock())
    context = await orchestrator.prepare_context(
        "tenant-1",
        "Remember that I prefer aisle seats on flights.",
        {"session_id": "session-1"},
    )

    messages = await orchestrator._build_llm_messages(
        context,
        "Remember that I prefer aisle seats on flights.",
    )
    system_prompt = messages[0]["content"]
    assert "[Session Working Memory]" in system_prompt
    assert "Objective: Remember that I prefer aisle seats on flights." in system_prompt
    assert "[Relevant Memories]" in system_prompt
    assert "User prefers aisle seats" in system_prompt


@pytest.mark.asyncio
async def test_post_process_skips_transient_memory_writes():
    momex = DummyMomex()
    orchestrator = Orchestrator(momex=momex, llm_client=MagicMock())
    context = await orchestrator.prepare_context(
        "tenant-1",
        "thanks",
        {"session_id": "session-1"},
    )

    result = AgentResult(
        agent_type="Orchestrator",
        status=AgentStatus.COMPLETED,
        raw_message="You're welcome!",
    )
    updated = await orchestrator.post_process(result, context)
    await asyncio.sleep(0)

    assert updated.metadata["memory_write"]["stored"] is False
    assert momex.add_calls == []


@pytest.mark.asyncio
async def test_post_process_persists_persistent_memory_turns():
    momex = DummyMomex()
    orchestrator = Orchestrator(momex=momex, llm_client=MagicMock())
    context = await orchestrator.prepare_context(
        "tenant-1",
        "Please remember that I prefer red-eye flights.",
        {"session_id": "session-1"},
    )

    result = AgentResult(
        agent_type="Orchestrator",
        status=AgentStatus.COMPLETED,
        raw_message="Understood — I will remember that you prefer red-eye flights.",
    )
    updated = await orchestrator.post_process(result, context)
    await asyncio.sleep(0)

    assert updated.metadata["memory_write"]["stored"] is True
    assert len(momex.add_calls) == 1
    assert momex.add_calls[0]["tenant_id"] == "tenant-1"

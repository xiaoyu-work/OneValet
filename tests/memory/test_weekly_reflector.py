import json
from datetime import date

import pytest

from koa.builtin_agents.reflection import weekly_reflector_agent
from koa.builtin_agents.reflection.weekly_reflector_agent import WeeklyReflectorAgent
from koa.memory.lifecycle.weekly_reflector import WeeklyReflection, run_weekly_reflection
from koa.message import Message
from koa.result import AgentStatus


class FakeEpisodeMemory:
    def __init__(self):
        self.writes = []

    async def recall_recent_episodes(self, tenant_id, *, subkind, limit):
        return [
            {
                "text": "Daily log: user moved recurring planning from Notion to Linear.",
                "metadata": {
                    "local_date": "2026-05-01",
                    "payload": {
                        "messages": {"total": 9},
                        "tools": {"linear": 4},
                        "state": {"mood": "focused"},
                    },
                },
            }
        ]

    async def write_episode(self, **kwargs):
        self.writes.append(kwargs)


@pytest.mark.asyncio
async def test_weekly_reflection_uses_existing_memory_and_emits_patches():
    episode_memory = FakeEpisodeMemory()
    captured = {}

    async def llm_call(system_prompt, user_prompt):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return json.dumps(
            {
                "highlight": "User consolidated planning into Linear.",
                "mood_trend": "stable",
                "top_topics": ["planning", "Linear"],
                "episodes": [],
                "facts": [
                    {
                        "operation": "upsert",
                        "namespace": "preference",
                        "fact_key": "Planning Tool",
                        "value": {"tool": "Linear"},
                        "summary": "User prefers Linear for recurring planning.",
                        "confidence": 0.7,
                        "why": "The weekly logs show repeated planning activity in Linear.",
                        "how_to_apply": "Use Linear when helping with recurring planning.",
                    },
                    {
                        "operation": "upsert",
                        "namespace": "preference",
                        "fact_key": "planning_tool",
                        "value": {"tool": "Linear"},
                        "summary": "User now prefers Linear for recurring planning.",
                        "confidence": 0.86,
                        "why": "The week shows a replacement from Notion to Linear.",
                        "how_to_apply": "Default planning workflows to Linear.",
                    },
                    {
                        "operation": "revoke",
                        "namespace": "routine",
                        "fact_key": "notion_weekly_planning",
                        "value": {"tool": "Notion"},
                        "summary": "User no longer uses Notion for weekly planning.",
                        "confidence": 0.82,
                        "why": "The week clearly replaced Notion planning with Linear.",
                        "how_to_apply": "Do not route planning work to Notion by default.",
                    },
                    {
                        "operation": "upsert",
                        "namespace": "project",
                        "fact_key": "weak_signal",
                        "summary": "This should be dropped.",
                        "confidence": 0.4,
                    },
                ],
            }
        )

    reflection = await run_weekly_reflection(
        "user-1",
        date(2026, 5, 3),
        llm_call=llm_call,
        episode_memory=episode_memory,
        existing_true_memory=[
            {
                "namespace": "preference",
                "fact_key": "planning_tool",
                "summary": "User prefers Notion for weekly planning.",
                "value": {"tool": "Notion"},
            }
        ],
    )

    assert isinstance(reflection, WeeklyReflection)
    assert "existing_true_memory" in captured["user_prompt"]
    assert "planning_tool" in captured["user_prompt"]
    assert len(reflection.fact_proposals) == 2

    by_key = {(p["namespace"], p["fact_key"]): p for p in reflection.fact_proposals}
    assert by_key[("preference", "planning_tool")]["value"] == {"tool": "Linear"}
    assert by_key[("preference", "planning_tool")]["confidence"] == 0.86
    assert by_key[("routine", "notion_weekly_planning")]["operation"] == "revoke"
    assert by_key[("routine", "notion_weekly_planning")]["value"] is None

    weekly_writes = [w for w in episode_memory.writes if w["subkind"] == "weekly_reflection"]
    assert weekly_writes
    assert "Memory maintenance proposals" in weekly_writes[0]["summary"]


@pytest.mark.asyncio
async def test_weekly_reflector_agent_returns_proposals_in_metadata(monkeypatch):
    captured = {}
    proposal = {
        "operation": "upsert",
        "namespace": "preference",
        "fact_key": "planning_tool",
        "summary": "User prefers Linear for planning.",
        "confidence": 0.9,
    }

    async def fake_run_weekly_reflection(
        user_id,
        week_end,
        llm_call,
        episode_memory,
        *,
        existing_true_memory=None,
    ):
        captured["existing_true_memory"] = existing_true_memory
        return WeeklyReflection(
            week_start=date(2026, 4, 27),
            week_end=date(2026, 5, 3),
            highlight="Planning moved to Linear.",
            mood_trend="stable",
            top_topics=["planning"],
            episodes_written=1,
            fact_proposals=[proposal],
            raw_response={},
        )

    monkeypatch.setattr(
        weekly_reflector_agent,
        "run_weekly_reflection",
        fake_run_weekly_reflection,
    )
    agent = WeeklyReflectorAgent(
        tenant_id="user-1",
        llm_client=object(),
        context_hints={
            "momex": object(),
            "user_id": "user-1",
            "true_memory": [{"namespace": "preference", "fact_key": "planning_tool"}],
        },
    )

    result = await agent.reply(Message(name="cron", content="run", role="user"))

    assert result.status == AgentStatus.COMPLETED
    assert result.metadata["true_memory_proposals"] == [proposal]
    assert captured["existing_true_memory"] == [
        {"namespace": "preference", "fact_key": "planning_tool"}
    ]

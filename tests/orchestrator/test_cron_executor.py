from unittest.mock import AsyncMock

import pytest

from koa.result import AgentResult, AgentStatus
from koa.triggers.cron.executor import CronExecutor
from koa.triggers.cron.models import AgentTurnPayload, CronJob, SessionTarget


class FakeAgent:
    agent_id = "FakeAgent_1"

    def __init__(self):
        self.messages = []

    async def reply(self, msg):
        self.messages.append(msg)
        return AgentResult(
            agent_type="FakeAgent",
            status=AgentStatus.COMPLETED,
            raw_message="agent ran",
            metadata={"true_memory_proposals": [{"namespace": "preference"}]},
        )


class FakeAgentPool:
    def __init__(self):
        self.removed = []

    async def remove_agent(self, tenant_id, agent_id):
        self.removed.append((tenant_id, agent_id))


class FakeOrchestrator:
    def __init__(self):
        self.database = None
        self.momex = object()
        self.trigger_engine = None
        self.agent_pool = FakeAgentPool()
        self.agent = FakeAgent()
        self.created = []
        self.handle_message = AsyncMock(side_effect=AssertionError("should not use ReAct path"))

    async def create_agent(self, **kwargs):
        self.created.append(kwargs)
        return self.agent


@pytest.mark.asyncio
async def test_cron_executor_runs_named_agent_directly():
    orchestrator = FakeOrchestrator()
    executor = CronExecutor(
        orchestrator=orchestrator,
        store=object(),
        run_log=object(),
        delivery=object(),
    )
    job = CronJob(
        id="job-1",
        agent_id="WeeklyReflectorAgent",
        user_id="user-1",
        name="sensing.weekly_reflection",
        session_target=SessionTarget.ISOLATED,
        payload=AgentTurnPayload(message="Run WeeklyReflectorAgent."),
    )

    summary, error = await executor._execute_core(job)

    assert error is None
    assert summary == "agent ran"
    orchestrator.handle_message.assert_not_called()
    assert orchestrator.created[0]["agent_type"] == "WeeklyReflectorAgent"
    assert orchestrator.created[0]["context_hints"]["momex"] is orchestrator.momex
    assert orchestrator.created[0]["context_hints"]["user_id"] == "user-1"
    assert orchestrator.agent.messages[0].metadata["direct_agent"] is True
    assert orchestrator.agent_pool.removed == [("user-1", "FakeAgent_1")]

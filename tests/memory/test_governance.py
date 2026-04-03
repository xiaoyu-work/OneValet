"""Tests for memory governance and session working memory."""

from onevalet.memory.governance import MemoryGovernance
from onevalet.memory.session_memory import SessionMemoryManager


class TestMemoryGovernance:

    def test_select_recalled_memories_deduplicates_and_caps(self):
        governance = MemoryGovernance(max_prompt_memories=2, max_prompt_chars=500)
        recalled = [
            {"text": "User prefers window seats", "type": "preference", "score": 0.9},
            {"text": "User prefers window seats", "type": "preference", "score": 0.7},
            {"text": "User lives in Seattle", "type": "profile", "score": 0.8},
        ]
        selected = governance.select_recalled_memories(recalled)
        assert len(selected) == 2
        assert selected[0]["text"] == "User prefers window seats"
        assert selected[1]["text"] == "User lives in Seattle"

    def test_decide_storage_skips_transient_turn(self):
        governance = MemoryGovernance()
        decision = governance.decide_storage(
            user_message="thanks",
            assistant_message="You're welcome!",
            result_status="completed",
        )
        assert decision.should_store is False
        assert "transient" in decision.reason

    def test_decide_storage_keeps_persistent_preference(self):
        governance = MemoryGovernance()
        decision = governance.decide_storage(
            user_message="Please remember that I prefer aisle seats on flights.",
            assistant_message="Got it — I will remember that you prefer aisle seats.",
            result_status="completed",
        )
        assert decision.should_store is True
        assert "persistent-signal" in decision.tags


class TestSessionMemoryManager:

    def test_prepare_session_sets_objective_and_constraint(self):
        manager = SessionMemoryManager()
        state = manager.prepare_session(
            "session-1",
            "Plan a trip to Tokyo, but don't book anything yet.",
            has_active_agents=False,
        )
        assert state["objective"].startswith("Plan a trip to Tokyo")
        assert state["constraints"] == ["Plan a trip to Tokyo, but don't book anything yet."]

    def test_update_from_result_tracks_pending_and_findings(self):
        manager = SessionMemoryManager()
        manager.prepare_session("session-1", "Help me draft an email.", has_active_agents=False)

        pending = manager.update_from_result(
            "session-1",
            user_message="Help me draft an email.",
            assistant_message="Who should I send it to?",
            result_status="waiting_for_input",
        )
        assert pending["pending_questions"] == ["Who should I send it to?"]

        completed = manager.update_from_result(
            "session-1",
            user_message="Send it to Sam.",
            assistant_message="Drafted the email to Sam and kept the tone concise.",
            result_status="completed",
            tool_calls=[{"name": "draft_email"}, {"name": "lookup_contact"}],
        )
        assert completed["pending_questions"] == []
        assert completed["recent_findings"][-1].startswith("Drafted the email to Sam")
        assert completed["recent_tools"] == ["draft_email", "lookup_contact"]

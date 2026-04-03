"""Tests for runtime execution policy checks."""

from onevalet.models import AgentTool
from onevalet.orchestrator.execution_policy import ExecutionPolicyEngine


def _make_tool(**overrides) -> AgentTool:
    kwargs = {
        "name": "send_email",
        "description": "Send an email",
        "parameters": {"type": "object", "properties": {}},
        "executor": lambda *_args, **_kwargs: None,
        "risk_level": "write",
        "needs_approval": True,
        "category": "communication",
    }
    kwargs.update(overrides)
    return AgentTool(
        **kwargs,
    )


class TestExecutionPolicyEngine:

    def test_blocks_denied_tool(self):
        engine = ExecutionPolicyEngine()
        decision = engine.evaluate(
            _make_tool(),
            request_context={"metadata": {"permissions": {"denied_tools": ["send_email"]}}},
        )
        assert decision.allowed is False
        assert "denied" in decision.reason

    def test_blocks_write_actions_in_read_only_mode(self):
        engine = ExecutionPolicyEngine()
        decision = engine.evaluate(
            _make_tool(),
            request_context={"metadata": {"permissions": {"read_only_mode": True}}},
        )
        assert decision.allowed is False
        assert "read_only_mode" in decision.reason

    def test_requires_approval_when_not_preapproved(self):
        engine = ExecutionPolicyEngine()
        decision = engine.evaluate(_make_tool())
        assert decision.allowed is True
        assert decision.require_approval is True

    def test_approved_tool_skips_extra_approval_requirement(self):
        engine = ExecutionPolicyEngine()
        decision = engine.evaluate(
            _make_tool(),
            request_context={"metadata": {"permissions": {"approved_tools": ["send_email"]}}},
        )
        assert decision.allowed is True
        assert decision.require_approval is False

    def test_enforces_tier_and_feature_flag(self):
        engine = ExecutionPolicyEngine()
        tool = _make_tool(
            needs_approval=False,
            enabled_tiers=["pro"],
            requires_feature_flag="email-write",
        )
        denied = engine.evaluate(
            tool,
            request_context={
                "metadata": {
                    "permissions": {
                        "user_tier": "starter",
                        "feature_flags": ["email-write"],
                    }
                }
            },
        )
        assert denied.allowed is False
        assert "tier" in denied.reason

        allowed = engine.evaluate(
            tool,
            request_context={
                "metadata": {
                    "permissions": {
                        "user_tier": "pro",
                        "feature_flags": ["email-write"],
                    }
                }
            },
        )
        assert allowed.allowed is True

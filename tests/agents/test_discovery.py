"""Tests for onevalet.agents.discovery"""

import pytest

from onevalet.agents.decorator import AGENT_REGISTRY, AgentMetadata, valet
from onevalet.agents.discovery import AgentDiscovery


class TestAgentDiscovery:

    def setup_method(self):
        self._original_registry = dict(AGENT_REGISTRY)
        self.discovery = AgentDiscovery()

    def teardown_method(self):
        AGENT_REGISTRY.clear()
        AGENT_REGISTRY.update(self._original_registry)

    def test_initial_state_empty(self):
        assert self.discovery.get_discovered_agents() == {}
        assert self.discovery.get_agent_names() == []

    def test_get_discovered_returns_copy(self):
        result = self.discovery.get_discovered_agents()
        result["fake"] = "data"
        assert "fake" not in self.discovery.get_discovered_agents()

    def test_scan_nonexistent_module_returns_zero(self):
        count = self.discovery.scan_module("nonexistent.module.path.xyz")
        assert count == 0

    def test_scan_module_finds_decorated_agents(self):
        # The composio agents should be discoverable
        count = self.discovery.scan_module("onevalet.builtin_agents.composio.slack_agent")
        assert count >= 1
        names = self.discovery.get_agent_names()
        assert "SlackComposioAgent" in names

    def test_get_agent_found(self):
        self.discovery.scan_module("onevalet.builtin_agents.composio.slack_agent")
        agent = self.discovery.get_agent("SlackComposioAgent")
        assert agent is not None
        assert isinstance(agent, AgentMetadata)

    def test_get_agent_not_found(self):
        assert self.discovery.get_agent("NonExistentAgent") is None

    def test_scan_package(self):
        count = self.discovery.scan_package("onevalet.builtin_agents.composio")
        assert count >= 2  # at least Slack + GitHub

    def test_no_duplicates_on_rescan(self):
        count1 = self.discovery.scan_module("onevalet.builtin_agents.composio.slack_agent")
        count2 = self.discovery.scan_module("onevalet.builtin_agents.composio.slack_agent")
        assert count1 >= 1
        assert count2 == 0  # already discovered

    def test_sync_from_global_registry(self):
        @valet
        class TempTestAgent:
            """Temporary agent for testing"""

        assert "TempTestAgent" in AGENT_REGISTRY
        count = self.discovery.sync_from_global_registry()
        assert count > 0
        assert "TempTestAgent" in self.discovery.get_discovered_agents()

    def test_sync_skips_already_discovered(self):
        @valet
        class TempTestAgent2:
            """Temporary"""

        self.discovery.sync_from_global_registry()
        count2 = self.discovery.sync_from_global_registry()
        # Second sync should add 0 new agents (all already discovered)
        assert count2 == 0

    def test_clear(self):
        self.discovery.scan_module("onevalet.builtin_agents.composio.slack_agent")
        assert len(self.discovery.get_discovered_agents()) > 0
        self.discovery.clear()
        assert len(self.discovery.get_discovered_agents()) == 0

    def test_scan_paths(self):
        count = self.discovery.scan_paths([
            "onevalet.builtin_agents.composio",
        ])
        assert count >= 2

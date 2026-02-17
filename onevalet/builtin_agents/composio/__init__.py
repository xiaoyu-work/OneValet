"""
Composio integration for OneValet.

Provides per-app agents powered by the Composio OAuth proxy platform,
enabling access to 1000+ third-party app integrations with a single API key.

Agents:
- SlackComposioAgent: Send/fetch messages, list channels, find users, create reminders.
- GitHubComposioAgent: Create issues/PRs, list issues/PRs, search repositories.
"""

from .slack_agent import SlackComposioAgent
from .github_agent import GitHubComposioAgent

__all__ = [
    "SlackComposioAgent",
    "GitHubComposioAgent",
]

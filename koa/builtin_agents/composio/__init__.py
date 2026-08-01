"""
Composio integration for Koa.

Provides per-app agents powered by the Composio OAuth proxy platform,
enabling access to 1000+ third-party app integrations with a single API key.

Agents:
- SlackComposioAgent: Send/fetch messages, list channels, find users, create reminders.
- TwitterComposioAgent: Post tweets, view timeline, search tweets, look up users.
- SpotifyComposioAgent: Control playback, search music, manage playlists.
- YouTubeComposioAgent: Search videos, get video details, list playlists.
- LinkedInComposioAgent: Create posts, view profile.
- DiscordComposioAgent: Send messages, list channels, list servers.

GitHub moved off Composio to GitHub's official MCP server — see
koa/builtin_agents/github/.
"""

from .discord_agent import DiscordComposioAgent
from .linkedin_agent import LinkedInComposioAgent
from .slack_agent import SlackComposioAgent
from .spotify_agent import SpotifyComposioAgent
from .twitter_agent import TwitterComposioAgent
from .youtube_agent import YouTubeComposioAgent

__all__ = [
    "SlackComposioAgent",
    "TwitterComposioAgent",
    "SpotifyComposioAgent",
    "YouTubeComposioAgent",
    "LinkedInComposioAgent",
    "DiscordComposioAgent",
]

"""Core fixtures for integration tests.

Provides a real LLM client + real agent registry + mock dependencies,
allowing end-to-end testing of orchestrator routing and agent tool selection.

Configuration via environment variables:
    INTEGRATION_TEST_API_KEY   - Required. API key for the LLM provider.
    INTEGRATION_TEST_PROVIDER  - Provider name (default: openai).
    INTEGRATION_TEST_MODEL     - Model name (default: gpt-4o-mini).
    INTEGRATION_TEST_BASE_URL  - Custom base URL (optional, for Azure/Ollama).
"""

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from onevalet.llm.base import LLMConfig
from onevalet.llm.litellm_client import LiteLLMClient
from onevalet.config.registry import AgentRegistry
from onevalet.llm.registry import LLMRegistry
from onevalet.models import AgentTool
from onevalet.orchestrator.orchestrator import Orchestrator

pytestmark = [
    pytest.mark.integration,
]


# ---------------------------------------------------------------------------
# LLM config from env vars
# ---------------------------------------------------------------------------

def _get_llm_config() -> Tuple[LLMConfig, str]:
    """Build LLM config from INTEGRATION_TEST_* env vars."""
    api_key = os.environ.get("INTEGRATION_TEST_API_KEY")
    if not api_key:
        pytest.skip("INTEGRATION_TEST_API_KEY not set — see tests/integration/README.md")

    provider = os.environ.get("INTEGRATION_TEST_PROVIDER", "openai")
    model = os.environ.get("INTEGRATION_TEST_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("INTEGRATION_TEST_BASE_URL") or None

    config = LLMConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=0.0,  # minimize non-determinism
        max_tokens=1024,
    )
    return config, provider


# ---------------------------------------------------------------------------
# Canned tool results
# ---------------------------------------------------------------------------

CANNED_DATA: Dict[str, str] = {
    # Expense tools
    "log_expense": json.dumps({
        "id": "exp_001", "amount": 15.00, "category": "food",
        "description": "lunch", "date": "2026-02-28", "currency": "USD",
    }),
    "query_expenses": json.dumps([
        {"id": "exp_001", "amount": 15.00, "category": "food", "date": "2026-02-28", "merchant": "Chipotle"},
        {"id": "exp_002", "amount": 8.50, "category": "food", "date": "2026-02-27", "merchant": "Starbucks"},
    ]),
    "spending_summary": json.dumps([
        {"category": "food", "total_amount": 250.00, "count": 15},
        {"category": "transport", "total_amount": 80.00, "count": 8},
    ]),
    "set_budget": json.dumps({"category": "food", "monthly_limit": 500.00, "currency": "USD"}),
    "budget_status": json.dumps([
        {"category": "food", "monthly_limit": 500.00, "spent": 250.00, "remaining": 250.00},
    ]),
    "delete_expense": json.dumps({"deleted": True, "description": "Starbucks coffee $5.00"}),
    "search_receipts": json.dumps([
        {"id": "rcp_001", "file_name": "starbucks_receipt.jpg", "created_at": "2026-02-20"},
    ]),
    "upload_receipt": json.dumps({"success": True, "url": "https://drive.google.com/file/receipt_001"}),

    # Briefing tools
    "get_briefing": (
        "## Calendar\n- 09:00: Team standup\n- 12:00: Lunch with Bob\n\n"
        "## Tasks\n- Buy groceries\n- Call dentist\n\n"
        "## Unread Emails\n- john@example.com: Q4 Report"
    ),
    "setup_daily_briefing": "Daily briefing scheduled at 07:00.\nJob ID: job_001\nSchedule: cron: 0 7 * * *",
    "manage_briefing": "Daily Briefing: enabled\nSchedule: cron: 0 8 * * *\nNext run: in 12h",

    # Calendar tools
    "query_events": json.dumps([
        {"summary": "Team standup", "start": {"dateTime": "2026-02-28T09:00:00"}},
        {"summary": "Lunch with Bob", "start": {"dateTime": "2026-02-28T12:00:00"}},
    ]),
    "create_event": json.dumps({"id": "evt_001", "summary": "Meeting", "start": "2026-03-01T14:00:00"}),
    "update_event": json.dumps({"id": "evt_001", "summary": "Updated Meeting", "updated": True}),
    "delete_event": json.dumps({"deleted": True, "id": "evt_001"}),

    # Email tools
    "search_emails": json.dumps([
        {"sender": "john@example.com", "subject": "Q4 Report", "snippet": "Please review the attached..."},
        {"sender": "boss@company.com", "subject": "Meeting tomorrow", "snippet": "Let's sync at 3pm"},
    ]),
    "send_email": json.dumps({"id": "msg_001", "status": "sent"}),
    "reply_email": json.dumps({"id": "msg_002", "status": "sent"}),
    "delete_emails": json.dumps({"deleted": True, "count": 1}),
    "archive_emails": json.dumps({"archived": True, "count": 1}),
    "mark_as_read": json.dumps({"marked": True, "count": 1}),

    # Todo tools
    "create_task": json.dumps({"id": "task_001", "title": "Buy groceries", "status": "pending"}),
    "query_tasks": json.dumps([
        {"id": "task_001", "title": "Buy groceries", "status": "pending"},
        {"id": "task_002", "title": "Call dentist", "status": "pending"},
    ]),
    "update_task": json.dumps({"id": "task_001", "title": "Buy groceries", "status": "completed"}),
    "delete_task": json.dumps({"deleted": True, "id": "task_001"}),
    "set_reminder": json.dumps({"id": "rem_001", "title": "Call mom", "schedule_datetime": "2026-03-01T10:00:00"}),
    "manage_reminders": json.dumps([
        {"id": "rem_001", "title": "Call mom", "status": "active"},
    ]),

    # Maps tools
    "search_places": json.dumps([
        {"name": "Olive Garden", "address": "123 Main St", "rating": 4.2, "type": "restaurant"},
        {"name": "Bella Italia", "address": "456 Oak Ave", "rating": 4.5, "type": "restaurant"},
    ]),
    "get_directions": json.dumps({
        "distance": "12.3 km", "duration": "18 min",
        "steps": ["Head north on Main St", "Turn right on Oak Ave"],
    }),
    "check_air_quality": json.dumps({"aqi": 42, "category": "Good", "location": "San Francisco"}),

    # Cron tools
    "cron_status": json.dumps({"running": True, "total_jobs": 3, "active_jobs": 2}),
    "cron_list": json.dumps([
        {"id": "job_001", "name": "Daily Briefing", "schedule": "0 8 * * *", "enabled": True},
    ]),
    "cron_add": json.dumps({"id": "job_002", "name": "Weekly Report", "schedule": "0 9 * * 1"}),
    "cron_update": json.dumps({"id": "job_001", "updated": True}),
    "cron_remove": json.dumps({"removed": True, "id": "job_001"}),
    "cron_run": json.dumps({"triggered": True, "id": "job_001"}),
    "cron_runs": json.dumps([{"id": "run_001", "status": "completed", "timestamp": "2026-02-28T08:00:00"}]),

    # Shipping tools
    "track_shipment": json.dumps({
        "tracking_number": "1Z999AA10123456784", "status": "In Transit",
        "estimated_delivery": "2026-03-02", "carrier": "UPS",
    }),

    # SmartHome tools
    "control_lights": json.dumps({"success": True, "action": "off", "device": "living room lights"}),
    "control_speaker": json.dumps({"success": True, "action": "play", "device": "living room speaker"}),

    # Cloud Storage tools
    "search_files": json.dumps([
        {"name": "Q4_Report.pdf", "id": "file_001", "size": "2.3 MB", "modified": "2026-02-15"},
    ]),
    "list_recent_files": json.dumps([
        {"name": "Notes.docx", "id": "file_002", "modified": "2026-02-27"},
    ]),
    "get_file_info": json.dumps({"name": "Q4_Report.pdf", "size": "2.3 MB", "shared": False}),
    "get_download_link": json.dumps({"url": "https://drive.google.com/download/file_001"}),
    "share_file": json.dumps({"shared": True, "link": "https://drive.google.com/share/file_001"}),
    "storage_usage": json.dumps({"used": "12.5 GB", "total": "15 GB", "percent": 83}),

    # Notion tools
    "notion_search": json.dumps([
        {"id": "page_001", "title": "Meeting Notes", "type": "page"},
    ]),
    "notion_read_page": json.dumps({"id": "page_001", "title": "Meeting Notes", "content": "Discussed Q4 goals..."}),
    "notion_query_database": json.dumps([
        {"id": "row_001", "properties": {"Name": "Task A", "Status": "In Progress"}},
    ]),
    "notion_create_page": json.dumps({"id": "page_002", "title": "New Page", "url": "https://notion.so/page_002"}),
    "notion_update_page": json.dumps({"id": "page_001", "updated": True}),

    # Google Workspace tools
    "google_drive_search": json.dumps([
        {"id": "gdoc_001", "name": "Q4 Report", "mimeType": "application/vnd.google-apps.document"},
    ]),
    "google_docs_read": json.dumps({"id": "gdoc_001", "title": "Q4 Report", "content": "Revenue grew by 15%..."}),
    "google_sheets_read": json.dumps({"id": "gsheet_001", "title": "Budget", "rows": [["Item", "Cost"], ["Rent", "2000"]]}),
    "google_docs_create": json.dumps({"id": "gdoc_002", "title": "New Doc", "url": "https://docs.google.com/gdoc_002"}),
    "google_sheets_write": json.dumps({"id": "gsheet_001", "updated": True, "range": "A1:B2"}),

    # Trip Planner tools
    "check_weather": json.dumps({"location": "Tokyo", "temp": "15°C", "condition": "Partly cloudy"}),
    "search_flights": json.dumps([
        {"airline": "ANA", "departure": "10:00", "arrival": "14:00", "price": "$850"},
    ]),
    "search_hotels": json.dumps([
        {"name": "Hotel Sunroute", "price": "$120/night", "rating": 4.3},
    ]),

    # Image tools — ImageAgent uses InputField not tools, so minimal canned data
    "generate_image": json.dumps({"url": "https://images.example.com/sunset_001.png", "prompt": "a sunset"}),

    # Composio agent tools
    "create_issue": json.dumps({"number": 42, "title": "Login bug", "url": "https://github.com/org/repo/issues/42"}),
    "list_issues": json.dumps([
        {"number": 42, "title": "Login bug", "state": "open"},
        {"number": 41, "title": "Fix typo", "state": "closed"},
    ]),
    "create_pull_request": json.dumps({"number": 10, "title": "Fix login", "url": "https://github.com/org/repo/pull/10"}),
    "list_pull_requests": json.dumps([
        {"number": 10, "title": "Fix login", "state": "open"},
    ]),
    "search_repositories": json.dumps([
        {"full_name": "org/repo", "description": "Main app", "stars": 100},
    ]),
    "connect_github": "GitHub is already connected.",

    "post_tweet": json.dumps({"id": "tweet_001", "text": "New product launch!", "url": "https://x.com/user/tweet_001"}),
    "get_timeline": json.dumps([
        {"id": "tweet_002", "text": "Hello world", "author": "@user"},
    ]),
    "search_tweets": json.dumps([
        {"id": "tweet_003", "text": "Great product!", "author": "@fan"},
    ]),
    "lookup_user": json.dumps({"username": "elonmusk", "followers": 100000000}),
    "connect_twitter": "Twitter is already connected.",

    "send_message": json.dumps({"ok": True, "channel": "#engineering", "ts": "1234567890.123456"}),
    "fetch_messages": json.dumps([
        {"text": "standup at 10am", "user": "U123", "ts": "1234567890.000000"},
    ]),
    "list_channels": json.dumps([
        {"id": "C001", "name": "engineering"},
        {"id": "C002", "name": "general"},
    ]),
    "find_users": json.dumps([
        {"id": "U123", "name": "John Doe", "email": "john@example.com"},
    ]),
    "create_reminder": json.dumps({"ok": True, "reminder": {"text": "Check PR"}}),
    "connect_slack": "Slack is already connected.",

    "play_music": json.dumps({"playing": True, "track": "Take Five", "artist": "Dave Brubeck"}),
    "pause_music": json.dumps({"paused": True}),
    "search_music": json.dumps([
        {"name": "Take Five", "artist": "Dave Brubeck", "type": "track"},
    ]),
    "get_playlists": json.dumps([
        {"id": "pl_001", "name": "Jazz Classics", "tracks": 50},
    ]),
    "add_to_playlist": json.dumps({"added": True, "playlist": "Jazz Classics"}),
    "now_playing": json.dumps({"track": "Take Five", "artist": "Dave Brubeck", "progress": "2:30"}),
    "connect_spotify": "Spotify is already connected.",

    "list_servers": json.dumps([
        {"id": "srv_001", "name": "Dev Server"},
    ]),
    "connect_discord": "Discord is already connected.",

    "create_post": json.dumps({"id": "post_001", "text": "Exciting update!", "visibility": "PUBLIC"}),
    "get_my_profile": json.dumps({"name": "John Doe", "headline": "Software Engineer"}),
    "connect_linkedin": "LinkedIn is already connected.",

    "search_videos": json.dumps([
        {"title": "Sunset Timelapse", "url": "https://youtube.com/watch?v=abc", "channel": "Nature HD"},
    ]),
    "get_video_details": json.dumps({
        "title": "Sunset Timelapse", "views": 1000000, "likes": 50000, "duration": "3:45",
    }),
    "list_playlists": json.dumps([
        {"id": "pl_001", "title": "Favorites", "videoCount": 25},
    ]),
    "connect_youtube": "YouTube is already connected.",
}


# ---------------------------------------------------------------------------
# ToolCallRecorder
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecorder:
    """Records all agent routing and tool calls during a test."""

    agent_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    canned_results: Dict[str, str] = field(default_factory=dict)

    def wrap_tool(self, tool: AgentTool) -> AgentTool:
        """Create a copy of the tool with executor replaced by a recording wrapper."""
        recorder = self

        async def recording_executor(args: Dict[str, Any], context: Any) -> str:
            # Skip approval for tests
            result = recorder.canned_results.get(tool.name, CANNED_DATA.get(tool.name, "{}"))
            recorder.tool_calls.append({
                "tool_name": tool.name,
                "arguments": copy.deepcopy(args),
                "result": result,
            })
            return result

        return AgentTool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            executor=recording_executor,
            needs_approval=False,  # disable approval in tests
            risk_level="read",     # override to avoid approval pauses
            category=tool.category,
        )

    def reset(self) -> None:
        self.agent_calls.clear()
        self.tool_calls.clear()


# ---------------------------------------------------------------------------
# TestOrchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator(Orchestrator):
    """Orchestrator subclass that intercepts agent creation for test recording."""

    def __init__(self, recorder: ToolCallRecorder, **kwargs):
        super().__init__(**kwargs)
        self.recorder = recorder

    async def create_agent(self, tenant_id, agent_type, context_hints=None, context=None):
        # Record the agent routing decision
        self.recorder.agent_calls.append({
            "agent_type": agent_type,
            "task_instruction": (context_hints or {}).get("task_instruction", ""),
        })

        agent = await super().create_agent(
            tenant_id=tenant_id,
            agent_type=agent_type,
            context_hints=context_hints,
            context=context,
        )

        if agent is not None and hasattr(agent, "tools") and agent.tools:
            # Replace each tool's executor with recording wrapper
            patched_tools = tuple(self.recorder.wrap_tool(t) for t in agent.tools)
            agent.tools = patched_tools

        return agent

    async def post_process(self, result, context):
        """Skip momex save and other post-processing in tests."""
        return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def llm_client():
    """Session-scoped real LLM client configured via env vars."""
    config, provider = _get_llm_config()
    client = LiteLLMClient(config=config, provider_name=provider)
    return client


@pytest.fixture(scope="session")
def agent_registry():
    """Session-scoped agent registry with all builtin agents loaded."""
    from onevalet.agents.discovery import AgentDiscovery

    discovery = AgentDiscovery()
    discovery.scan_package("onevalet.builtin_agents")
    discovery.sync_from_global_registry()

    llm_registry = LLMRegistry.get_instance()
    registry = AgentRegistry(llm_registry=llm_registry)
    return registry


@pytest.fixture
async def orchestrator_factory(llm_client, agent_registry):
    """Factory that creates a TestOrchestrator with recording.

    Usage:
        orch, recorder = await orchestrator_factory()
        # or with custom canned results:
        orch, recorder = await orchestrator_factory(canned_results={"log_expense": "..."})
    """
    # Register the LLM client as "default" so agents can use it
    llm_registry = agent_registry.llm_registry
    llm_registry.register("default", llm_client)

    async def _create(canned_results: Optional[Dict[str, str]] = None):
        recorder = ToolCallRecorder(
            canned_results=canned_results or {},
        )

        # Mock momex — returns empty history, no-ops on save
        mock_momex = MagicMock()
        mock_momex.get_history.return_value = []
        mock_momex.save_message = MagicMock()
        mock_momex.save_conversation = AsyncMock()

        orch = TestOrchestrator(
            recorder=recorder,
            momex=mock_momex,
            llm_client=llm_client,
            agent_registry=agent_registry,
            credential_store=None,
            database=None,
        )
        await orch.initialize()

        return orch, recorder

    return _create


@pytest.fixture
def llm_judge(llm_client):
    """LLM-as-judge helper for evaluating response quality."""

    async def _judge(
        user_input: str,
        response: str,
        criteria: str,
    ) -> bool:
        prompt = (
            "You are evaluating an AI assistant's response.\n\n"
            f'User input: "{user_input}"\n\n'
            f'AI response: "{response}"\n\n'
            f"Criteria: {criteria}\n\n"
            'Reply with ONLY "PASS" or "FAIL" followed by a one-line reason.'
        )
        result = await llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            config={"temperature": 0.0},
        )
        return result.content.strip().upper().startswith("PASS")

    return _judge

"""CLI account connection via OAuth.

Opens the browser for OAuth flows and lists connected accounts.

Usage:
    koa connect              # list connected accounts
    koa connect gmail        # connect Gmail via Google OAuth
    koa connect outlook      # connect Outlook via Microsoft OAuth
    koa connect notion       # connect Notion
"""

import sys
import webbrowser
from typing import Optional

# Maps CLI service names to OAuth authorize endpoints and display names.
# Google OAuth covers: gmail, google_calendar, google_tasks
# Microsoft OAuth covers: outlook, outlook_calendar, microsoft_todo, onedrive
_OAUTH_SERVICES = {
    # Google (single OAuth flow covers all Google services)
    "gmail": {
        "endpoint": "/api/oauth/google/authorize",
        "label": "Google (Gmail, Calendar, Tasks)",
    },
    "google-calendar": {
        "endpoint": "/api/oauth/google/authorize",
        "label": "Google (Gmail, Calendar, Tasks)",
    },
    "google-tasks": {
        "endpoint": "/api/oauth/google/authorize",
        "label": "Google (Gmail, Calendar, Tasks)",
    },
    "google": {
        "endpoint": "/api/oauth/google/authorize",
        "label": "Google (Gmail, Calendar, Tasks)",
    },
    # Microsoft (single OAuth flow covers all Microsoft services)
    "outlook": {
        "endpoint": "/api/oauth/microsoft/authorize",
        "label": "Microsoft (Outlook, Calendar, To Do, OneDrive)",
    },
    "outlook-calendar": {
        "endpoint": "/api/oauth/microsoft/authorize",
        "label": "Microsoft (Outlook, Calendar, To Do, OneDrive)",
    },
    "microsoft-todo": {
        "endpoint": "/api/oauth/microsoft/authorize",
        "label": "Microsoft (Outlook, Calendar, To Do, OneDrive)",
    },
    "onedrive": {
        "endpoint": "/api/oauth/microsoft/authorize",
        "label": "Microsoft (Outlook, Calendar, To Do, OneDrive)",
    },
    "microsoft": {
        "endpoint": "/api/oauth/microsoft/authorize",
        "label": "Microsoft (Outlook, Calendar, To Do, OneDrive)",
    },
    # Individual services
    "todoist": {"endpoint": "/api/oauth/todoist/authorize", "label": "Todoist"},
    "notion": {"endpoint": "/api/oauth/notion/authorize", "label": "Notion"},
    "dropbox": {"endpoint": "/api/oauth/dropbox/authorize", "label": "Dropbox"},
    "hue": {"endpoint": "/api/oauth/hue/authorize", "label": "Philips Hue"},
    "philips-hue": {"endpoint": "/api/oauth/hue/authorize", "label": "Philips Hue"},
    "sonos": {"endpoint": "/api/oauth/sonos/authorize", "label": "Sonos"},
    # Composio-powered services
    "slack": {"endpoint": "/api/oauth/slack/authorize", "label": "Slack (via Composio)"},
    "github": {"endpoint": "/api/oauth/github/authorize", "label": "GitHub (via Composio)"},
    "spotify": {"endpoint": "/api/oauth/spotify/authorize", "label": "Spotify (via Composio)"},
    "discord": {"endpoint": "/api/oauth/discord/authorize", "label": "Discord (via Composio)"},
    "twitter": {"endpoint": "/api/oauth/twitter/authorize", "label": "Twitter/X (via Composio)"},
    "linkedin": {"endpoint": "/api/oauth/linkedin/authorize", "label": "LinkedIn (via Composio)"},
    "youtube": {"endpoint": "/api/oauth/youtube/authorize", "label": "YouTube (via Composio)"},
}

# Primary services shown in `koa connect` help (no duplicates)
_PRIMARY_SERVICES = [
    ("google", "Gmail, Calendar, Tasks"),
    ("microsoft", "Outlook, Calendar, To Do, OneDrive"),
    ("todoist", "Todoist task management"),
    ("notion", "Notion workspace"),
    ("dropbox", "Dropbox file storage"),
    ("hue", "Philips Hue smart lights"),
    ("sonos", "Sonos speakers"),
    ("slack", "Slack (via Composio)"),
    ("github", "GitHub (via Composio)"),
    ("spotify", "Spotify (via Composio)"),
]


def connect_account(
    url: str,
    service: str,
    tenant_id: str = "default",
    api_key: Optional[str] = None,
) -> None:
    """Open browser to start OAuth flow for a service."""
    service = service.lower().replace("_", "-")

    if service not in _OAUTH_SERVICES:
        print(f"\033[31m✗ Unknown service: {service}\033[0m")
        print("\nAvailable services:")
        for name, desc in _PRIMARY_SERVICES:
            print(f"  \033[1m{name:<14}\033[0m {desc}")
        sys.exit(1)

    info = _OAUTH_SERVICES[service]
    oauth_url = f"{url.rstrip('/')}{info['endpoint']}?tenant_id={tenant_id}"

    print(f"\033[1mConnecting {info['label']}...\033[0m")
    print("\033[90mOpening browser for OAuth authorization.\033[0m")
    print(f"\033[90mURL: {oauth_url}\033[0m\n")

    try:
        webbrowser.open(oauth_url)
        print("✓ Browser opened. Complete the authorization in your browser.")
        print("\033[90mThe account will appear in `koa connect` once authorized.\033[0m")
    except Exception:
        print(f"Could not open browser. Visit this URL manually:\n{oauth_url}")


def list_accounts(
    url: str,
    tenant_id: str = "default",
    api_key: Optional[str] = None,
) -> None:
    """List connected accounts for a tenant."""
    try:
        import httpx
    except ImportError:
        print("Error: httpx is required.")
        sys.exit(1)

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(
                f"{url.rstrip('/')}/api/credentials",
                params={"tenant_id": tenant_id},
                headers=headers,
            )

            if resp.status_code == 503:
                print("\033[31m✗ Server not configured. Run `koa serve` first.\033[0m")
                sys.exit(1)
            if resp.status_code == 401:
                print("\033[31m✗ Authentication failed. Check KOA_API_KEY.\033[0m")
                sys.exit(1)

            accounts = resp.json()
    except Exception as e:
        print(f"\033[31m✗ Cannot connect to server: {e}\033[0m")
        print("\033[90mMake sure the server is running: koa serve\033[0m")
        sys.exit(1)

    if not accounts:
        print("No accounts connected.\n")
        print("Connect an account:")
        for name, desc in _PRIMARY_SERVICES[:4]:
            print(f"  koa connect {name:<14} # {desc}")
        return

    print("\033[1mConnected accounts:\033[0m\n")
    for acct in accounts:
        service = acct.get("service", "?")
        email = acct.get("email", "")
        name = acct.get("account_name", "")
        label = f"{email}" if email else name
        print(f"  \033[32m✓\033[0m {service:<20} {label}")

    print("\n\033[90mConnect more: koa connect <service>\033[0m")

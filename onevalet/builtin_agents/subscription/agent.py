"""
SubscriptionAgent — Agent for querying and managing user subscriptions.

Subscriptions are auto-detected from email receipts and stored in the
subscriptions table. This agent provides a conversational interface
to query them.
"""

from onevalet import valet
from onevalet.standard_agent import StandardAgent

from .tools import query_subscriptions, check_expiring_subscriptions


@valet(domain="lifestyle")
class SubscriptionAgent(StandardAgent):
    """Query and manage subscriptions (Netflix, Spotify, iCloud, T-Mobile, etc.). Use when the user asks about their subscriptions, recurring charges, monthly bills, or wants to know what services they are paying for."""

    max_turns = 3

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a subscription management assistant.

Available tools:
- query_subscriptions: List user's subscriptions, optionally filtered by status or category.

Today's date: {today} ({weekday})

Instructions:
1. Use query_subscriptions to look up the user's subscriptions.
2. You can filter by status ("active", "cancelled", "trial", "all") or category
   ("streaming", "cloud", "productivity", "saas", "developer", "telecom", "vpn",
    "fitness", "news", "gaming", "education", "finance", "home", "shopping").
3. Present results clearly — include monthly cost estimates when relevant.
4. If the user asks about total spending, calculate the monthly total from active subscriptions.
5. Respond in the same language the user used."""

    def get_system_prompt(self) -> str:
        now, _ = self._user_now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime('%Y-%m-%d'),
            weekday=now.strftime('%A'),
        )

    tools = (
        query_subscriptions,
        check_expiring_subscriptions,
    )

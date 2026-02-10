"""
Calendar Account Resolver — resolves account specs to calendar credentials.

Reuses AccountResolver logic, searching google_calendar and outlook_calendar services.
"""

import logging
from typing import List, Optional

from onevalet.providers.email.resolver import AccountResolver

logger = logging.getLogger(__name__)

_CALENDAR_SERVICES = ("google_calendar", "outlook_calendar")


class CalendarAccountResolver:
    """
    Resolve calendar account references to credential dicts.

    Supports class-level calls (used by all calendar agents):
        account = CalendarAccountResolver.resolve_account(tenant_id, "primary")
    """

    @staticmethod
    async def resolve_account(
        tenant_id: str,
        account_spec: Optional[str] = None,
    ) -> Optional[dict]:
        """Resolve a single calendar account across all calendar services."""
        resolver = AccountResolver()
        return await resolver._resolve_account_all_services(
            tenant_id, account_spec, _CALENDAR_SERVICES
        )

    @staticmethod
    async def resolve_accounts(
        tenant_id: str,
        account_specs: Optional[List[str]] = None,
    ) -> List[dict]:
        """Resolve multiple calendar accounts across all calendar services."""
        resolver = AccountResolver()
        all_accounts = []
        seen_emails: set = set()

        for service in _CALENDAR_SERVICES:
            accounts = await resolver._resolve_for_service(tenant_id, service, account_specs)
            for acc in accounts:
                email = acc.get("email", "")
                if email not in seen_emails:
                    all_accounts.append(acc)
                    seen_emails.add(email)

        return all_accounts

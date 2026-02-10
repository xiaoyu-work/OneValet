"""
Todo Account Resolver - resolves account specs to todo credentials.

Reuses AccountResolver logic, searching todoist, google_tasks, and microsoft_todo services.
"""

import logging
from typing import List, Optional

from onevalet.providers.email.resolver import AccountResolver

logger = logging.getLogger(__name__)

_TODO_SERVICES = ("todoist", "google_tasks", "microsoft_todo")


class TodoAccountResolver:
    """
    Resolve todo account references to credential dicts.

    Supports class-level calls (used by all todo agents):
        account = TodoAccountResolver.resolve_account(tenant_id, "primary")
        accounts = TodoAccountResolver.resolve_accounts(tenant_id, ["all"])
    """

    @staticmethod
    async def resolve_account(
        tenant_id: str,
        account_spec: Optional[str] = None,
    ) -> Optional[dict]:
        """Resolve a single todo account across all todo services."""
        resolver = AccountResolver()
        return await resolver._resolve_account_all_services(
            tenant_id, account_spec, _TODO_SERVICES
        )

    @staticmethod
    async def resolve_accounts(
        tenant_id: str,
        account_specs: Optional[List[str]] = None,
    ) -> List[dict]:
        """Resolve multiple todo accounts across all todo services."""
        resolver = AccountResolver()
        all_accounts = []
        seen_emails: set = set()

        for service in _TODO_SERVICES:
            accounts = await resolver._resolve_for_service(tenant_id, service, account_specs)
            for acc in accounts:
                email = acc.get("email", "")
                if email not in seen_emails:
                    all_accounts.append(acc)
                    seen_emails.add(email)

        return all_accounts

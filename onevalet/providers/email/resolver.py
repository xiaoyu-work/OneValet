"""
Account Resolver - Resolve account names/aliases to credentials

Uses CredentialStore instead of direct database queries.

Handles:
- "primary" -> primary account
- "work" -> account with account_name="work"
- "john@gmail.com" -> account matching that email
- "all" -> all active accounts
- None -> default to primary account
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AccountResolver:
    """
    Resolve user-friendly account references to credential dicts
    using CredentialStore.
    """

    def __init__(self, credential_store):
        """
        Args:
            credential_store: CredentialStore instance (from onevalet.credentials)
        """
        self.credential_store = credential_store

    async def resolve_account(
        self,
        tenant_id: str,
        service: str,
        account_spec: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Resolve a single account specification to a credentials dict.

        Args:
            tenant_id: Tenant/user ID
            service: Service name (e.g., "google", "outlook")
            account_spec: Account specification (can be):
                - None or "primary": Primary account
                - Account name: "work", "personal", etc.
                - Email address: "john@gmail.com"

        Returns:
            Credentials dict or None if not found
        """
        # Default to primary account
        if not account_spec or account_spec.lower() == "primary":
            creds = await self.credential_store.get(tenant_id, service, "primary")
            if not creds:
                logger.warning(f"No primary account found for tenant {tenant_id}, service {service}")
            return creds

        # Try by account_name directly
        creds = await self.credential_store.get(tenant_id, service, account_spec)
        if creds:
            logger.info(f"Resolved '{account_spec}' to account_name match")
            return creds

        # Try by email in list of all accounts for this service
        all_accounts = await self.credential_store.list(tenant_id, service)
        for acc in all_accounts:
            acc_creds = acc.get("credentials", {})
            if acc_creds.get("email", "").lower() == account_spec.lower():
                logger.info(f"Resolved '{account_spec}' by email match")
                return acc_creds

        logger.warning(f"No account found matching spec: '{account_spec}'")
        return None

    async def resolve_accounts(
        self,
        tenant_id: str,
        service: str,
        account_specs: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Resolve multiple account specifications to credential dicts.

        Args:
            tenant_id: Tenant/user ID
            service: Service name
            account_specs: List of account specs, or special values:
                - None: Default to primary account only
                - ["all"]: All active accounts for this service
                - ["work", "personal"]: Multiple specific accounts

        Returns:
            List of credentials dicts (deduplicated)
        """
        # Default: primary account only
        if not account_specs:
            primary = await self.credential_store.get(tenant_id, service, "primary")
            if primary:
                logger.info(f"Using primary account for service {service}")
                return [primary]
            else:
                logger.warning(f"No primary account found for tenant {tenant_id}, service {service}")
                return []

        # Special case: "all" accounts
        if len(account_specs) == 1 and account_specs[0].lower() == "all":
            all_accounts = await self.credential_store.list(tenant_id, service)
            results = [acc["credentials"] for acc in all_accounts if "credentials" in acc]
            logger.info(f"Resolved 'all' to {len(results)} accounts")
            return results

        # Resolve each spec individually
        accounts = []
        seen_emails = set()

        for spec in account_specs:
            creds = await self.resolve_account(tenant_id, service, spec)
            if creds:
                email = creds.get("email", "")
                if email not in seen_emails:
                    accounts.append(creds)
                    seen_emails.add(email)
                else:
                    logger.debug(f"Skipping duplicate account: {email}")

        if not accounts:
            logger.warning(f"No accounts resolved from specs: {account_specs}")

        return accounts

    @staticmethod
    def get_account_display_name(credentials: dict) -> str:
        """
        Get human-readable display name for an account.

        Args:
            credentials: Credentials dict

        Returns:
            Display string like "Work (john@company.com)" or "john@gmail.com"
        """
        account_name = credentials.get("account_name", "Unknown")
        email = credentials.get("email", "unknown@example.com")

        if account_name.lower() == email.lower():
            return email

        return f"{account_name} ({email})"

    async def list_user_accounts(
        self,
        tenant_id: str,
        service: Optional[str] = None,
    ) -> List[str]:
        """
        Get list of account display names for a tenant.

        Args:
            tenant_id: Tenant/user ID
            service: Optional service filter

        Returns:
            List of display strings
        """
        all_accounts = await self.credential_store.list(tenant_id, service)

        display_names = []
        for acc in all_accounts:
            account_name = acc.get("account_name", "unknown")
            creds = acc.get("credentials", {})
            email = creds.get("email", "unknown")

            if account_name == "primary":
                display_names.append(f"Primary: {email}")
            else:
                display_names.append(f"{account_name} ({email})")

        return display_names

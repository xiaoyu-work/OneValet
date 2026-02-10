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


_EMAIL_SERVICES = ("gmail", "outlook")


class AccountResolver:
    """
    Resolve user-friendly account references to credential dicts
    using CredentialStore.
    """

    _default_store = None

    @classmethod
    def set_default_store(cls, credential_store):
        """Set the default credential store for class-level access."""
        cls._default_store = credential_store

    def __init__(self, credential_store=None):
        """
        Args:
            credential_store: CredentialStore instance (from onevalet.credentials)
        """
        self.credential_store = credential_store or self._default_store

    async def resolve_account(
        self,
        tenant_id: str,
        account_spec_or_service: Optional[str] = None,
        account_spec: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Resolve a single account specification to a credentials dict.

        Supports two calling conventions:
            # Instance call with explicit service:
            resolver.resolve_account(tenant_id, "gmail", "primary")

            # Class-level call (searches all email services):
            AccountResolver.resolve_account(tenant_id, "primary")
        """
        # Detect class-level call: self is a string (tenant_id)
        if isinstance(self, str):
            actual_tenant = self
            actual_spec = tenant_id  # second arg is account_spec
            resolver = AccountResolver()
            return await resolver._resolve_account_all_services(
                actual_tenant, actual_spec, _EMAIL_SERVICES
            )

        # Instance call
        if account_spec is not None:
            # resolve_account(tenant_id, service, spec)
            return await self._resolve_account_for_service(
                tenant_id, account_spec_or_service, account_spec
            )
        else:
            # resolve_account(tenant_id, spec) — search all services
            return await self._resolve_account_all_services(
                tenant_id, account_spec_or_service, _EMAIL_SERVICES
            )

    async def _resolve_account_all_services(
        self, tenant_id: str, account_spec: Optional[str], services: tuple
    ) -> Optional[dict]:
        """Search across multiple services for a matching account."""
        for service in services:
            result = await self._resolve_account_for_service(tenant_id, service, account_spec)
            if result:
                return result
        return None

    @staticmethod
    def _enrich(creds: dict, account_name: str, service: str) -> dict:
        """Inject account_name and account_identifier that agents expect."""
        if creds is None:
            return None
        creds.setdefault("account_name", account_name)
        creds.setdefault("account_identifier", creds.get("email", ""))
        creds.setdefault("service", service)
        return creds

    async def _resolve_account_for_service(
        self, tenant_id: str, service: str, account_spec: Optional[str] = None,
    ) -> Optional[dict]:
        """Resolve a single account for a specific service."""
        if not self.credential_store:
            logger.error("No credential store available")
            return None

        # Default to primary account
        if not account_spec or account_spec.lower() == "primary":
            creds = await self.credential_store.get(tenant_id, service, "primary")
            return self._enrich(creds, "primary", service)

        # Try by account_name directly
        creds = await self.credential_store.get(tenant_id, service, account_spec)
        if creds:
            return self._enrich(creds, account_spec, service)

        # Try by email in list of all accounts for this service
        all_accounts = await self.credential_store.list(tenant_id, service)
        for acc in all_accounts:
            acc_creds = acc.get("credentials", {})
            if acc_creds.get("email", "").lower() == account_spec.lower():
                return self._enrich(acc_creds, acc.get("account_name", "primary"), service)

        return None

    async def resolve_accounts(
        self,
        tenant_id: str,
        service_or_specs=None,
        account_specs: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Resolve account specifications to credential dicts.

        Supports two calling conventions:
            # Instance call with explicit service:
            resolver.resolve_accounts(tenant_id, "gmail", ["all"])

            # Class-level call (searches all email services):
            AccountResolver.resolve_accounts(tenant_id, ["all"])
        """
        # Detect class-level call: AccountResolver.resolve_accounts(tenant_id, specs)
        # In that case, `self` is actually tenant_id (a string), and
        # `tenant_id` is account_specs.
        if isinstance(self, str):
            # Called as AccountResolver.resolve_accounts(tenant_id, account_specs)
            actual_tenant = self
            actual_specs = tenant_id if isinstance(tenant_id, list) else None
            resolver = AccountResolver()
            return await resolver._resolve_all_email_accounts(actual_tenant, actual_specs)

        # Instance call with explicit service
        if isinstance(service_or_specs, list):
            # resolve_accounts(tenant_id, ["all"]) — no explicit service
            return await self._resolve_all_email_accounts(tenant_id, service_or_specs)
        elif isinstance(service_or_specs, str):
            # resolve_accounts(tenant_id, "gmail", ["all"]) — explicit service
            return await self._resolve_for_service(tenant_id, service_or_specs, account_specs)
        else:
            return await self._resolve_all_email_accounts(tenant_id, None)

    async def _resolve_all_email_accounts(
        self,
        tenant_id: str,
        account_specs: Optional[List[str]] = None,
    ) -> List[dict]:
        """Resolve accounts across all email services (gmail, outlook)."""
        all_accounts = []
        seen_emails: set = set()

        for service in _EMAIL_SERVICES:
            accounts = await self._resolve_for_service(tenant_id, service, account_specs)
            for acc in accounts:
                email = acc.get("email", "")
                if email not in seen_emails:
                    all_accounts.append(acc)
                    seen_emails.add(email)

        return all_accounts

    async def _resolve_for_service(
        self,
        tenant_id: str,
        service: str,
        account_specs: Optional[List[str]] = None,
    ) -> List[dict]:
        """Resolve accounts for a single service."""
        if not self.credential_store:
            logger.error("No credential store available")
            return []

        # Default: primary account only
        if not account_specs:
            primary = await self.credential_store.get(tenant_id, service, "primary")
            if primary:
                logger.info(f"Using primary account for service {service}")
                return [self._enrich(primary, "primary", service)]
            else:
                return []

        # Special case: "all" accounts
        if len(account_specs) == 1 and account_specs[0].lower() == "all":
            all_accounts = await self.credential_store.list(tenant_id, service)
            results = []
            for acc in all_accounts:
                if "credentials" in acc:
                    results.append(self._enrich(acc["credentials"], acc.get("account_name", "primary"), service))
            logger.info(f"Resolved 'all' for {service}: {len(results)} accounts")
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

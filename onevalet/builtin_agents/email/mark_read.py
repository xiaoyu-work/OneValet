"""
Mark Read Email Agent

Agent for marking emails as read. Can use cached results from ReadEmailAgent
or search for specific emails.
"""
import logging
import json
import re
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class MarkReadEmailAgent(StandardAgent):
    """Mark emails as read agent"""

    target = InputField(
        prompt="Which emails would you like to mark as read?",
        description="Which emails to mark as read: 'all' for cached emails, numbers like '1,2,3' for specific emails, or search query",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

        self.emails_to_mark = []
        self.message_ids = []
        self.accounts = []

    def needs_approval(self) -> bool:
        return False

    def get_approval_prompt(self) -> str:
        """Generate confirmation prompt"""
        if not self.emails_to_mark:
            return "No emails to mark as read."

        count = len(self.emails_to_mark)
        response_parts = [f"Mark {count} email(s) as read:\n"]

        for i, email in enumerate(self.emails_to_mark[:5], 1):
            subject = email.get("subject", "No subject")
            if len(subject) > 40:
                subject = subject[:37] + "..."
            response_parts.append(f"{i}. {subject}")

        if count > 5:
            response_parts.append(f"\n... and {count - 5} more")

        response_parts.append("\n\nMark as read?")

        return "\n".join(response_parts)

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract target from user input"""
        user_lower = user_input.lower().strip()

        if any(kw in user_lower for kw in ["all", "them", "these", "those"]):
            return {"target": "all"}

        numbers = re.findall(r'\d+', user_input)
        if numbers:
            return {"target": ",".join(numbers)}

        return {"target": user_input}

    async def _load_emails_to_mark(self) -> bool:
        """Load emails to mark as read from cache or search"""
        target = self.collected_fields.get("target", "all")

        cached_data = self.context_hints.get("email_cache", {})
        cached_emails = cached_data.get("emails", [])
        cached_accounts = cached_data.get("accounts", [])

        logger.info(f"MarkReadAgent: target={target}, cached_emails={len(cached_emails)}")

        if target == "all" and cached_emails:
            self.emails_to_mark = cached_emails
            self.message_ids = [e.get("message_id") for e in cached_emails if e.get("message_id")]
            self.accounts = cached_accounts
            return True

        elif target and "," in target or target.isdigit():
            if not cached_emails:
                return False

            indices = [int(n.strip()) - 1 for n in target.split(",") if n.strip().isdigit()]
            self.emails_to_mark = [cached_emails[i] for i in indices if 0 <= i < len(cached_emails)]
            self.message_ids = [e.get("message_id") for e in self.emails_to_mark if e.get("message_id")]
            self.accounts = cached_accounts
            return len(self.emails_to_mark) > 0

        else:
            return await self._search_emails(target)

    async def _search_emails(self, search_query: str) -> bool:
        """Search for emails to mark as read"""
        from onevalet.providers.email.resolver import AccountResolver
        from onevalet.providers.email.factory import EmailProviderFactory

        try:
            accounts = await AccountResolver.resolve_accounts(self.tenant_id, ["primary"])
            if not accounts:
                return False

            account = accounts[0]
            provider = EmailProviderFactory.create_provider(account)
            if not provider:
                return False

            if not await provider.ensure_valid_token():
                return False

            result = await provider.search_emails(
                query=search_query,
                unread_only=True,
                days_back=7,
                max_results=50
            )

            if result.get("success"):
                self.emails_to_mark = result.get("data", [])
                self.message_ids = [e.get("message_id") for e in self.emails_to_mark if e.get("message_id")]
                self.accounts = [{"account_name": account["account_name"], "account_identifier": account["account_identifier"], "provider": account["provider"]}]
                return len(self.emails_to_mark) > 0

            return False

        except Exception as e:
            logger.error(f"Email search failed: {e}", exc_info=True)
            return False

    async def on_running(self, msg: Message) -> AgentResult:
        """Mark emails as read"""
        from onevalet.providers.email.resolver import AccountResolver
        from onevalet.providers.email.factory import EmailProviderFactory

        if not self.message_ids:
            if not await self._load_emails_to_mark():
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="No emails to mark as read."
                )

        try:
            emails_by_account = {}
            for email in self.emails_to_mark:
                account_name = email.get("_account_name", "primary")
                if account_name not in emails_by_account:
                    emails_by_account[account_name] = []
                emails_by_account[account_name].append(email.get("message_id"))

            total_marked = 0
            failed_accounts = []

            for account_name, msg_ids in emails_by_account.items():
                account = await AccountResolver.resolve_account(self.tenant_id, account_name)
                if not account:
                    failed_accounts.append({
                        "account_name": account_name,
                        "email": "",
                        "reason": "not_found"
                    })
                    continue

                provider = EmailProviderFactory.create_provider(account)
                if not provider:
                    failed_accounts.append({
                        "account_name": account_name,
                        "email": account.get("account_identifier", ""),
                        "reason": "unsupported_provider"
                    })
                    continue

                if not await provider.ensure_valid_token():
                    failed_accounts.append({
                        "account_name": account_name,
                        "email": account.get("account_identifier", ""),
                        "reason": "token_expired"
                    })
                    continue

                result = await provider.mark_as_read(msg_ids)
                if result.get("success"):
                    total_marked += len(msg_ids)
                    logger.info(f"Marked {len(msg_ids)} emails as read in {account_name}")
                else:
                    failed_accounts.append({
                        "account_name": account_name,
                        "email": account.get("account_identifier", ""),
                        "reason": "mark_failed"
                    })

            if total_marked > 0:
                response = f"Done! I've marked {total_marked} email(s) as read."
                if failed_accounts:
                    response += self._format_failed_accounts(failed_accounts)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=response
                )
            else:
                if failed_accounts:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message="I couldn't mark those emails as read." + self._format_failed_accounts(failed_accounts)
                    )
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I couldn't find any emails to mark as read."
                )

        except Exception as e:
            logger.error(f"Mark as read failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Something went wrong while marking emails. Want me to try again?"
            )

    def _format_failed_accounts(self, failed_accounts: list) -> str:
        """Format failed accounts into friendly messages"""
        messages = []
        for failed in failed_accounts:
            if isinstance(failed, str):
                messages.append(f" I couldn't access your {failed} account.")
            else:
                email = failed.get("email", "")
                account_name = failed.get("account_name", "")
                reason = failed.get("reason", "unknown")

                account_display = email if email else (account_name if account_name else "email")

                if reason == "token_expired":
                    messages.append(
                        f" I lost access to your {account_display} account - "
                        f"could you reconnect it in settings?"
                    )
                elif reason == "not_found":
                    messages.append(f" I couldn't find your {account_display} account.")
                elif reason == "unsupported_provider":
                    messages.append(f" Sorry, {account_display} uses an email provider I don't support yet.")
                else:
                    messages.append(f" I had some trouble with {account_display}.")

        return "".join(messages)

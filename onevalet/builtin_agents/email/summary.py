"""
Email Summary Agent - Summarize unread emails for morning digest

This agent scans all connected email accounts for unread emails
from the last 24 hours (primary inbox only) and generates a brief
LLM-powered summary. Designed for use in morning digest pipeline.
"""
import logging
import html
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class EmailSummaryAgent(StandardAgent):
    """Summarize unread emails. Use when the user asks for an email digest or overview of their inbox."""

    def __init__(
        self,
        tenant_id: str = "",
        llm_client=None,
        **kwargs
    ):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

    def needs_approval(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        return {}

    async def on_running(self, msg: Message) -> AgentResult:
        """Scan and summarize unread emails from last 24 hours"""
        from onevalet.providers.email.resolver import AccountResolver
        from onevalet.providers.email.factory import EmailProviderFactory

        logger.info(f"EmailSummaryAgent: Scanning unread emails for user {self.tenant_id}")

        try:
            accounts = await AccountResolver.resolve_accounts(self.tenant_id, ["all"])

            if not accounts:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="no email accounts connected"
                )

            all_emails = []
            checked_accounts = []
            failed_accounts = []

            for account in accounts:
                provider = EmailProviderFactory.create_provider(account)
                if not provider:
                    failed_accounts.append(account.get("account_name", "unknown"))
                    continue

                if not await provider.ensure_valid_token():
                    failed_accounts.append(account.get("account_name", "unknown"))
                    continue

                result = await provider.search_emails(
                    unread_only=True,
                    days_back=1,
                    include_categories=["primary"],
                    max_results=20
                )

                if result.get("success"):
                    emails = result.get("data", [])
                    for email in emails:
                        email["_account_name"] = account.get("account_name", "")
                        email["_account_email"] = account.get("account_identifier", "")
                    all_emails.extend(emails)
                    checked_accounts.append(account.get("account_identifier", account.get("account_name", "")))
                else:
                    failed_accounts.append(account.get("account_name", "unknown"))

            summary = await self._generate_summary(all_emails, checked_accounts, failed_accounts)

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=summary
            )

        except Exception as e:
            logger.error(f"EmailSummaryAgent failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="couldn't check emails"
            )

    async def _generate_summary(
        self,
        emails: List[Dict],
        checked_accounts: List[str],
        failed_accounts: List[str]
    ) -> str:
        """Generate a brief summary of unread emails using LLM"""

        if not emails:
            if failed_accounts and not checked_accounts:
                return "couldn't access email accounts"
            return "no new emails in the last 24 hours"

        emails_context = ""
        for i, email in enumerate(emails[:15], 1):
            sender = html.unescape(email.get("sender", "Unknown"))
            subject = html.unescape(email.get("subject", "No subject"))
            snippet = html.unescape(email.get("snippet", ""))[:100]
            account = email.get("_account_email", "")

            emails_context += f"\n{i}. [{account}] From: {sender}\n"
            emails_context += f"   Subject: {subject}\n"
            if snippet:
                emails_context += f"   Preview: {snippet}\n"

        prompt = f"""Summarize the user's unread emails from the last 24 hours in 2-3 sentences.

Unread emails ({len(emails)} total):
{emails_context}

Instructions:
1. Give a brief overview of what's in their inbox
2. Highlight anything that looks important or needs attention
3. Keep it casual and concise (2-3 sentences max)
4. If there are action items or deadlines, mention them
5. Don't list every email - just summarize the themes/highlights

Example outputs:
- "you have 5 unread emails - mostly newsletters, but there's one from your bank about a payment due tomorrow"
- "3 new emails: a meeting invite for 2pm, a shipping notification, and a promo from amazon"
- "quiet inbox today - just a couple newsletters and a receipt from uber"

Your summary:"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You summarize emails briefly and casually."},
                    {"role": "user", "content": prompt}
                ],
                enable_thinking=False
            )

            summary = result.content.strip()

            if failed_accounts:
                summary += f" (couldn't check: {', '.join(failed_accounts)})"

            return summary

        except Exception as e:
            logger.error(f"Failed to generate email summary: {e}")
            return f"{len(emails)} unread email(s) in the last 24 hours"

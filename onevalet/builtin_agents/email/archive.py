"""
Archive Email Agent

Multi-step agent for archiving emails with search and approval.

State Flow:
1. INITIALIZING -> extract fields, search emails
2. WAITING_FOR_APPROVAL -> show found emails, wait for user confirmation
3. RUNNING -> execute archiving
4. COMPLETED
"""
import logging
import json
import re
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet(triggers=["archive email", "file email"])
class ArchiveEmailAgent(StandardAgent):
    """Archive email agent with search and approval"""

    account = InputField(
        prompt="Which email account would you like to archive from?",
        description="Email account to archive from (default: primary)",
        required=False,
    )
    search_query = InputField(
        prompt="What emails would you like to archive?",
        description="Search query for emails to archive",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

        self.found_emails = []
        self.message_ids = []
        self.account = None
        self.days_back = 7
        self.error_message = None
        self._search_completed = False

    def needs_approval(self) -> bool:
        return True

    def get_approval_prompt(self) -> str:
        """Generate confirmation prompt with list of emails to archive"""
        if self.error_message:
            return self.error_message

        emails = self.found_emails

        if not emails:
            return "I couldn't find any emails matching your search. Want to try different keywords?"

        if len(emails) == 1:
            email = emails[0]
            subject = email.get('subject', 'No subject')
            sender = email.get('sender', 'Unknown')
            return f"Found this email:\n\"{subject}\" from {sender}\n\nArchive it? (yes/no)"

        response_parts = [f"Found {len(emails)} emails:\n"]

        for i, email in enumerate(emails[:5], 1):
            subject = email.get('subject', 'No subject')
            sender = email.get('sender', 'Unknown')
            if len(subject) > 40:
                subject = subject[:37] + "..."
            response_parts.append(f"{i}. \"{subject}\" from {sender}\n")

        if len(emails) > 5:
            response_parts.append(f"...and {len(emails) - 5} more\n")

        response_parts.append(f"\nWhich ones? (reply 1, 1-3, all, or no)")

        return "".join(response_parts)

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract field values from user input"""
        if not self.llm_client:
            return {"search_query": user_input}

        extraction_prompt = f"""Extract email archive criteria from the user's request:

User request: {user_input}

Return JSON:
{{
  "account": "",
  "search_query": "",
  "days_back": 7
}}"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Extract information. Return only JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            extracted = json.loads(result.content.strip())

            result_dict = {}

            account = extracted.get("account", "").strip()
            if account:
                result_dict["account"] = account

            search_query = extracted.get("search_query", "").strip()
            if search_query:
                result_dict["search_query"] = search_query

            self.days_back = extracted.get("days_back", 7)

            return result_dict

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}

    async def on_initializing(self, msg: Message) -> AgentResult:
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        if not self._search_completed:
            await self._search_emails()
            self._search_completed = True

        if not self.found_emails and not self.collected_fields.get("search_query"):
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="What emails would you like to archive? Tell me the sender, subject, or keywords."
            )

        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        await self._search_emails()
        self._search_completed = True

        if not self.found_emails:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="I couldn't find any emails matching that. Try different keywords?"
            )

        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        user_input = msg.get_text() if msg else ""
        approval = await self._parse_approval_with_selection(user_input)

        if approval == "approved":
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

        elif approval == "rejected":
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message="Got it, I won't archive those emails."
            )

        else:
            self._search_completed = False
            self.found_emails = []
            self.message_ids = []

            await self._extract_and_collect_fields(user_input)

            if self.collected_fields.get("search_query"):
                await self._search_emails()
                self._search_completed = True

                if self.found_emails:
                    return self.make_result(
                        status=AgentStatus.WAITING_FOR_APPROVAL,
                        raw_message=self.get_approval_prompt()
                    )

            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="What emails would you like to archive instead?"
            )

    async def _parse_approval_with_selection(self, user_response: str) -> str:
        """Parse user's approval response with email selection support"""
        if not self.llm_client:
            response_lower = user_response.lower().strip()
            if response_lower in ["yes", "y", "ok", "all", "go", "confirm", "do it"]:
                return "approved"
            elif response_lower in ["no", "n", "cancel", "stop", "nevermind"]:
                return "rejected"
            else:
                return "modify"

        email_count = len(self.found_emails)
        prompt = f"""Parse the user's response to select which emails to archive.

Total emails: {email_count} (numbered 1 to {email_count})
User response: "{user_response}"

Determine intent:
- Archive ALL: {{"action": "all"}}
- Archive SPECIFIC: {{"action": "select", "indices": [1, 2, 3]}}
- CANCEL: {{"action": "cancel"}}
- CHANGE search: {{"action": "modify"}}

Return ONLY valid JSON:"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format="json_object",
                enable_thinking=False
            )
            parsed = json.loads(result.content.strip())
            action = parsed.get("action", "modify")

            if action == "all":
                return "approved"
            elif action == "select":
                indices = parsed.get("indices", [])
                selected_emails = []
                selected_ids = []
                for idx in indices:
                    zero_idx = idx - 1
                    if 0 <= zero_idx < len(self.found_emails):
                        selected_emails.append(self.found_emails[zero_idx])
                        selected_ids.append(self.message_ids[zero_idx])

                if selected_emails:
                    self.found_emails = selected_emails
                    self.message_ids = selected_ids
                    return "approved"
                else:
                    return "modify"
            elif action == "cancel":
                return "rejected"
            else:
                return "modify"

        except Exception as e:
            logger.error(f"Failed to parse approval response: {e}")
            return "modify"

    async def _search_emails(self) -> None:
        """Search for emails matching the criteria"""
        from onevalet.providers.email.resolver import AccountResolver
        from onevalet.providers.email.factory import EmailProviderFactory

        search_query = self.collected_fields.get("search_query")

        # Check email cache first
        email_cache = self.context_hints.get("email_cache", {})
        if email_cache:
            cached_emails_list = email_cache.get("emails", [])
            cached_message_ids = email_cache.get("message_ids", [])
            cached_accounts = email_cache.get("accounts", [])

            if cached_emails_list:
                if not search_query:
                    all_cached = self._build_cached_emails(cached_emails_list, cached_message_ids, cached_accounts)
                    if all_cached:
                        self.found_emails = all_cached
                        self.message_ids = cached_message_ids
                        if cached_accounts:
                            account_spec = cached_accounts[0] if cached_accounts[0] else "primary"
                            self.account = AccountResolver.resolve_account(self.tenant_id, account_spec)
                        return

                cached_emails = await self._filter_emails_with_llm(
                    self._build_cached_emails(cached_emails_list, cached_message_ids, cached_accounts),
                    search_query
                )
                if cached_emails:
                    self.found_emails = cached_emails
                    self.message_ids = [e.get("message_id") for e in cached_emails if e.get("message_id")]
                    if cached_emails[0].get("account"):
                        self.account = AccountResolver.resolve_account(self.tenant_id, cached_emails[0].get("account", "primary"))
                    return

        if not search_query:
            return

        try:
            account_spec = self.collected_fields.get("account", "primary")
            account = AccountResolver.resolve_account(self.tenant_id, account_spec)

            if not account:
                self.error_message = f"I couldn't find your '{account_spec}' email account."
                return

            self.account = account

            provider = EmailProviderFactory.create_provider(account)
            if not provider:
                self.error_message = "Sorry, I can't access that email account yet."
                return

            if not await provider.ensure_valid_token():
                self.error_message = "I lost access to your email account."
                return

            result = await provider.search_emails(
                query=search_query,
                max_results=20,
                days_back=self.days_back if self.days_back > 0 else None
            )

            if not result.get("success"):
                return

            emails = result.get("data", [])

            if emails and self.llm_client:
                emails = await self._filter_emails_with_llm(emails, search_query)

            if not emails and search_query:
                emails = await self._fallback_search_with_llm(provider, search_query)

            if emails:
                self.found_emails = emails
                self.message_ids = [email["message_id"] for email in emails]

        except Exception as e:
            logger.error(f"Failed to search emails: {e}", exc_info=True)

    def _build_cached_emails(self, emails: List, message_ids: List, accounts: List) -> List[Dict]:
        """Build email objects from cache data"""
        result = []
        for i, email in enumerate(emails):
            email_copy = email.copy() if isinstance(email, dict) else {"subject": str(email)}
            if i < len(message_ids):
                email_copy["message_id"] = message_ids[i]
            if i < len(accounts):
                email_copy["account"] = accounts[i]
            result.append(email_copy)
        return result

    async def _filter_emails_with_llm(self, emails: List[Dict], search_query: str) -> List[Dict]:
        """Use LLM to filter emails"""
        if not emails or not self.llm_client or not search_query:
            return emails

        email_list = [
            {"index": i, "subject": e.get("subject", ""), "from": e.get("sender", e.get("from", "")), "snippet": e.get("snippet", "")[:200]}
            for i, e in enumerate(emails)
        ]

        prompt = f"""Find emails matching: "{search_query}"

Emails: {json.dumps(email_list)}

Return a JSON array of matching indices (0-based), like: [0, 3, 5]"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                enable_thinking=False
            )
            match = re.search(r'\[[\d,\s]*\]', result.content)
            if match:
                indices = json.loads(match.group())
                return [emails[i] for i in indices if 0 <= i < len(emails)]
        except Exception as e:
            logger.error(f"Filter LLM failed: {e}")

        return emails

    async def _fallback_search_with_llm(self, provider, search_query: str) -> List[Dict]:
        """Fallback: get recent emails and filter with LLM"""
        result = await provider.search_emails(query=None, max_results=30, days_back=7)
        if not result.get("success"):
            return []

        all_emails = result.get("data", [])
        if not all_emails or not self.llm_client:
            return []

        return await self._filter_emails_with_llm(all_emails, search_query)

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute email archiving"""
        from onevalet.providers.email.factory import EmailProviderFactory

        if not self.message_ids:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I couldn't find any emails to archive."
            )

        if not self.account:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I'm not sure which email account to use."
            )

        try:
            provider = EmailProviderFactory.create_provider(self.account)
            if not provider:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="Sorry, I can't access that email account."
                )

            if not await provider.ensure_valid_token():
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I lost access to your email account."
                )

            result = await provider.archive_emails(message_ids=self.message_ids)

            if result.get("success"):
                archived_count = result.get("archived_count", len(self.message_ids))
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Done! I've archived {archived_count} email(s)."
                )
            else:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"I had trouble archiving those emails. {result.get('error', '')}"
                )

        except Exception as e:
            logger.error(f"Failed to archive emails: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Something went wrong. Want me to try again?"
            )

"""
Read Email Agent - Query and search emails

This agent handles email search queries:
- Check for new/unread emails
- Search by sender
- Search by date range
- Comprehensive search queries

This is a read-only agent (no approval needed).
"""
import logging
import json
import html
import random
from typing import Dict, Any, List, Optional
from datetime import datetime

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet(triggers=["check email", "read email", "search email", "any email", "new email"])
class ReadEmailAgent(StandardAgent):
    """Email search and query agent"""

    accounts = InputField(
        prompt="Which email accounts would you like to search?",
        description="Email accounts to search (account names like 'work', 'personal', or 'all', default: primary)",
        required=False,
    )
    search_query = InputField(
        prompt="What emails would you like me to search for?",
        description="General search query or description of what emails to find",
        required=False,
    )

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

        # Pagination state
        self._cached_emails = []
        self._cached_summaries = {}
        self._current_offset = 0
        self._page_size = 10
        self._cached_accounts = []
        self._cached_failed_accounts = []

    def needs_approval(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract search criteria from user input"""
        # Check if this is a "show more" request
        if self.collected_fields.get("show_more"):
            logger.info(f"User requesting more emails from cache (offset: {self._current_offset})")
            return {"show_more": True}

        # Check for "more" keywords when agent has cached emails
        if self._cached_emails:
            more_keywords = ["more", "next", "continue", "rest", "remaining"]
            user_lower = user_input.lower().strip()
            if any(user_lower == kw or user_lower.startswith(kw + " ") for kw in more_keywords):
                logger.info(f"Detected 'more' request from user input: {user_input}")
                return {"show_more": True}

        if not self.llm_client:
            return {"search_query": user_input}

        try:
            from onevalet.providers.email.resolver import AccountResolver
            all_accounts = await AccountResolver.resolve_accounts(self.tenant_id, ["all"])

            account_context = ""
            if all_accounts:
                account_info = []
                for acc in all_accounts:
                    name = acc.get("account_name", "unknown")
                    email = acc.get("account_identifier", "")
                    if email:
                        account_info.append(f"{name} ({email})")
                    else:
                        account_info.append(name)
                account_context = f"\n**User's available email accounts:** {', '.join(account_info)}\n"

            prompt = f"""Extract email search criteria from the user's message.
{account_context}

User message: "{user_input}"

Extract the following information if present:
1. accounts: Which email accounts to search (array of account names, ["all"], or ["primary"])
2. sender: Email address or name of sender to filter by
3. date_range: Date or date range (e.g., "today", "yesterday", "last week")
4. unread_only: Whether to show only unread emails (true/false)
5. days_back: Number of days to search back (integer)
6. include_categories: Gmail categories to search (array: ["primary"], ["social"], ["promotions"])
7. search_query: General search terms or keywords

CRITICAL DEFAULT BEHAVIOR:
ALWAYS default to unread emails UNLESS user specifies a specific search.
For generic queries like "check email", "any email", etc.:
- Set unread_only: true
- Set days_back: 7
- Set include_categories: ["primary"]

Return ONLY the JSON object, no explanations.

JSON Output:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract email search criteria from text and return JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            content = result.content.strip()
            extracted = json.loads(content)

            if not extracted:
                extracted = {"search_query": user_input}
            else:
                if "search_query" not in extracted:
                    extracted["search_query"] = None

            logger.info(f"Extracted email search criteria: {extracted}")
            return extracted

        except Exception as e:
            logger.error(f"Field extraction failed: {e}", exc_info=True)
            return {"search_query": user_input}

    async def on_running(self, msg: Message) -> AgentResult:
        """Search emails based on extracted criteria"""
        from onevalet.providers.email.resolver import AccountResolver
        from onevalet.providers.email.factory import EmailProviderFactory

        fields = self.collected_fields
        logger.info(f"Searching emails with criteria: {fields}")

        # Handle "show more" request
        if fields.get("show_more") and self._cached_emails:
            result = await self._show_more_emails()
            self.collected_fields.pop("show_more", None)
            return result

        # Clear show_more flag and reset pagination when doing new search
        self.collected_fields.pop("show_more", None)
        self._current_offset = 0

        # Default to unread if no search criteria
        search_criteria = {k: v for k, v in fields.items() if k not in ["accounts", "show_more"]}
        if not search_criteria:
            fields["unread_only"] = True
            logger.info("No search criteria specified, defaulting to unread emails")

        if "unread_only" not in fields:
            fields["unread_only"] = True

        # Default to primary inbox for unread/generic queries (skip promotions, social, etc.)
        if fields.get("unread_only") and not fields.get("include_categories"):
            fields["include_categories"] = ["primary"]

        # Default to last 7 days for generic unread queries
        if fields.get("unread_only") and not fields.get("days_back") and not fields.get("date_range"):
            fields["days_back"] = 7

        try:
            account_specs = fields.get("accounts")
            if account_specs and isinstance(account_specs, str):
                account_specs = [account_specs]

            if not account_specs:
                account_specs = ["all"]

            accounts = await AccountResolver.resolve_accounts(self.tenant_id, account_specs)

            if not accounts:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="No email accounts found. Please connect an email account first."
                )

            all_emails = []
            failed_accounts = []

            for account in accounts:
                provider = EmailProviderFactory.create_provider(account)
                if not provider:
                    failed_accounts.append({
                        "account_name": account["account_name"],
                        "email": account.get("account_identifier", ""),
                        "reason": "unsupported_provider"
                    })
                    continue

                if not await provider.ensure_valid_token():
                    failed_accounts.append({
                        "account_name": account["account_name"],
                        "email": account.get("account_identifier", ""),
                        "reason": "token_expired"
                    })
                    continue

                search_query = fields.get("search_query")
                meta_keywords = {"unread", "new", "recent", "latest", "all"}
                if search_query and search_query.lower() in meta_keywords:
                    search_query = None

                result = await provider.search_emails(
                    query=search_query,
                    sender=fields.get("sender"),
                    date_range=fields.get("date_range"),
                    unread_only=fields.get("unread_only", False),
                    days_back=fields.get("days_back"),
                    include_categories=fields.get("include_categories"),
                    max_results=20
                )

                if result.get("success"):
                    emails = result.get("data", [])
                    for email in emails:
                        email["_account_name"] = account["account_name"]
                        email["_account_email"] = account["account_identifier"]
                        email["_provider"] = account["provider"]
                    all_emails.extend(emails)
                else:
                    failed_accounts.append({
                        "account_name": account["account_name"],
                        "email": account.get("account_identifier", ""),
                        "reason": "search_failed",
                        "error": result.get("error", "Unknown error")
                    })

            # Cache results for pagination
            self._cached_emails = all_emails
            self._cached_accounts = accounts
            self._cached_failed_accounts = failed_accounts
            self._current_offset = 0

            # Generate summaries
            page_emails = all_emails[:self._page_size]
            email_summaries = await self._generate_email_summaries(page_emails)
            self._cached_summaries = email_summaries

            # Format results
            formatted_results = await self._format_search_results(
                page_emails, accounts, failed_accounts, email_summaries,
                total_count=len(all_emails), offset=0
            )

            self._current_offset = self._page_size

            has_more = len(all_emails) > self._page_size

            if has_more:
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message=formatted_results,
                    metadata={"has_more": True}
                )
            else:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=formatted_results
                )

        except Exception as e:
            logger.error(f"Email search failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't search your emails. Mind trying again later?"
            )

    async def _show_more_emails(self) -> AgentResult:
        """Show next page of cached emails"""
        if not self._cached_emails:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No cached emails. Try searching again."
            )

        if self._current_offset >= len(self._cached_emails):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No more emails to show."
            )

        page_emails = self._cached_emails[self._current_offset:self._current_offset + self._page_size]

        if not page_emails:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No more emails to show."
            )

        email_summaries = await self._generate_email_summaries(page_emails)

        formatted_results = await self._format_search_results(
            page_emails,
            self._cached_accounts,
            [],
            email_summaries,
            total_count=len(self._cached_emails),
            offset=self._current_offset
        )

        self._current_offset += self._page_size

        has_more = self._current_offset < len(self._cached_emails)

        if has_more:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=formatted_results,
                metadata={"has_more": True}
            )
        else:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=formatted_results
            )

    async def _generate_email_summaries(self, emails: List[Dict]) -> Dict[str, str]:
        """Generate one-line summaries for emails using LLM"""
        if not emails:
            return {}

        try:
            emails_text = ""
            for i, email in enumerate(emails[:10], 1):
                sender = html.unescape(email.get("sender", "Unknown"))
                subject = html.unescape(email.get("subject", "No subject"))
                snippet = html.unescape(email.get("snippet", ""))[:150]

                emails_text += f"\nEmail {i}:\n"
                emails_text += f"From: {sender}\n"
                emails_text += f"Subject: {subject}\n"
                emails_text += f"Preview: {snippet}\n"

            prompt = f"""Summarize each email in 5-8 words MAX. Be extremely concise - capture only the core point.

{emails_text}

Return ONLY a JSON object with email numbers as keys and summaries as values. Example:
{{"1": "Domain renewal notice expiring soon", "2": "Meeting tomorrow at 2pm"}}

Keep each summary under 40 characters.

JSON Output:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You summarize emails in 5-8 words, under 40 characters."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            content = result.content
            if not content:
                return {}

            summaries = json.loads(content.strip())
            logger.info(f"Generated {len(summaries)} email summaries")
            return summaries

        except Exception as e:
            logger.error(f"Failed to generate email summaries: {e}", exc_info=True)
            return {}

    async def _analyze_action_items(
        self,
        emails: List[Dict],
        email_summaries: Dict[str, str]
    ) -> Optional[str]:
        """Analyze unread emails to identify action items"""
        unread_emails = [email for email in emails if email.get("unread", False)]

        if not unread_emails:
            return None

        try:
            emails_context = ""
            for i, email in enumerate(unread_emails[:5], 1):
                sender = html.unescape(email.get("sender", "Unknown"))
                subject = html.unescape(email.get("subject", "No subject"))
                snippet = html.unescape(email.get("snippet", ""))[:150]

                summary_key = str(i)
                if summary_key in email_summaries:
                    summary = html.unescape(email_summaries[summary_key])
                else:
                    summary = snippet

                emails_context += f"\nEmail {i}:\n"
                emails_context += f"From: {sender}\n"
                emails_context += f"Subject: {subject}\n"
                emails_context += f"Summary: {summary}\n"

            current_time = datetime.now()

            prompt = f"""Analyze the user's unread emails and tell them if there are any tasks that need completion TODAY ({current_time.strftime('%B %d, %Y')}).

Context:
- Current time: {current_time.strftime('%A, %B %d, %Y at %I:%M %p')}
- User has {len(unread_emails)} unread email(s)

Unread Emails:
{emails_context}

Instructions:
1. Analyze if there are tasks, meetings, deadlines, or actions requiring attention TODAY
2. If YES: Tell the user what they need to complete today (2-3 sentences max)
3. If NO: Tell them casually there's nothing urgent (1 sentence)

CRITICAL:
- You are ONLY notifying the user, you cannot do anything for them
- DO NOT say "I'd check", "I'll do", or "I recommend"
- Just state what needs attention
- Only mention tasks for TODAY
- Keep it short (2-3 sentences max)

Your Response:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You analyze emails for action items."},
                    {"role": "user", "content": prompt}
                ],
                enable_thinking=False
            )

            return result.content.strip()

        except Exception as e:
            logger.error(f"Failed to analyze action items: {e}", exc_info=True)
            return None

    async def _format_search_results(
        self,
        emails: List[Dict],
        searched_accounts: List[Dict],
        failed_accounts: List[Dict],
        email_summaries: Dict[str, str],
        total_count: int = None,
        offset: int = 0
    ) -> str:
        """Format email search results"""
        if not emails and not failed_accounts:
            return await self._generate_no_results_message()

        response_parts = []

        if not emails:
            response_parts.append("No emails found.")
        else:
            if total_count and total_count > len(emails):
                if offset == 0:
                    response_parts.append(f"Found {total_count} email(s), showing 1-{len(emails)}:")
                else:
                    end_idx = offset + len(emails)
                    response_parts.append(f"Emails {offset + 1}-{end_idx} of {total_count}:")
            else:
                response_parts.append(f"Found {len(emails)} email(s):")

            for i, email in enumerate(emails, 1):
                summary_key = str(i)
                if summary_key in email_summaries:
                    summary = html.unescape(email_summaries[summary_key])
                else:
                    summary = html.unescape(email.get("subject", "No subject"))

                if len(summary) > 50:
                    summary = summary[:47] + "..."

                # Extract sender name (strip email address for brevity)
                sender_raw = html.unescape(email.get("sender", "Unknown"))
                if "<" in sender_raw:
                    sender = sender_raw.split("<")[0].strip().strip('"')
                else:
                    sender = sender_raw

                # Extract short date
                date_raw = email.get("date", "")
                date_short = self._format_short_date(date_raw)

                if len(searched_accounts) > 1:
                    account_name = email.get('_account_name', 'Unknown')
                    summary = f"[{account_name}] {summary}"

                global_idx = offset + i
                email_text = f"{global_idx}. {summary}\n   From: {sender}  |  {date_short}"
                response_parts.append(email_text)

            if total_count and offset + len(emails) < total_count:
                remaining = total_count - offset - len(emails)
                response_parts.append(f"\n+{remaining} more (say 'more' to see)")

        if failed_accounts:
            for failed in failed_accounts:
                if isinstance(failed, str):
                    response_parts.append(f"\nI couldn't check your {failed} account.")
                else:
                    account_name = failed.get("account_name", "")
                    email = failed.get("email", "")
                    reason = failed.get("reason", "unknown")

                    account_display = email if email else account_name if account_name else "email"

                    if reason == "token_expired":
                        response_parts.append(
                            f"\nI lost access to your {account_display} account. "
                            f"Could you reconnect it in settings?"
                        )
                    elif reason == "unsupported_provider":
                        response_parts.append(
                            f"\nSorry, I can't access {account_display} yet - that email provider isn't supported."
                        )
                    else:
                        response_parts.append(
                            f"\nI had trouble checking {account_display}. Want me to try again later?"
                        )

        return "\n".join(response_parts)

    @staticmethod
    def _format_short_date(date_str: str) -> str:
        """Format email date string to short display format like 'Feb 9' or 'Jan 28, 2025'."""
        if not date_str:
            return ""
        try:
            from dateutil import parser as date_parser
            dt = date_parser.parse(date_str)
            now = datetime.now()
            if dt.year == now.year:
                return dt.strftime("%b %d, %I:%M %p").lstrip("0")
            else:
                return dt.strftime("%b %d, %Y").lstrip("0")
        except Exception:
            return date_str[:16] if len(date_str) > 16 else date_str

    async def _generate_no_results_message(self) -> str:
        """Generate contextual message when no emails found"""
        criteria = []
        generic_queries = ["any email", "email", "emails", "check email", "check my email", "new email", "my email"]

        if self.collected_fields.get("sender"):
            criteria.append(f"from {self.collected_fields['sender']}")

        search_query = self.collected_fields.get("search_query", "")
        if search_query and search_query.lower() not in generic_queries:
            criteria.append(f"about '{search_query}'")

        if self.collected_fields.get("date_range"):
            criteria.append(f"from {self.collected_fields['date_range']}")

        if not criteria:
            responses = [
                "Your inbox is all caught up - no new emails right now.",
                "Nothing new in your inbox at the moment.",
                "All clear! No unread emails to show.",
                "No new emails waiting for you.",
                "Your inbox is empty - you're all caught up!"
            ]
            return random.choice(responses)

        criteria_str = " ".join(criteria)
        return f"I didn't find any emails {criteria_str}."

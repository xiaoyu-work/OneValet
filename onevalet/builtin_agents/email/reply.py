"""
Reply Email Agent

Agent for replying to emails with LLM-composed content based on user's instructions.
Uses state machine: on_initializing -> on_waiting_for_input -> on_waiting_for_approval -> on_running
"""
import logging
import json
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message, ApprovalResult

logger = logging.getLogger(__name__)


@valet()
class ReplyEmailAgent(StandardAgent):
    """Reply to email agent with LLM-composed reply content"""

    target = InputField(
        prompt="Which email would you like to reply to?",
        description="Which email to reply to: email number (1, 2, etc.), sender name/email, or subject keywords",
    )
    reply_instructions = InputField(
        prompt="What would you like me to say in the reply?",
        description="What to say in the reply - user's instructions for the reply content",
    )
    reply_all = InputField(
        prompt="Should I reply to all recipients?",
        description="Whether to reply to all recipients (true/false)",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

        self.target_email = None
        self.account = None
        self.composed_reply = None

    def needs_approval(self) -> bool:
        return True

    def get_approval_prompt(self) -> str:
        """Generate reply draft for approval"""
        if not self.target_email or not self.composed_reply:
            return "I couldn't compose a reply. Please try again."

        original_subject = self.target_email.get("subject", "No subject")
        original_sender = self.target_email.get("sender", "Unknown")
        reply_all = self.collected_fields.get("reply_all", False)

        response_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {original_sender}:",
            "",
            "---",
            self.composed_reply,
            "---",
            ""
        ]

        if reply_all:
            response_parts.append("(Replying to all recipients)")

        response_parts.append("\nSend this reply?")

        return "\n".join(response_parts)

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract target email and reply instructions from user input"""
        if not self.llm_client:
            return {}

        # Use email cache from context_hints instead of orchestrator_callback
        cached_emails = self.context_hints.get("email_cache", {}).get("emails", [])

        email_context = ""
        if cached_emails:
            email_lines = []
            for i, email in enumerate(cached_emails[:10], 1):
                sender = email.get("sender", "Unknown")
                subject = email.get("subject", "No subject")
                email_lines.append(f"{i}. From: {sender} | Subject: {subject}")
            email_context = "Recent emails:\n" + "\n".join(email_lines)

        prompt = f"""Extract the target email and reply instructions from this user message.

{email_context}

User message: "{user_input}"

Return JSON with:
- "target": email number (1, 2, etc.), sender name/email, or subject keywords to identify which email
- "reply_instructions": what the user wants to say in the reply (their intent/content)
- "reply_all": true if user wants to reply to all, false otherwise

Return only valid JSON:"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format="json_object",
                enable_thinking=False
            )

            response_text = result.content.strip()
            if not response_text:
                return {}

            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Failed to extract reply fields: {e}")
            return {}

    async def _find_target_email(self, target: str) -> bool:
        """Find the email to reply to from cache or search"""
        from onevalet.providers.email.resolver import AccountResolver
        from onevalet.providers.email.factory import EmailProviderFactory

        # Use email cache from context_hints
        cached_emails = self.context_hints.get("email_cache", {}).get("emails", [])

        if target.isdigit():
            idx = int(target) - 1
            if 0 <= idx < len(cached_emails):
                self.target_email = cached_emails[idx]
                account_name = self.target_email.get("_account_name", "primary")
                self.account = await AccountResolver.resolve_account(self.tenant_id, account_name)
                return True

        target_lower = target.lower()
        for email in cached_emails:
            sender = email.get("sender", "").lower()
            subject = email.get("subject", "").lower()
            if target_lower in sender or target_lower in subject:
                self.target_email = email
                account_name = email.get("_account_name", "primary")
                self.account = await AccountResolver.resolve_account(self.tenant_id, account_name)
                return True

        accounts = await AccountResolver.resolve_accounts(self.tenant_id, ["all"])
        for account in accounts:
            provider = EmailProviderFactory.create_provider(account)
            if not provider:
                continue
            if not await provider.ensure_valid_token():
                continue

            result = await provider.search_emails(
                query=target,
                max_results=5,
                days_back=7
            )

            if result.get("success") and result.get("data"):
                self.target_email = result["data"][0]
                self.target_email["_account_name"] = account["account_name"]
                self.account = account
                return True

        return False

    async def _compose_reply(self, instructions: str) -> str:
        """Use LLM to compose reply based on user's instructions"""
        if not self.llm_client or not self.target_email:
            return instructions

        original_sender = self.target_email.get("sender", "Unknown")
        original_subject = self.target_email.get("subject", "")
        original_snippet = self.target_email.get("snippet", "")

        prompt = f"""Extract the core message from the user's reply instructions. Keep it simple and direct.

Original email:
- From: {original_sender}
- Subject: {original_subject}
- Preview: {original_snippet}

User's instructions: "{instructions}"

Examples:
- "tell them yes" -> "Yes, sounds good."
- "say I'll be there" -> "I'll be there."
- "let them know I'm busy" -> "I'm busy, sorry."
- "confirm the meeting" -> "Confirmed."

Return ONLY the simple, direct reply message:"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                enable_thinking=False
            )
            return result.content.strip()
        except Exception as e:
            logger.error(f"Failed to compose reply: {e}")
            return instructions

    # ===== State Handlers =====

    async def on_initializing(self, msg: Message) -> AgentResult:
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        missing = self._get_missing_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing
            )

        target = self.collected_fields.get("target", "")
        if not await self._find_target_email(target):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"I couldn't find an email matching '{target}'. Could you be more specific?"
            )

        instructions = self.collected_fields.get("reply_instructions", "")
        self.composed_reply = await self._compose_reply(instructions)

        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        missing = self._get_missing_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing
            )

        target = self.collected_fields.get("target", "")
        if not await self._find_target_email(target):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"I couldn't find an email matching '{target}'. Could you be more specific?"
            )

        instructions = self.collected_fields.get("reply_instructions", "")
        self.composed_reply = await self._compose_reply(instructions)

        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        user_input = msg.get_text() if msg else ""
        approval = await self._parse_approval_async(user_input)

        if approval == ApprovalResult.APPROVED:
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

        elif approval == ApprovalResult.REJECTED:
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message="OK, cancelled."
            )

        else:  # MODIFY
            await self._extract_and_collect_fields(user_input)

            instructions = self.collected_fields.get("reply_instructions", "")
            if instructions:
                self.composed_reply = await self._compose_reply(instructions)

            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )

    async def _parse_approval_async(self, user_input: str):
        """Parse user's approval response using LLM."""
        prompt = f"""The user was asked to approve sending an email reply. Their response was:
"{user_input}"

What is the user's intent?
- APPROVED: User wants to send the reply (yes, ok, send it, go ahead, etc.)
- REJECTED: User wants to cancel (no, cancel, don't send, nevermind, etc.)
- MODIFY: User wants to change something (change the message, different wording, etc.)

Return ONLY one word: APPROVED, REJECTED, or MODIFY"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                enable_thinking=False
            )
            response = result.content.strip().upper()

            if "APPROVED" in response:
                return ApprovalResult.APPROVED
            elif "REJECTED" in response:
                return ApprovalResult.REJECTED
            else:
                return ApprovalResult.MODIFY
        except Exception as e:
            logger.error(f"Failed to parse approval: {e}")
            return ApprovalResult.MODIFY

    async def on_running(self, msg: Message) -> AgentResult:
        """Send the composed reply"""
        from onevalet.providers.email.factory import EmailProviderFactory

        fields = self.collected_fields
        target = fields.get("target", "")
        instructions = fields.get("reply_instructions", "")
        reply_all = fields.get("reply_all", False)

        if not self.target_email:
            if not await self._find_target_email(target):
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"I couldn't find an email matching '{target}'. Could you be more specific?"
                )

        if not self.composed_reply:
            self.composed_reply = await self._compose_reply(instructions)

        if not self.account:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I'm not sure which email account to use. Could you specify?"
            )

        account_email = self.account.get("account_identifier", "your email")

        provider = EmailProviderFactory.create_provider(self.account)
        if not provider:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Sorry, I can't access {account_email} yet - that email provider isn't supported."
            )

        if not await provider.ensure_valid_token():
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"I lost access to your {account_email} account. Could you reconnect it in settings?"
            )

        if not hasattr(provider, 'reply_email'):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Reply functionality isn't supported for this email provider yet."
            )

        reply_with_signature = self.composed_reply

        message_id = self.target_email.get("message_id")
        result = await provider.reply_email(
            original_message_id=message_id,
            body=reply_with_signature,
            reply_all=reply_all
        )

        if result.get("success"):
            replied_to = result.get("replied_to", self.target_email.get("sender", ""))
            logger.info(f"Reply sent to {replied_to}")
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Done! I've sent your reply to {replied_to}."
            )
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Reply failed: {error_msg}")
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"I couldn't send that reply. {error_msg}"
            )

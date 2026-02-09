"""
Send Email Agent - Sends emails using StandardAgent base class
"""
import logging
import json
import re
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message, ApprovalResult

logger = logging.getLogger(__name__)


def validate_email(email: str) -> str | None:
    """Validate email format. Returns None if valid, error message if invalid."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email.strip()):
        return None
    return "Invalid email format"


@valet(triggers=["send email", "compose email", "write email"])
class SendEmailAgent(StandardAgent):
    """Send email agent with field collection and approval"""

    from_account = InputField(
        prompt="Which email account would you like to send from?",
        description="Email account to send from",
        required=False,
    )
    recipients = InputField(
        prompt="Who should I send this to?",
        description="Email recipient address(es)",
        validator=validate_email,
    )
    subject = InputField(
        prompt="What should the email subject be?",
        description="Email subject",
        required=False,
    )
    body = InputField(
        prompt="What would you like to say in the email?",
        description="Email body content",
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )
        self.from_email = None
        self.from_account_name = None

    def needs_approval(self) -> bool:
        return True

    async def parse_approval_async(self, user_input: str):
        """Parse user's approval response using LLM."""
        prompt = f"""The user was asked to approve sending an email. Their response was:
"{user_input}"

What is the user's intent?
- APPROVED: User wants to send the email (yes, ok, send it, go ahead, etc.)
- REJECTED: User wants to cancel (no, cancel, don't send, nevermind, etc.)
- MODIFY: User wants to change something (change the subject, different recipient, etc.)

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

    # ===== State Handlers =====

    async def on_initializing(self, msg: Message) -> AgentResult:
        """
        Called when agent first starts.
        Extract fields and resolve sender account.
        """
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        # Resolve sender account early
        await self._resolve_sender_account()

        # Check missing fields
        missing = self._get_missing_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing
            )

        # All fields collected - prepare and go to approval
        await self._prepare_email_content()
        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        """
        Called when collecting fields from user.
        """
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        missing = self._get_missing_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing
            )

        # All fields collected - prepare and go to approval
        await self._prepare_email_content()
        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        """
        Called when waiting for user approval.
        Handle yes/no/modify responses.
        """
        user_input = msg.get_text() if msg else ""
        approval = await self.parse_approval_async(user_input)

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

            missing = self._get_missing_fields()
            if missing:
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message=self._get_next_prompt(),
                    missing_fields=missing
                )

            await self._prepare_email_content()

            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )

    async def on_running(self, msg: Message) -> AgentResult:
        """
        Called when approved. Execute email sending.
        """
        from onevalet.providers.email.resolver import AccountResolver
        from onevalet.providers.email.factory import EmailProviderFactory

        fields = self.collected_fields
        recipients = fields["recipients"]
        subject = fields.get("subject", "Quick note")
        body = fields["body"]

        # Add signature from user profile if available
        user_profile = self.context_hints.get("user_profile", {})
        first_name = user_profile.get("first_name", "") if user_profile else ""
        if first_name:
            body_with_signature = f"{body}\n\nThanks,\n{first_name}"
        else:
            body_with_signature = f"{body}\n\nThanks"

        logger.info(f"Sending email to {recipients}: {subject}")

        try:
            from_account_spec = fields.get("from_account", "primary")
            account = AccountResolver.resolve_account(self.tenant_id, from_account_spec)

            if not account:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I couldn't find your email account. Please connect one in settings."
                )

            account_email = account.get("account_identifier", from_account_spec)

            provider = EmailProviderFactory.create_provider(account)
            if not provider:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Sorry, I can't send emails from {account_email} - that provider isn't supported yet."
                )

            if not await provider.ensure_valid_token():
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"I lost access to your {account_email} account. Please reconnect it in settings."
                )

            result = await provider.send_email(
                to=recipients,
                subject=subject,
                body=body_with_signature
            )

            if result.get("success"):
                logger.info(f"Email sent from {account_email} to {recipients}")
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Done! Sent to {recipients}."
                )
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Email send failed: {error_msg}")
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't send the email: {error_msg}"
                )

        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Something went wrong sending your email. Want to try again?"
            )

    # ===== Helper Methods =====

    async def _resolve_sender_account(self):
        """Resolve the sender email account."""
        from onevalet.providers.email.resolver import AccountResolver

        from_account_spec = self.collected_fields.get("from_account", "primary")
        account = AccountResolver.resolve_account(self.tenant_id, from_account_spec)

        if account:
            self.from_email = account.get("account_identifier", "")
            self.from_account_name = account.get("account_name", "")
            logger.info(f"Resolved sender: {self.from_email} ({self.from_account_name})")
        else:
            logger.warning(f"No email account found for tenant {self.tenant_id}")
            self.from_email = None
            self.from_account_name = None

    async def _prepare_email_content(self):
        """Prepare email content before approval - generate body and subject if needed."""
        raw_body = self.collected_fields.get("body", "")
        if raw_body and self._needs_body_generation(raw_body):
            generated_body = await self._generate_email_body(raw_body)
            self.collected_fields["body"] = generated_body
            logger.info(f"Generated body from intent: '{raw_body[:30]}...'")

        if not self.collected_fields.get("subject"):
            body = self.collected_fields.get("body", "")
            subject = await self._generate_subject(body)
            self.collected_fields["subject"] = subject
            logger.info(f"Auto-generated subject: {subject}")

    def _needs_body_generation(self, body: str) -> bool:
        """Check if body looks like user intent rather than actual email content."""
        intent_phrases = ["telling", "say", "let them know", "inform", "wish", "ask", "remind"]
        body_lower = body.lower().strip()
        for phrase in intent_phrases:
            if body_lower.startswith(phrase):
                return True
        if len(body) < 20:
            return True
        return False

    async def _generate_email_body(self, intent: str) -> str:
        """Extract the core message from user's intent without embellishment."""
        try:
            prompt = f"""Extract the core message from this user intent. Do NOT add any extra words or embellishment.

User intent: "{intent}"

Examples:
- "telling him happy holidays" -> "Happy holidays!"
- "say I'll be late" -> "I'll be late."
- "wish her happy birthday" -> "Happy birthday!"
- "let them know the meeting is cancelled" -> "The meeting is cancelled."
- "remind him about dinner tomorrow" -> "Just a reminder about dinner tomorrow."

Return ONLY the simple, direct message:"""

            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                enable_thinking=False
            )
            return result.content.strip()
        except Exception as e:
            logger.error(f"Failed to extract body: {e}")
            return intent

    async def _generate_subject(self, body: str) -> str:
        """Generate email subject from body content."""
        if not body:
            return "Quick note"

        try:
            prompt = f"""Generate a short email subject (3-6 words) for this email:

{body}

Return ONLY the subject line:"""

            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                enable_thinking=False
            )
            subject = result.content.strip().strip('"\'')
            return subject if subject else "Quick note"
        except Exception as e:
            logger.error(f"Failed to generate subject: {e}")
            return "Quick note"

    def get_approval_prompt(self) -> str:
        """Generate email draft for user approval."""
        recipients = self.collected_fields.get("recipients", "")
        subject = self.collected_fields.get("subject", "")
        body = self.collected_fields.get("body", "")

        # Add signature from user profile if available
        user_profile = self.context_hints.get("user_profile", {})
        first_name = user_profile.get("first_name", "") if user_profile else ""
        if first_name:
            body_with_signature = f"{body}\n\nThanks,\n{first_name}"
        else:
            body_with_signature = f"{body}\n\nThanks"

        from_email = self.from_email or "your-email@example.com"

        return f"""Email Draft:
From: {from_email}
To: {recipients}
Subject: {subject}

{body_with_signature}

---
Send this?"""

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract email information from user input using LLM."""
        extraction_prompt = f"""Extract email information from the user's message.

User message: {user_input}

Return JSON with these fields (leave empty if not mentioned):
{{
  "from_account": "",
  "recipients": "",
  "subject": "",
  "body": ""
}}

Rules:
- ONLY extract explicitly stated information
- recipients: email address(es), comma-separated if multiple
- body: the message content or intent (e.g., "telling him happy birthday")"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Extract email info. Return valid JSON only."},
                    {"role": "user", "content": extraction_prompt}
                ],
                enable_thinking=False
            )

            response_text = result.content.strip()
            if not response_text:
                return {}

            extracted = json.loads(response_text)
            result_dict = {}

            for field in ["from_account", "recipients", "subject", "body"]:
                value = extracted.get(field, "").strip()
                if value:
                    result_dict[field] = value

            logger.info(f"Extracted fields: {list(result_dict.keys())}")
            return result_dict

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}

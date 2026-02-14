"""OneValet Email Event Handler — LLM-powered email importance evaluation."""

import json
import logging
from typing import Any, Dict, Optional, Set

import httpx

from .event_bus import Event, EventBus
from ..llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

_IMPORTANCE_SYSTEM_PROMPT = """\
You are an email importance classifier. Evaluate the email and respond with ONLY a JSON object.

Rules for IMPORTANT emails (require immediate attention):
- OTP / verification codes
- Security alerts (login attempts, password resets, suspicious activity)
- Payment failures or billing issues
- Delivery problems (failed delivery, return to sender)
- Time-sensitive actions required (expiring offers that matter, deadlines)
- Personal urgent messages from real people

Rules for NOT IMPORTANT emails:
- Newsletters and digests
- Order confirmations and shipping updates (routine, no problems)
- Receipts for completed transactions
- Social media notifications
- Marketing and promotional emails
- Automated status updates that require no action

Respond with ONLY this JSON (no markdown, no extra text):
{"important": true/false, "reason": "brief reason", "summary": "one-line summary of the email"}
"""


class EmailEventHandler:
    """Handles email events by evaluating importance via LLM and sending callbacks.

    Args:
        llm_client: LLM client for importance evaluation
        event_bus: EventBus to subscribe to email events
        callback_url: URL to POST important email notifications to
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        event_bus: EventBus,
        callback_url: str,
    ):
        self._llm_client = llm_client
        self._event_bus = event_bus
        self._callback_url = callback_url
        self._processed_ids: Set[str] = set()

    async def start(self) -> None:
        """Subscribe to email events on the event bus."""
        await self._event_bus.subscribe("email:*", self.handle_email)
        logger.info("EmailEventHandler subscribed to email:* events")

    async def handle_email(self, event: Event) -> None:
        """Process an incoming email event.

        Evaluates importance via LLM, and if important, POSTs a callback.
        Skips duplicate message_ids.
        """
        data = event.data or {}
        message_id = data.get("message_id", "")

        # Duplicate prevention
        if message_id and message_id in self._processed_ids:
            logger.debug(f"Skipping duplicate email: {message_id}")
            return
        if message_id:
            self._processed_ids.add(message_id)

        sender = data.get("sender", "")
        subject = data.get("subject", "")
        snippet = data.get("snippet", "")

        # Evaluate importance via LLM
        evaluation = await self._evaluate_importance(sender, subject, snippet)
        if evaluation is None:
            logger.warning(f"LLM evaluation failed for email {message_id}")
            return

        if not evaluation.get("important", False):
            logger.debug(f"Email not important: {subject} (reason: {evaluation.get('reason', 'N/A')})")
            return

        # Important email — send callback
        logger.info(f"Important email detected: {subject} — {evaluation.get('reason', '')}")
        await self._send_callback(
            tenant_id=event.tenant_id,
            summary=evaluation.get("summary", subject),
            sender=sender,
            subject=subject,
            message_id=message_id,
            reason=evaluation.get("reason", ""),
        )

    async def _evaluate_importance(
        self, sender: str, subject: str, snippet: str
    ) -> Optional[Dict[str, Any]]:
        """Call the LLM to evaluate email importance.

        Returns:
            Dict with keys: important (bool), reason (str), summary (str)
            None if the LLM call fails.
        """
        user_message = (
            f"Sender: {sender}\n"
            f"Subject: {subject}\n"
            f"Preview: {snippet}"
        )

        try:
            response = await self._llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": _IMPORTANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                config={"temperature": 0.0, "max_tokens": 256},
            )
            content = response.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Email importance evaluation failed: {e}")
            return None

    async def _send_callback(
        self,
        tenant_id: str,
        summary: str,
        sender: str,
        subject: str,
        message_id: str,
        reason: str,
    ) -> None:
        """POST important email notification to the callback URL."""
        payload = {
            "tenant_id": tenant_id,
            "message": summary,
            "priority": "urgent",
            "category": "email_alert",
            "metadata": {
                "subject": subject,
                "sender": sender,
                "message_id": message_id,
                "reason": reason,
            },
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self._callback_url,
                    json=payload,
                    timeout=15.0,
                )
                resp.raise_for_status()
                logger.info(f"Email callback sent for message {message_id}")
        except Exception as e:
            logger.error(f"Email callback failed for {message_id}: {e}")

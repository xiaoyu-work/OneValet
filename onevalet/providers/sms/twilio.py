"""
Twilio SMS Provider - Sends SMS via Twilio Messaging Service
"""

import logging
from twilio.rest import Client as TwilioClient

from .base import BaseSMSProvider

logger = logging.getLogger(__name__)


class TwilioProvider(BaseSMSProvider):
    """
    Twilio SMS provider using Messaging Service.

    Requires configuration:
    - account_sid: Twilio account SID
    - auth_token: Twilio auth token
    - messaging_service_sid: Twilio Messaging Service SID

    Optional:
    - from_number: Specify which number from the pool to use
    """

    def __init__(self, account_sid: str, auth_token: str, messaging_service_sid: str, from_number: str = ""):
        super().__init__()
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.messaging_service_sid = messaging_service_sid
        self.from_number = from_number

        if self.is_enabled():
            if self.from_number:
                logger.info(f"Twilio SMS Provider initialized (Messaging Service: {self.messaging_service_sid}, From: {self.from_number})")
            else:
                logger.info(f"Twilio SMS Provider initialized (Messaging Service: {self.messaging_service_sid}, auto-select number)")
        else:
            logger.warning("Twilio SMS Provider disabled - missing configuration")

    def is_enabled(self) -> bool:
        """Check if Twilio is configured."""
        return all([self.account_sid, self.auth_token, self.messaging_service_sid])

    async def send_sms(self, to: str, body: str) -> bool:
        """Send SMS via Twilio Messaging Service."""
        if not self.is_enabled():
            logger.warning("Twilio not configured - cannot send SMS")
            return False

        to = self.normalize_phone_number(to)
        logger.info(f"[Twilio] Sending SMS to {to}: {body[:200]}...")

        try:
            client = TwilioClient(self.account_sid, self.auth_token)

            create_params = {
                "messaging_service_sid": self.messaging_service_sid,
                "to": to,
                "body": body,
            }

            if self.from_number:
                create_params["from_"] = self.from_number
                logger.info(f"[Twilio] Using Messaging Service with specific number: {self.from_number}")
            else:
                logger.info("[Twilio] Using Messaging Service (auto-select from pool)")

            message = client.messages.create(**create_params)
            logger.info(f"[Twilio] SMS sent via Messaging Service - SID: {message.sid}")
            return True

        except Exception as e:
            logger.error(f"[Twilio] Failed to send SMS: {e}", exc_info=True)
            return False

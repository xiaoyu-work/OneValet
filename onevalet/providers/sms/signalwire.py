"""
SignalWire SMS Provider - Sends SMS via SignalWire API
"""

import logging
from signalwire.rest import Client as SignalWireClient

from .base import BaseSMSProvider

logger = logging.getLogger(__name__)


class SignalWireProvider(BaseSMSProvider):
    """
    SignalWire SMS provider.

    Requires configuration:
    - project_id: SignalWire project ID
    - api_token: SignalWire API token
    - space_url: SignalWire space URL
    - from_number: From phone number (E.164 format)
    """

    def __init__(self, project_id: str, api_token: str, space_url: str, from_number: str):
        super().__init__()
        self.project_id = project_id
        self.api_token = api_token
        self.space_url = space_url
        self.from_number = from_number

        if self.is_enabled():
            logger.info("SignalWire SMS Provider initialized")
        else:
            logger.warning("SignalWire SMS Provider disabled - missing configuration")

    def is_enabled(self) -> bool:
        """Check if SignalWire is configured."""
        return all([self.project_id, self.api_token, self.space_url, self.from_number])

    async def send_sms(self, to: str, body: str) -> bool:
        """Send SMS via SignalWire."""
        if not self.is_enabled():
            logger.warning("SignalWire not configured - cannot send SMS")
            return False

        to = self.normalize_phone_number(to)
        logger.info(f"[SignalWire] Sending SMS to {to}: {body[:200]}...")

        try:
            client = SignalWireClient(
                self.project_id,
                self.api_token,
                signalwire_space_url=self.space_url,
            )

            message = client.messages.create(
                from_=self.from_number,
                to=to,
                body=body,
            )

            logger.info(f"[SignalWire] SMS sent successfully - SID: {message.sid}")
            return True

        except Exception as e:
            logger.error(f"[SignalWire] Failed to send SMS: {e}", exc_info=True)
            return False

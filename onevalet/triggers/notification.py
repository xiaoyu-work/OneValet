"""OneValet Notification Channels — deliver trigger results to users."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SMSNotification:
    """Send notifications via SMS (Twilio/SignalWire).

    Args:
        account_sid: Twilio/SignalWire account SID
        auth_token: Auth token
        from_number: Sender phone number
        provider: "twilio" or "signalwire" (default "twilio")
        signalwire_space: SignalWire space URL (required for signalwire)
    """

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        provider: str = "twilio",
        signalwire_space: Optional[str] = None,
        phone_resolver: Optional[Any] = None,
    ):
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._from_number = from_number
        self._provider = provider
        self._signalwire_space = signalwire_space
        self._phone_resolver = phone_resolver  # callable: async (tenant_id) -> phone_number
        self._client = None

    async def send(self, tenant_id: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send SMS to user."""
        to_number = None
        if self._phone_resolver:
            to_number = await self._phone_resolver(tenant_id)
        if not to_number:
            logger.warning(f"No phone number for tenant {tenant_id}, skipping SMS")
            return False

        try:
            if self._provider == "twilio":
                from twilio.rest import Client
                if not self._client:
                    self._client = Client(self._account_sid, self._auth_token)
                self._client.messages.create(
                    body=message,
                    from_=self._from_number,
                    to=to_number,
                )
            elif self._provider == "signalwire":
                from signalwire.rest import Client as SWClient
                if not self._client:
                    self._client = SWClient(
                        self._account_sid,
                        self._auth_token,
                        signalwire_space_url=self._signalwire_space,
                    )
                self._client.messages.create(
                    body=message,
                    from_=self._from_number,
                    to=to_number,
                )
            logger.info(f"SMS sent to {tenant_id}")
            return True
        except Exception as e:
            logger.error(f"SMS send failed for {tenant_id}: {e}")
            return False


class PushNotification:
    """Send push notifications.

    Args:
        push_sender: Async callable that sends the push.
            Signature: async (tenant_id, title, body, data) -> bool
    """

    def __init__(self, push_sender: Any = None):
        self._push_sender = push_sender

    async def send(self, tenant_id: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send push notification to user."""
        if not self._push_sender:
            logger.debug(f"No push_sender configured, skipping push for {tenant_id}")
            return False

        try:
            meta = metadata or {}
            title = meta.get("task_name", "OneValet")
            return await self._push_sender(tenant_id, title, message, meta)
        except Exception as e:
            logger.error(f"Push notification failed for {tenant_id}: {e}")
            return False

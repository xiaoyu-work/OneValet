"""
SMS Provider Factory - Creates SMS provider instances based on configuration
"""

import logging
from typing import Optional

from .base import BaseSMSProvider

logger = logging.getLogger(__name__)


class SMSProviderFactory:
    """
    Factory for creating SMS providers.

    Supported providers:
    - signalwire: SignalWire SMS service
    - twilio: Twilio SMS service
    """

    @staticmethod
    def create_provider(provider_type: str, **config) -> Optional[BaseSMSProvider]:
        """
        Create SMS provider instance.

        Args:
            provider_type: Type of provider ("signalwire", "twilio")
            **config: Provider-specific configuration

        Returns:
            SMS provider instance, or None if provider type not supported

        Examples:
            provider = SMSProviderFactory.create_provider(
                "twilio",
                account_sid="xxx",
                auth_token="xxx",
                messaging_service_sid="xxx",
                from_number="+1xxx",
            )

            provider = SMSProviderFactory.create_provider(
                "signalwire",
                project_id="xxx",
                api_token="xxx",
                space_url="xxx.signalwire.com",
                from_number="+1xxx",
            )
        """
        provider_type = provider_type.lower()

        if provider_type == "signalwire":
            from .signalwire import SignalWireProvider
            return SignalWireProvider(
                project_id=config.get("project_id", ""),
                api_token=config.get("api_token", ""),
                space_url=config.get("space_url", ""),
                from_number=config.get("from_number", ""),
            )

        elif provider_type == "twilio":
            from .twilio import TwilioProvider
            return TwilioProvider(
                account_sid=config.get("account_sid", ""),
                auth_token=config.get("auth_token", ""),
                messaging_service_sid=config.get("messaging_service_sid", ""),
                from_number=config.get("from_number", ""),
            )

        else:
            logger.error(f"Unknown SMS provider type: {provider_type}")
            return None

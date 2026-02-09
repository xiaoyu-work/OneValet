"""
Package Manager - Extract package tracking info from emails using LLM

NOTE: This module depends on an LLM being available.
The extract_package_from_email function is designed to be called by agents,
not by providers directly.
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def extract_package_from_email(
    email_data: dict,
    llm=None,
) -> Optional[Dict[str, Any]]:
    """
    Extract package tracking information from email using LLM.

    Args:
        email_data: Email data with sender, subject, snippet
        llm: LLM instance with chat_completion method. If None,
             attempts to get one from the registry.

    Returns:
        Package info dict or None if not a package email
    """
    sender = email_data.get("sender", "")
    subject = email_data.get("subject", "")
    snippet = email_data.get("snippet", "")

    prompt = f"""Analyze this email and determine if it contains package/shipment tracking information.

Email from: {sender}
Subject: {subject}
Content preview: {snippet}

If this email is about a package shipment (shipping confirmation, delivery update, etc.), extract:
1. The tracking number (the alphanumeric code used to track the package)
2. The carrier name (ups, fedex, usps, dhl, amazon, ontrac, lasership, etc.)
3. A brief description of what's being shipped

IMPORTANT: Return ONLY valid JSON, no other text.

If it IS a package email: {{"is_package_email": true, "tracking_number": "1Z999AA10123456784", "carrier": "ups", "description": "iPhone case"}}

If it is NOT a package email: {{"is_package_email": false}}
"""

    try:
        if llm is None:
            logger.error("No LLM instance provided for package extraction")
            return None

        response = await llm.chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content.strip() if response.content else ""
        logger.info(f"LLM package extraction response: {content[:200]}")

        if not content:
            logger.warning("LLM returned empty content for package extraction")
            return None

        result = json.loads(content)

        if result.get("is_package_email") and result.get("tracking_number"):
            package_info = {
                "tracking_number": result["tracking_number"],
                "carrier": result.get("carrier") or "unknown",
                "description": result.get("description") or "Package",
                "source": "email",
                "email_sender": sender,
                "email_subject": subject[:100],
            }
            logger.info(f"LLM extracted package: {package_info['tracking_number']} ({package_info['carrier']})")
            return package_info

    except Exception as e:
        logger.error(f"Failed to extract package info via LLM: {e}")

    return None

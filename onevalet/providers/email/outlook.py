"""
Outlook Provider - Microsoft Outlook/Office 365 API implementation

Uses Microsoft Graph API for email operations.
Requires OAuth scopes: Mail.Send, Mail.ReadWrite, Mail.Read
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta, timezone

import httpx

from .base import BaseEmailProvider

logger = logging.getLogger(__name__)


class OutlookProvider(BaseEmailProvider):
    """Outlook/Microsoft 365 email provider using Microsoft Graph API."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)
        self.api_base_url = "https://graph.microsoft.com/v1.0"

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Send email via Microsoft Graph API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            message_payload: Dict[str, Any] = {
                "message": {
                    "subject": subject,
                    "body": {"contentType": "Text", "content": body},
                    "toRecipients": [{"emailAddress": {"address": to}}],
                }
            }

            if cc:
                message_payload["message"]["ccRecipients"] = [
                    {"emailAddress": {"address": addr}} for addr in cc
                ]
            if bcc:
                message_payload["message"]["bccRecipients"] = [
                    {"emailAddress": {"address": addr}} for addr in bcc
                ]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/me/sendMail",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json",
                    },
                    json=message_payload,
                    timeout=30.0,
                )

                if response.status_code == 202:
                    logger.info("Outlook sent email successfully")
                    return {"success": True, "message_id": "sent"}
                else:
                    logger.error(f"Outlook send failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Outlook send error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def search_emails(
        self,
        query: Optional[str] = None,
        sender: Optional[str] = None,
        date_range: Optional[str] = None,
        unread_only: bool = False,
        max_results: int = 20,
        days_back: Optional[int] = None,
        include_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search emails via Microsoft Graph API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            filters = []
            if unread_only:
                filters.append("isRead eq false")
            if sender:
                filters.append(f"from/emailAddress/address eq '{sender}'")
            if days_back:
                after_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
                filters.append(f"receivedDateTime ge {after_date}")

            filter_str = " and ".join(filters) if filters else None

            params: Dict[str, Any] = {
                "$top": max_results,
                "$orderby": "receivedDateTime DESC",
                "$select": "id,subject,from,receivedDateTime,isRead,bodyPreview",
            }
            if filter_str:
                params["$filter"] = filter_str
            if query:
                params["$search"] = f'"{query}"'

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/me/messages",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Prefer": 'outlook.body-content-type="text"',
                    },
                    params=params,
                    timeout=30.0,
                )

                if response.status_code != 200:
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

                result = response.json()
                messages = result.get("value", [])

                email_list = []
                for msg in messages:
                    email_list.append({
                        "message_id": msg["id"],
                        "sender": msg.get("from", {}).get("emailAddress", {}).get("address", "Unknown"),
                        "subject": msg.get("subject", "(No subject)"),
                        "date": msg.get("receivedDateTime", "Unknown"),
                        "unread": not msg.get("isRead", True),
                        "snippet": msg.get("bodyPreview", "")[:200],
                    })

                logger.info(f"Outlook search found {len(email_list)} emails")
                return {"success": True, "data": email_list, "count": len(email_list)}

        except Exception as e:
            logger.error(f"Outlook search error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_emails(
        self,
        message_ids: List[str],
        permanent: bool = False,
    ) -> Dict[str, Any]:
        """Delete emails via Microsoft Graph API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            deleted_count = 0
            async with httpx.AsyncClient() as client:
                for msg_id in message_ids:
                    response = await client.delete(
                        f"{self.api_base_url}/me/messages/{msg_id}",
                        headers={"Authorization": f"Bearer {self.access_token}"},
                        timeout=30.0,
                    )
                    if response.status_code == 204:
                        deleted_count += 1

            logger.info(f"Outlook deleted {deleted_count}/{len(message_ids)} emails")
            return {"success": True, "deleted_count": deleted_count}

        except Exception as e:
            logger.error(f"Outlook delete error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh Microsoft OAuth token."""
        try:
            import os

            client_id = os.getenv("MICROSOFT_CLIENT_ID")
            client_secret = os.getenv("MICROSOFT_CLIENT_SECRET")
            tenant_id = os.getenv("MICROSOFT_TENANT_ID", "common")

            if not client_id or not client_secret:
                return {"success": False, "error": "Microsoft OAuth credentials not configured"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": self.refresh_token,
                        "grant_type": "refresh_token",
                        "scope": "https://graph.microsoft.com/Mail.ReadWrite https://graph.microsoft.com/Mail.Send",
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    expires_in = data.get("expires_in", 3600)
                    token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                    logger.info(f"Outlook token refreshed for {self.account_name}")
                    return {
                        "success": True,
                        "access_token": data["access_token"],
                        "expires_in": expires_in,
                        "token_expiry": token_expiry,
                    }
                else:
                    logger.error(f"Outlook token refresh failed: {response.text}")
                    return {"success": False, "error": f"Token refresh failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Outlook token refresh error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def create_subscription(
        self,
        webhook_url: str,
        client_state: str,
        expiration_minutes: int = 4230,
    ) -> Dict[str, Any]:
        """Create Outlook webhook subscription for new emails."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            expiration = (datetime.utcnow() + timedelta(minutes=expiration_minutes)).isoformat() + "Z"

            payload = {
                "changeType": "created",
                "notificationUrl": webhook_url,
                "resource": "me/mailFolders('Inbox')/messages",
                "expirationDateTime": expiration,
                "clientState": client_state,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/subscriptions",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code == 201:
                    data = response.json()
                    logger.info(f"Outlook subscription created for {self.account_name}")
                    return {
                        "success": True,
                        "subscription_id": data.get("id"),
                        "expiration": data.get("expirationDateTime"),
                    }
                else:
                    logger.error(f"Outlook subscription failed: {response.text}")
                    return {"success": False, "error": f"Subscription failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Outlook create subscription error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def renew_subscription(
        self,
        subscription_id: str,
        expiration_minutes: int = 4230,
    ) -> Dict[str, Any]:
        """Renew Outlook webhook subscription."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            expiration = (datetime.utcnow() + timedelta(minutes=expiration_minutes)).isoformat() + "Z"

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{self.api_base_url}/subscriptions/{subscription_id}",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json",
                    },
                    json={"expirationDateTime": expiration},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info("Outlook subscription renewed")
                    return {"success": True, "expiration": data.get("expirationDateTime")}
                else:
                    return {"success": False, "error": f"Renew failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Outlook renew subscription error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """Delete Outlook webhook subscription."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.api_base_url}/subscriptions/{subscription_id}",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    timeout=30.0,
                )

                if response.status_code == 204:
                    logger.info("Outlook subscription deleted")
                    return {"success": True}
                else:
                    return {"success": False, "error": f"Delete failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Outlook delete subscription error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_message_details(self, message_id: str) -> Dict[str, Any]:
        """Get full message details by ID."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/me/messages/{message_id}",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Prefer": 'outlook.body-content-type="text"',
                    },
                    params={"$select": "id,subject,from,receivedDateTime,isRead,bodyPreview"},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    msg = response.json()
                    return {
                        "success": True,
                        "data": {
                            "message_id": msg["id"],
                            "sender": msg.get("from", {}).get("emailAddress", {}).get("address", "Unknown"),
                            "subject": msg.get("subject", "(No subject)"),
                            "date": msg.get("receivedDateTime", "Unknown"),
                            "snippet": msg.get("bodyPreview", "")[:200],
                            "unread": not msg.get("isRead", True),
                        },
                    }
                else:
                    return {"success": False, "error": f"Get message failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Outlook get message error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

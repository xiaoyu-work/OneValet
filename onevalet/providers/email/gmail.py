"""
Gmail Provider - Gmail API implementation

Uses Google Gmail API for email operations.
Requires OAuth scopes: gmail.send, gmail.modify, gmail.readonly
"""

import base64
import logging
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta, timezone

import httpx

from .base import BaseEmailProvider

logger = logging.getLogger(__name__)


class GmailProvider(BaseEmailProvider):
    """Gmail email provider implementation using Gmail API v1."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)
        self.api_base_url = "https://gmail.googleapis.com/gmail/v1"

    async def send_email(
        self,
        to: str | List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Send email via Gmail API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            message = MIMEMultipart()
            message["To"] = ", ".join(to) if isinstance(to, list) else to
            message["From"] = self.email
            message["Subject"] = subject

            if cc:
                message["Cc"] = ", ".join(cc)
            if bcc:
                message["Bcc"] = ", ".join(bcc)

            message.attach(MIMEText(body, "plain"))
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/users/me/messages/send",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json",
                    },
                    json={"raw": raw_message},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    message_id = result.get("id")
                    logger.info(f"Gmail sent: {message_id}")
                    return {"success": True, "message_id": message_id}
                else:
                    logger.error(f"Gmail send failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Gmail API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Gmail send error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def reply_email(
        self,
        original_message_id: str,
        body: str,
        reply_all: bool = False,
    ) -> Dict[str, Any]:
        """Reply to an email via Gmail API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                # Get original message details
                response = await client.get(
                    f"{self.api_base_url}/users/me/messages/{original_message_id}",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    params={"format": "metadata", "metadataHeaders": ["From", "To", "Cc", "Subject", "Message-ID"]},
                    timeout=30.0,
                )

                if response.status_code != 200:
                    return {"success": False, "error": f"Failed to get original message: {response.status_code}"}

                msg_data = response.json()
                thread_id = msg_data.get("threadId")
                headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}

                original_from = headers.get("From", "")
                original_to = headers.get("To", "")
                original_cc = headers.get("Cc", "")
                original_subject = headers.get("Subject", "")
                original_message_id_header = headers.get("Message-ID", "")

                def extract_email(s):
                    match = re.search(r'<([^>]+)>', s)
                    return match.group(1) if match else s.strip()

                reply_to = extract_email(original_from)

                reply_subject = original_subject
                if not reply_subject.lower().startswith("re:"):
                    reply_subject = f"Re: {reply_subject}"

                message = MIMEMultipart()
                message["To"] = reply_to
                message["From"] = self.email
                message["Subject"] = reply_subject

                if reply_all:
                    all_recipients = set()
                    if original_to:
                        for addr in original_to.split(","):
                            email = extract_email(addr.strip())
                            if email and email.lower() != self.email.lower():
                                all_recipients.add(email)
                    if original_cc:
                        for addr in original_cc.split(","):
                            email = extract_email(addr.strip())
                            if email and email.lower() != self.email.lower():
                                all_recipients.add(email)
                    if all_recipients:
                        message["Cc"] = ", ".join(all_recipients)

                if original_message_id_header:
                    message["In-Reply-To"] = original_message_id_header
                    message["References"] = original_message_id_header

                message.attach(MIMEText(body, "plain"))
                raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

                response = await client.post(
                    f"{self.api_base_url}/users/me/messages/send",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json",
                    },
                    json={"raw": raw_message, "threadId": thread_id},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    message_id = result.get("id")
                    logger.info(f"Gmail reply sent: {message_id}")
                    return {"success": True, "message_id": message_id, "replied_to": reply_to}
                else:
                    logger.error(f"Gmail reply failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Gmail API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Gmail reply error: {e}", exc_info=True)
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
        """Search emails via Gmail API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            query_parts = []
            if unread_only:
                query_parts.append("is:unread")
            if sender:
                query_parts.append(f"from:{sender}")
            if query:
                query_parts.append(query)
            if days_back:
                after_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
                query_parts.append(f"after:{after_date}")
            if include_categories:
                for category in include_categories:
                    query_parts.append(f"category:{category}")

            gmail_query = " ".join(query_parts) if query_parts else ""

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/users/me/messages",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    params={"q": gmail_query, "maxResults": max_results},
                    timeout=30.0,
                )

                # Handle 401 - token may be expired
                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/users/me/messages",
                            headers={"Authorization": f"Bearer {self.access_token}"},
                            params={"q": gmail_query, "maxResults": max_results},
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Gmail API error: {response.status_code}"}

                result = response.json()
                messages = result.get("messages", [])

                email_list = []
                for msg in messages[:max_results]:
                    msg_id = msg["id"]
                    detail_response = await client.get(
                        f"{self.api_base_url}/users/me/messages/{msg_id}",
                        headers={"Authorization": f"Bearer {self.access_token}"},
                        params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
                        timeout=30.0,
                    )
                    if detail_response.status_code == 200:
                        msg_data = detail_response.json()
                        hdrs = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}
                        email_list.append({
                            "message_id": msg_id,
                            "sender": hdrs.get("From", "Unknown"),
                            "subject": hdrs.get("Subject", "(No subject)"),
                            "date": hdrs.get("Date", "Unknown"),
                            "unread": "UNREAD" in msg_data.get("labelIds", []),
                            "snippet": msg_data.get("snippet", ""),
                        })

                logger.info(f"Gmail search found {len(email_list)} emails")
                return {"success": True, "data": email_list, "count": len(email_list)}

        except Exception as e:
            logger.error(f"Gmail search error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_emails(
        self,
        message_ids: List[str],
        permanent: bool = False,
    ) -> Dict[str, Any]:
        """Delete emails via Gmail API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            deleted_count = 0
            async with httpx.AsyncClient() as client:
                for msg_id in message_ids:
                    if permanent:
                        response = await client.delete(
                            f"{self.api_base_url}/users/me/messages/{msg_id}",
                            headers={"Authorization": f"Bearer {self.access_token}"},
                            timeout=30.0,
                        )
                    else:
                        response = await client.post(
                            f"{self.api_base_url}/users/me/messages/{msg_id}/trash",
                            headers={"Authorization": f"Bearer {self.access_token}"},
                            timeout=30.0,
                        )
                    if response.status_code in [200, 204]:
                        deleted_count += 1

            logger.info(f"Gmail deleted {deleted_count}/{len(message_ids)} emails")
            return {"success": True, "deleted_count": deleted_count}

        except Exception as e:
            logger.error(f"Gmail delete error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def archive_emails(self, message_ids: List[str]) -> Dict[str, Any]:
        """Archive emails via Gmail API (remove INBOX label)."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            archived_count = 0
            async with httpx.AsyncClient() as client:
                for msg_id in message_ids:
                    response = await client.post(
                        f"{self.api_base_url}/users/me/messages/{msg_id}/modify",
                        headers={"Authorization": f"Bearer {self.access_token}"},
                        json={"removeLabelIds": ["INBOX"]},
                        timeout=30.0,
                    )
                    if response.status_code == 200:
                        archived_count += 1

            logger.info(f"Gmail archived {archived_count}/{len(message_ids)} emails")
            return {"success": True, "archived_count": archived_count}

        except Exception as e:
            logger.error(f"Gmail archive error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def mark_as_read(self, message_ids: List[str]) -> Dict[str, Any]:
        """Mark emails as read via Gmail API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            marked_count = 0
            async with httpx.AsyncClient() as client:
                for msg_id in message_ids:
                    response = await client.post(
                        f"{self.api_base_url}/users/me/messages/{msg_id}/modify",
                        headers={"Authorization": f"Bearer {self.access_token}"},
                        json={"removeLabelIds": ["UNREAD"]},
                        timeout=30.0,
                    )
                    if response.status_code == 200:
                        marked_count += 1

            logger.info(f"Gmail marked {marked_count}/{len(message_ids)} emails as read")
            return {"success": True, "marked_count": marked_count}

        except Exception as e:
            logger.error(f"Gmail mark as read error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh Gmail OAuth token."""
        try:
            import os

            client_id = os.getenv("GOOGLE_CLIENT_ID")
            client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

            if not client_id or not client_secret:
                return {"success": False, "error": "Google OAuth credentials not configured"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": self.refresh_token,
                        "grant_type": "refresh_token",
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    expires_in = data.get("expires_in", 3600)
                    token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                    logger.info(f"Gmail token refreshed for {self.account_name}")
                    return {
                        "success": True,
                        "access_token": data["access_token"],
                        "expires_in": expires_in,
                        "token_expiry": token_expiry,
                    }
                else:
                    logger.error(f"Gmail token refresh failed: {response.text}")
                    return {"success": False, "error": f"Token refresh failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Gmail token refresh error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def setup_watch(
        self,
        topic_name: str,
        label_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Set up Gmail push notifications using Watch API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            payload = {
                "topicName": topic_name,
                "labelIds": label_ids or ["INBOX"],
                "labelFilterBehavior": "INCLUDE",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/users/me/watch",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Gmail watch set up for {self.account_name}")
                    return {
                        "success": True,
                        "history_id": data.get("historyId"),
                        "expiration": int(data.get("expiration")),
                    }
                else:
                    logger.error(f"Gmail watch setup failed: {response.text}")
                    return {"success": False, "error": f"Watch setup failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Gmail watch setup error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def stop_watch(self) -> Dict[str, Any]:
        """Stop Gmail push notifications."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/users/me/stop",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    timeout=30.0,
                )

                if response.status_code == 204:
                    logger.info(f"Gmail watch stopped for {self.account_name}")
                    return {"success": True}
                else:
                    return {"success": False, "error": f"Stop watch failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Gmail stop watch error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_history(
        self,
        start_history_id: str,
        history_types: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """Get email history changes (for processing webhook notifications)."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            params: Dict[str, Any] = {
                "startHistoryId": start_history_id,
                "maxResults": max_results,
            }
            params["historyTypes"] = history_types or ["messageAdded"]

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/users/me/history",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    params=params,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    history_records = data.get("history", [])
                    new_history_id = data.get("historyId")

                    new_messages = []
                    for record in history_records:
                        if "messagesAdded" in record:
                            for added in record["messagesAdded"]:
                                message = added.get("message", {})
                                if "INBOX" in message.get("labelIds", []):
                                    new_messages.append({
                                        "message_id": message.get("id"),
                                        "thread_id": message.get("threadId"),
                                    })

                    logger.info(f"Gmail history: {len(new_messages)} new messages")
                    return {
                        "success": True,
                        "data": {
                            "history": history_records,
                            "historyId": new_history_id,
                            "messages": new_messages,
                        },
                    }
                elif response.status_code == 404:
                    logger.warning("Gmail history not found (historyId too old)")
                    return {"success": False, "error": "History ID too old, full sync required"}
                else:
                    logger.error(f"Gmail history fetch failed: {response.text}")
                    return {"success": False, "error": f"History fetch failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Gmail get history error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_message_details(self, message_id: str) -> Dict[str, Any]:
        """Get full message details by ID."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/users/me/messages/{message_id}",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    msg_data = response.json()
                    hdrs = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}
                    return {
                        "success": True,
                        "data": {
                            "message_id": message_id,
                            "sender": hdrs.get("From", "Unknown"),
                            "subject": hdrs.get("Subject", "(No subject)"),
                            "date": hdrs.get("Date", "Unknown"),
                            "snippet": msg_data.get("snippet", ""),
                            "unread": "UNREAD" in msg_data.get("labelIds", []),
                        },
                    }
                else:
                    return {"success": False, "error": f"Get message failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Gmail get message error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

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
from ..http_mixin import OAuthHTTPMixin

logger = logging.getLogger(__name__)


class GmailProvider(BaseEmailProvider, OAuthHTTPMixin):
    """Gmail email provider implementation using Gmail API v1."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)
        self.api_base_url = "https://gmail.googleapis.com/gmail/v1"

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

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

            response = await self._oauth_request(
                "POST",
                f"{self.api_base_url}/users/me/messages/send",
                json={"raw": raw_message},
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
            # Get original message details
            response = await self._oauth_request(
                "GET",
                f"{self.api_base_url}/users/me/messages/{original_message_id}",
                params={"format": "metadata", "metadataHeaders": ["From", "To", "Cc", "Subject", "Message-ID"]},
            )

            if response.status_code != 200:
                return {"success": False, "error": f"Failed to get original message: {response.status_code}"}

            msg_data = response.json()
            thread_id = msg_data.get("threadId")
            hdrs = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}

            original_from = hdrs.get("From", "")
            original_to = hdrs.get("To", "")
            original_cc = hdrs.get("Cc", "")
            original_subject = hdrs.get("Subject", "")
            original_message_id_header = hdrs.get("Message-ID", "")

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

            response = await self._oauth_request(
                "POST",
                f"{self.api_base_url}/users/me/messages/send",
                json={"raw": raw_message, "threadId": thread_id},
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

            response = await self._oauth_request(
                "GET",
                f"{self.api_base_url}/users/me/messages",
                params={"q": gmail_query, "maxResults": max_results},
            )

            if response.status_code != 200:
                return {"success": False, "error": f"Gmail API error: {response.status_code}"}

            result = response.json()
            messages = result.get("messages", [])

            email_list = []
            for msg in messages[:max_results]:
                msg_id = msg["id"]
                detail_response = await self._oauth_request(
                    "GET",
                    f"{self.api_base_url}/users/me/messages/{msg_id}",
                    params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
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
            deleted_count = 0
            for msg_id in message_ids:
                if permanent:
                    response = await self._oauth_request(
                        "DELETE",
                        f"{self.api_base_url}/users/me/messages/{msg_id}",
                    )
                else:
                    response = await self._oauth_request(
                        "POST",
                        f"{self.api_base_url}/users/me/messages/{msg_id}/trash",
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
            archived_count = 0
            for msg_id in message_ids:
                response = await self._oauth_request(
                    "POST",
                    f"{self.api_base_url}/users/me/messages/{msg_id}/modify",
                    json={"removeLabelIds": ["INBOX"]},
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
            marked_count = 0
            for msg_id in message_ids:
                response = await self._oauth_request(
                    "POST",
                    f"{self.api_base_url}/users/me/messages/{msg_id}/modify",
                    json={"removeLabelIds": ["UNREAD"]},
                )
                if response.status_code == 200:
                    marked_count += 1

            logger.info(f"Gmail marked {marked_count}/{len(message_ids)} emails as read")
            return {"success": True, "marked_count": marked_count}

        except Exception as e:
            logger.error(f"Gmail mark as read error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def setup_watch(
        self,
        topic_name: str,
        label_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Set up Gmail push notifications using Watch API."""
        try:
            payload = {
                "topicName": topic_name,
                "labelIds": label_ids or ["INBOX"],
                "labelFilterBehavior": "INCLUDE",
            }

            response = await self._oauth_request(
                "POST",
                f"{self.api_base_url}/users/me/watch",
                json=payload,
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
            response = await self._oauth_request(
                "POST",
                f"{self.api_base_url}/users/me/stop",
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
            params: Dict[str, Any] = {
                "startHistoryId": start_history_id,
                "maxResults": max_results,
            }
            params["historyTypes"] = history_types or ["messageAdded"]

            response = await self._oauth_request(
                "GET",
                f"{self.api_base_url}/users/me/history",
                params=params,
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
            response = await self._oauth_request(
                "GET",
                f"{self.api_base_url}/users/me/messages/{message_id}",
                params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
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

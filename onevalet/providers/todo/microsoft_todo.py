"""
Microsoft To Do Provider - Microsoft Graph API implementation

Uses Microsoft Graph API for To Do task operations.
Requires OAuth scope: https://graph.microsoft.com/Tasks.ReadWrite
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta, timezone

import httpx

from .base import BaseTodoProvider

logger = logging.getLogger(__name__)

# Mapping from Microsoft importance to our priority
_IMPORTANCE_TO_PRIORITY = {
    "low": "low",
    "normal": "medium",
    "high": "high",
}

# Mapping from our priority to Microsoft importance
_PRIORITY_TO_IMPORTANCE = {
    "low": "low",
    "medium": "normal",
    "high": "high",
    "urgent": "high",
}


class MicrosoftTodoProvider(BaseTodoProvider):
    """Microsoft To Do provider implementation using Graph API."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)
        self.api_base_url = "https://graph.microsoft.com/v1.0/me/todo"
        self._default_list_id: Optional[str] = None
        self._default_list_name: Optional[str] = None

    def _auth_headers(self) -> dict:
        """Return authorization headers for API requests."""
        return {"Authorization": f"Bearer {self.access_token}"}

    async def _get_default_list_id(self) -> Optional[str]:
        """Get default task list ID, caching for subsequent calls."""
        if self._default_list_id:
            return self._default_list_id

        result = await self.list_task_lists()
        if result.get("success") and result.get("data"):
            self._default_list_id = result["data"][0]["id"]
            self._default_list_name = result["data"][0]["name"]
            return self._default_list_id
        return None

    def _format_task(self, task: dict, list_id: str, list_name: str) -> dict:
        """Convert Microsoft Graph task to unified task format."""
        due_dt = task.get("dueDateTime", {})
        due = None
        if due_dt and due_dt.get("dateTime"):
            try:
                due = due_dt["dateTime"][:10]
            except (IndexError, TypeError):
                due = None

        return {
            "id": task["id"],
            "title": task.get("title", ""),
            "due": due,
            "priority": _IMPORTANCE_TO_PRIORITY.get(task.get("importance", "normal"), "medium"),
            "completed": task.get("status") == "completed",
            "description": task.get("body", {}).get("content", ""),
            "list_name": list_name,
            "list_id": list_id,
            "_provider": "microsoft_todo",
            "_account_name": self.account_name,
            "_account_email": self.email,
        }

    async def list_tasks(
        self,
        list_id: Optional[str] = None,
        completed: bool = False,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """List tasks from Microsoft To Do."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            if not list_id:
                list_id = await self._get_default_list_id()
                if not list_id:
                    return {"success": False, "error": "No task lists found"}

            params: Dict[str, Any] = {"$top": max_results}
            if completed:
                params["$filter"] = "status eq 'completed'"
            else:
                params["$filter"] = "status ne 'completed'"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/lists/{list_id}/tasks",
                    headers=self._auth_headers(),
                    params=params,
                    timeout=30.0,
                )

                # Handle 401 - token may be expired
                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/lists/{list_id}/tasks",
                            headers=self._auth_headers(),
                            params=params,
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

                data = response.json()
                tasks = data.get("value", [])

                # Resolve list name
                list_name = self._default_list_name if list_id == self._default_list_id else list_id
                formatted = [self._format_task(t, list_id, list_name) for t in tasks]

                logger.info(f"Microsoft To Do listed {len(formatted)} tasks")
                return {"success": True, "data": formatted}

        except Exception as e:
            logger.error(f"Microsoft To Do list tasks error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def search_tasks(
        self,
        query: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search tasks by keyword (client-side filtering)."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            if not list_id:
                list_id = await self._get_default_list_id()
                if not list_id:
                    return {"success": False, "error": "No task lists found"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/lists/{list_id}/tasks",
                    headers=self._auth_headers(),
                    params={"$top": 200},
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/lists/{list_id}/tasks",
                            headers=self._auth_headers(),
                            params={"$top": 200},
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

                data = response.json()
                tasks = data.get("value", [])

                list_name = self._default_list_name if list_id == self._default_list_id else list_id
                query_lower = query.lower()
                matched = []
                for t in tasks:
                    title = t.get("title", "").lower()
                    body = t.get("body", {}).get("content", "").lower()
                    if query_lower in title or query_lower in body:
                        matched.append(self._format_task(t, list_id, list_name))

                logger.info(f"Microsoft To Do search found {len(matched)} tasks for '{query}'")
                return {"success": True, "data": matched}

        except Exception as e:
            logger.error(f"Microsoft To Do search error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def create_task(
        self,
        title: str,
        due: Optional[str] = None,
        priority: Optional[str] = None,
        description: Optional[str] = None,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new task in Microsoft To Do."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            if not list_id:
                list_id = await self._get_default_list_id()
                if not list_id:
                    return {"success": False, "error": "No task lists found"}

            body: Dict[str, Any] = {"title": title}

            if description:
                body["body"] = {"content": description, "contentType": "text"}

            if due:
                body["dueDateTime"] = {
                    "dateTime": f"{due}T00:00:00",
                    "timeZone": "UTC",
                }

            if priority:
                importance = _PRIORITY_TO_IMPORTANCE.get(priority.lower(), "normal")
                body["importance"] = importance

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/lists/{list_id}/tasks",
                    headers={
                        **self._auth_headers(),
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=30.0,
                )

                if response.status_code == 201:
                    task = response.json()
                    list_name = self._default_list_name if list_id == self._default_list_id else list_id
                    formatted = self._format_task(task, list_id, list_name)
                    logger.info(f"Microsoft To Do task created: {task.get('id')}")
                    return {"success": True, "data": formatted}
                else:
                    logger.error(f"Microsoft To Do create failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Microsoft To Do create error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def complete_task(
        self,
        task_id: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark a task as completed in Microsoft To Do."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            if not list_id:
                list_id = await self._get_default_list_id()
                if not list_id:
                    return {"success": False, "error": "No task lists found"}

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{self.api_base_url}/lists/{list_id}/tasks/{task_id}",
                    headers={
                        **self._auth_headers(),
                        "Content-Type": "application/json",
                    },
                    json={"status": "completed"},
                    timeout=30.0,
                )

                if response.status_code == 200:
                    logger.info(f"Microsoft To Do task completed: {task_id}")
                    return {"success": True}
                else:
                    logger.error(f"Microsoft To Do complete failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Microsoft To Do complete error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def update_task(
        self,
        task_id: str,
        title: Optional[str] = None,
        due: Optional[str] = None,
        priority: Optional[str] = None,
        description: Optional[str] = None,
        completed: Optional[bool] = None,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing task in Microsoft To Do."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            if not list_id:
                list_id = await self._get_default_list_id()
                if not list_id:
                    return {"success": False, "error": "No task lists found"}

            body: Dict[str, Any] = {}

            if title is not None:
                body["title"] = title

            if description is not None:
                body["body"] = {"content": description, "contentType": "text"}

            if due is not None:
                body["dueDateTime"] = {
                    "dateTime": f"{due}T00:00:00",
                    "timeZone": "UTC",
                }

            if priority is not None:
                body["importance"] = _PRIORITY_TO_IMPORTANCE.get(priority.lower(), "normal")

            if completed is not None:
                body["status"] = "completed" if completed else "notStarted"

            if not body:
                return {"success": False, "error": "No fields to update"}

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{self.api_base_url}/lists/{list_id}/tasks/{task_id}",
                    headers={
                        **self._auth_headers(),
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    task = response.json()
                    list_name = self._default_list_name if list_id == self._default_list_id else list_id
                    formatted = self._format_task(task, list_id, list_name)
                    logger.info(f"Microsoft To Do task updated: {task_id}")
                    return {"success": True, "data": formatted}
                else:
                    logger.error(f"Microsoft To Do update failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Microsoft To Do update error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_task(
        self,
        task_id: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a task from Microsoft To Do."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            if not list_id:
                list_id = await self._get_default_list_id()
                if not list_id:
                    return {"success": False, "error": "No task lists found"}

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.api_base_url}/lists/{list_id}/tasks/{task_id}",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 204:
                    logger.info(f"Microsoft To Do task deleted: {task_id}")
                    return {"success": True}
                else:
                    logger.error(f"Microsoft To Do delete failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Microsoft To Do delete error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def list_task_lists(self) -> Dict[str, Any]:
        """List all task lists from Microsoft To Do."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/lists",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/lists",
                            headers=self._auth_headers(),
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

                data = response.json()
                lists = [
                    {"id": lst["id"], "name": lst.get("displayName", "")}
                    for lst in data.get("value", [])
                ]

                logger.info(f"Microsoft To Do found {len(lists)} task lists")
                return {"success": True, "data": lists}

        except Exception as e:
            logger.error(f"Microsoft To Do list task lists error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh Microsoft OAuth token."""
        try:
            client_id = os.getenv("MICROSOFT_CLIENT_ID")
            client_secret = os.getenv("MICROSOFT_CLIENT_SECRET")
            tenant = os.getenv("MICROSOFT_TENANT_ID", "common")

            if not client_id or not client_secret:
                return {"success": False, "error": "Microsoft OAuth credentials not configured"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": self.refresh_token,
                        "grant_type": "refresh_token",
                        "scope": "https://graph.microsoft.com/Tasks.ReadWrite offline_access",
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    expires_in = data.get("expires_in", 3600)
                    token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                    logger.info(f"Microsoft To Do token refreshed for {self.account_name}")
                    return {
                        "success": True,
                        "access_token": data["access_token"],
                        "expires_in": expires_in,
                        "token_expiry": token_expiry,
                    }
                else:
                    logger.error(f"Microsoft To Do token refresh failed: {response.text}")
                    return {"success": False, "error": f"Token refresh failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Microsoft To Do token refresh error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

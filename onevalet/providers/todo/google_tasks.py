"""
Google Tasks Provider - Google Tasks API v1 implementation

Uses Google Tasks API for task operations.
Requires OAuth scope: https://www.googleapis.com/auth/tasks
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

from .base import BaseTodoProvider
from ..http_mixin import OAuthHTTPMixin

logger = logging.getLogger(__name__)


class GoogleTasksProvider(BaseTodoProvider, OAuthHTTPMixin):
    """Google Tasks provider implementation using Tasks API v1."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)
        self.api_base_url = "https://tasks.googleapis.com/tasks/v1"

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def _format_task(self, task: dict, list_name: str, list_id: str) -> dict:
        """Format a Google Tasks API task into the unified task format."""
        due = None
        if task.get("due"):
            try:
                due = datetime.fromisoformat(task["due"].replace("Z", "+00:00")).strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                due = None

        return {
            "id": task["id"],
            "title": task.get("title", ""),
            "due": due,
            "priority": "medium",
            "completed": task.get("status") == "completed",
            "description": task.get("notes", ""),
            "list_name": list_name,
            "list_id": list_id,
            "_provider": "google_tasks",
            "_account_name": self.account_name,
            "_account_email": self.email,
        }

    async def list_tasks(
        self,
        list_id: Optional[str] = None,
        completed: bool = False,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """List tasks from a Google Tasks list."""
        try:
            tasklist = list_id or "@default"

            list_response = await self._oauth_request(
                "GET",
                f"{self.api_base_url}/users/@me/lists/{tasklist}",
            )
            list_name = "My Tasks"
            if list_response.status_code == 200:
                list_name = list_response.json().get("title", "My Tasks")

            params: Dict[str, Any] = {
                "maxResults": max_results,
                "showCompleted": str(completed).lower(),
                "showHidden": "true",
            }

            response = await self._oauth_request(
                "GET",
                f"{self.api_base_url}/lists/{tasklist}/tasks",
                params=params,
            )

            if response.status_code != 200:
                return {"success": False, "error": f"Google Tasks API error: {response.status_code}"}

            result = response.json()
            items = result.get("items", [])

            tasks = [self._format_task(task, list_name, tasklist) for task in items]
            logger.info(f"Google Tasks listed {len(tasks)} tasks from {list_name}")
            return {"success": True, "data": tasks}

        except Exception as e:
            logger.error(f"Google Tasks list error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def search_tasks(
        self,
        query: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search tasks by keyword across all lists."""
        try:
            query_lower = query.lower()

            lists_response = await self._oauth_request(
                "GET",
                f"{self.api_base_url}/users/@me/lists",
            )

            if lists_response.status_code != 200:
                return {"success": False, "error": f"Google Tasks API error: {lists_response.status_code}"}

            task_lists = lists_response.json().get("items", [])
            matching_tasks = []

            for tl in task_lists:
                tl_id = tl["id"]
                tl_name = tl.get("title", "")

                response = await self._oauth_request(
                    "GET",
                    f"{self.api_base_url}/lists/{tl_id}/tasks",
                    params={"showCompleted": "true", "showHidden": "true"},
                )

                if response.status_code != 200:
                    continue

                items = response.json().get("items", [])
                for task in items:
                    title = task.get("title", "").lower()
                    notes = task.get("notes", "").lower()
                    if query_lower in title or query_lower in notes:
                        matching_tasks.append(self._format_task(task, tl_name, tl_id))

            logger.info(f"Google Tasks search found {len(matching_tasks)} tasks for '{query}'")
            return {"success": True, "data": matching_tasks}

        except Exception as e:
            logger.error(f"Google Tasks search error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def create_task(
        self,
        title: str,
        due: Optional[str] = None,
        priority: Optional[str] = None,
        description: Optional[str] = None,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new task in Google Tasks."""
        try:
            tasklist = list_id or "@default"
            body: Dict[str, Any] = {"title": title}

            if description:
                body["notes"] = description

            if due:
                try:
                    due_date = datetime.strptime(due, "%Y-%m-%d")
                    body["due"] = due_date.strftime("%Y-%m-%dT00:00:00.000Z")
                except ValueError:
                    body["due"] = due

            response = await self._oauth_request(
                "POST",
                f"{self.api_base_url}/lists/{tasklist}/tasks",
                json=body,
            )

            if response.status_code == 200:
                task = response.json()
                logger.info(f"Google Tasks created: {task.get('id')}")
                return {"success": True, "data": self._format_task(task, "", tasklist)}
            else:
                logger.error(f"Google Tasks create failed: {response.status_code} - {response.text}")
                return {"success": False, "error": f"Google Tasks API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Google Tasks create error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def complete_task(
        self,
        task_id: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark a task as completed in Google Tasks."""
        try:
            tasklist = list_id or "@default"

            response = await self._oauth_request(
                "PATCH",
                f"{self.api_base_url}/lists/{tasklist}/tasks/{task_id}",
                json={"status": "completed"},
            )

            if response.status_code == 200:
                logger.info(f"Google Tasks completed: {task_id}")
                return {"success": True}
            else:
                logger.error(f"Google Tasks complete failed: {response.status_code} - {response.text}")
                return {"success": False, "error": f"Google Tasks API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Google Tasks complete error: {e}", exc_info=True)
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
        """Update an existing task in Google Tasks."""
        try:
            tasklist = list_id or "@default"
            body: Dict[str, Any] = {}

            if title is not None:
                body["title"] = title
            if description is not None:
                body["notes"] = description
            if completed is not None:
                body["status"] = "completed" if completed else "needsAction"
            if due is not None:
                try:
                    due_date = datetime.strptime(due, "%Y-%m-%d")
                    body["due"] = due_date.strftime("%Y-%m-%dT00:00:00.000Z")
                except ValueError:
                    body["due"] = due

            if not body:
                return {"success": False, "error": "No fields to update"}

            response = await self._oauth_request(
                "PATCH",
                f"{self.api_base_url}/lists/{tasklist}/tasks/{task_id}",
                json=body,
            )

            if response.status_code == 200:
                task = response.json()
                logger.info(f"Google Tasks updated: {task_id}")
                return {"success": True, "data": self._format_task(task, "", tasklist)}
            else:
                logger.error(f"Google Tasks update failed: {response.status_code} - {response.text}")
                return {"success": False, "error": f"Google Tasks API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Google Tasks update error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_task(
        self,
        task_id: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a task from Google Tasks."""
        try:
            tasklist = list_id or "@default"

            response = await self._oauth_request(
                "DELETE",
                f"{self.api_base_url}/lists/{tasklist}/tasks/{task_id}",
            )

            if response.status_code in [200, 204]:
                logger.info(f"Google Tasks deleted: {task_id}")
                return {"success": True}
            else:
                logger.error(f"Google Tasks delete failed: {response.status_code} - {response.text}")
                return {"success": False, "error": f"Google Tasks API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Google Tasks delete error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def list_task_lists(self) -> Dict[str, Any]:
        """List all task lists in Google Tasks."""
        try:
            response = await self._oauth_request(
                "GET",
                f"{self.api_base_url}/users/@me/lists",
            )

            if response.status_code == 200:
                result = response.json()
                items = result.get("items", [])
                lists = [{"id": tl["id"], "name": tl.get("title", "")} for tl in items]
                logger.info(f"Google Tasks listed {len(lists)} task lists")
                return {"success": True, "data": lists}
            else:
                logger.error(f"Google Tasks list lists failed: {response.status_code} - {response.text}")
                return {"success": False, "error": f"Google Tasks API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Google Tasks list lists error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


"""
Todoist Provider - Todoist REST API v2 implementation

Uses Todoist REST API for task operations.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta, timezone

import httpx

from .base import BaseTodoProvider

logger = logging.getLogger(__name__)

# Priority mapping: Todoist -> unified
TODOIST_PRIORITY_TO_UNIFIED = {1: "low", 2: "medium", 3: "high", 4: "urgent"}
UNIFIED_PRIORITY_TO_TODOIST = {"low": 1, "medium": 2, "high": 3, "urgent": 4}


class TodoistProvider(BaseTodoProvider):
    """Todoist task provider implementation using REST API v2."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)
        self.api_base_url = "https://api.todoist.com/rest/v2"

    def _get_headers(self) -> dict:
        """Get authorization headers for API requests."""
        return {"Authorization": f"Bearer {self.access_token}"}

    def _format_task(self, task: dict, project_name: str = "") -> dict:
        """Convert Todoist task to unified format."""
        return {
            "id": task["id"],
            "title": task["content"],
            "due": task["due"]["date"] if task.get("due") else None,
            "priority": TODOIST_PRIORITY_TO_UNIFIED.get(task.get("priority", 1), "low"),
            "completed": task.get("is_completed", False),
            "description": task.get("description", ""),
            "list_name": project_name,
            "list_id": task.get("project_id", ""),
            "_provider": "todoist",
            "_account_name": self.account_name,
            "_account_email": self.email,
        }

    async def _get_project_map(self, client: httpx.AsyncClient) -> Dict[str, str]:
        """Fetch project ID -> name mapping."""
        try:
            response = await client.get(
                f"{self.api_base_url}/projects",
                headers=self._get_headers(),
                timeout=30.0,
            )
            if response.status_code == 200:
                return {p["id"]: p["name"] for p in response.json()}
        except Exception as e:
            logger.warning(f"Failed to fetch projects for name mapping: {e}")
        return {}

    async def list_tasks(
        self,
        list_id: Optional[str] = None,
        completed: bool = False,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """List tasks via Todoist API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                params: Dict[str, Any] = {}
                if list_id:
                    params["project_id"] = list_id

                response = await client.get(
                    f"{self.api_base_url}/tasks",
                    headers=self._get_headers(),
                    params=params,
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/tasks",
                            headers=self._get_headers(),
                            params=params,
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Todoist API error: {response.status_code}"}

                tasks = response.json()
                project_map = await self._get_project_map(client)

                task_list = []
                for task in tasks[:max_results]:
                    project_name = project_map.get(task.get("project_id", ""), "")
                    task_list.append(self._format_task(task, project_name))

                logger.info(f"Todoist listed {len(task_list)} tasks")
                return {"success": True, "data": task_list}

        except Exception as e:
            logger.error(f"Todoist list tasks error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def search_tasks(
        self,
        query: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search tasks by keyword via Todoist API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                params: Dict[str, Any] = {}
                if list_id:
                    params["project_id"] = list_id

                response = await client.get(
                    f"{self.api_base_url}/tasks",
                    headers=self._get_headers(),
                    params=params,
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/tasks",
                            headers=self._get_headers(),
                            params=params,
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Todoist API error: {response.status_code}"}

                tasks = response.json()
                project_map = await self._get_project_map(client)

                query_lower = query.lower()
                matched = []
                for task in tasks:
                    if (query_lower in task.get("content", "").lower()
                            or query_lower in task.get("description", "").lower()):
                        project_name = project_map.get(task.get("project_id", ""), "")
                        matched.append(self._format_task(task, project_name))

                logger.info(f"Todoist search found {len(matched)} tasks for query '{query}'")
                return {"success": True, "data": matched}

        except Exception as e:
            logger.error(f"Todoist search tasks error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def create_task(
        self,
        title: str,
        due: Optional[str] = None,
        priority: Optional[str] = None,
        description: Optional[str] = None,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new task via Todoist API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            body: Dict[str, Any] = {"content": title}
            if due:
                body["due_string"] = due
            if priority:
                body["priority"] = UNIFIED_PRIORITY_TO_TODOIST.get(priority, 1)
            if description:
                body["description"] = description
            if list_id:
                body["project_id"] = list_id

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/tasks",
                    headers=self._get_headers(),
                    json=body,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    task = response.json()
                    project_map = await self._get_project_map(client)
                    project_name = project_map.get(task.get("project_id", ""), "")
                    logger.info(f"Todoist created task: {task['id']}")
                    return {"success": True, "data": self._format_task(task, project_name)}
                else:
                    logger.error(f"Todoist create task failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Todoist API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Todoist create task error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def complete_task(
        self,
        task_id: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark a task as completed via Todoist API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/tasks/{task_id}/close",
                    headers=self._get_headers(),
                    timeout=30.0,
                )

                if response.status_code == 204:
                    logger.info(f"Todoist completed task: {task_id}")
                    return {"success": True}
                else:
                    logger.error(f"Todoist complete task failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Todoist API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Todoist complete task error: {e}", exc_info=True)
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
        """Update an existing task via Todoist API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            # Handle completion state change via separate endpoints
            if completed is True:
                result = await self.complete_task(task_id, list_id)
                if not result.get("success"):
                    return result
            elif completed is False:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.api_base_url}/tasks/{task_id}/reopen",
                        headers=self._get_headers(),
                        timeout=30.0,
                    )
                    if response.status_code != 204:
                        return {"success": False, "error": f"Todoist API error: {response.status_code}"}

            # Update other fields
            body: Dict[str, Any] = {}
            if title is not None:
                body["content"] = title
            if due is not None:
                body["due_string"] = due
            if priority is not None:
                body["priority"] = UNIFIED_PRIORITY_TO_TODOIST.get(priority, 1)
            if description is not None:
                body["description"] = description

            if body:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.api_base_url}/tasks/{task_id}",
                        headers=self._get_headers(),
                        json=body,
                        timeout=30.0,
                    )

                    if response.status_code == 200:
                        task = response.json()
                        project_map = await self._get_project_map(client)
                        project_name = project_map.get(task.get("project_id", ""), "")
                        logger.info(f"Todoist updated task: {task_id}")
                        return {"success": True, "data": self._format_task(task, project_name)}
                    else:
                        logger.error(f"Todoist update task failed: {response.status_code} - {response.text}")
                        return {"success": False, "error": f"Todoist API error: {response.status_code}"}

            # If only completed was changed and succeeded, return success
            logger.info(f"Todoist updated task: {task_id}")
            return {"success": True, "data": {"id": task_id}}

        except Exception as e:
            logger.error(f"Todoist update task error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_task(
        self,
        task_id: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a task via Todoist API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.api_base_url}/tasks/{task_id}",
                    headers=self._get_headers(),
                    timeout=30.0,
                )

                if response.status_code == 204:
                    logger.info(f"Todoist deleted task: {task_id}")
                    return {"success": True}
                else:
                    logger.error(f"Todoist delete task failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Todoist API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Todoist delete task error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def list_task_lists(self) -> Dict[str, Any]:
        """List all projects (task lists) via Todoist API."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/projects",
                    headers=self._get_headers(),
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - attempting to refresh token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{self.api_base_url}/projects",
                            headers=self._get_headers(),
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {"success": False, "error": f"Todoist API error: {response.status_code}"}

                projects = response.json()
                project_list = [{"id": p["id"], "name": p["name"]} for p in projects]

                logger.info(f"Todoist listed {len(project_list)} projects")
                return {"success": True, "data": project_list}

        except Exception as e:
            logger.error(f"Todoist list projects error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Todoist tokens don't expire, so just return success with current token."""
        return {
            "success": True,
            "access_token": self.access_token,
            "expires_in": 0,
            "token_expiry": None,
        }

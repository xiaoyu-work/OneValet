"""
Base Todo Provider - Abstract interface for all todo/task providers

Each todo provider (Todoist, Google Tasks, Microsoft To Do, etc.) must implement this interface.
Providers receive a credentials dict directly - they do not query any database.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


class BaseTodoProvider(ABC):
    """
    Abstract base class for todo/task providers.

    All todo providers must implement:
    - list_tasks()
    - search_tasks()
    - create_task()
    - complete_task()
    - update_task()
    - delete_task()
    - list_task_lists()
    - refresh_access_token()
    """

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        """
        Initialize provider with credentials dict.

        Args:
            credentials: Dict containing:
                - provider: str (todoist, google, microsoft, etc.)
                - account_name: str (user-defined name like "work", "personal")
                - email: str (email address)
                - access_token: str
                - refresh_token: str
                - token_expiry: str or datetime
                - scopes: List[str]
                - provider_data: dict (provider-specific metadata)
            on_token_refreshed: Optional callback invoked with the full new
                credentials dict after a successful token refresh.
        """
        self.credentials = credentials
        self.provider = credentials["provider"]
        self.account_name = credentials.get("account_name", "primary")
        self.email = credentials.get("email", "")
        self.access_token = credentials["access_token"]
        self.refresh_token = credentials.get("refresh_token")
        self.token_expiry = credentials.get("token_expiry")
        self.scopes = credentials.get("scopes", [])
        self.provider_data = credentials.get("provider_data", {})
        self._on_token_refreshed = on_token_refreshed

    # ===== Abstract methods (must be implemented by subclasses) =====

    @abstractmethod
    async def list_tasks(
        self,
        list_id: Optional[str] = None,
        completed: bool = False,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """
        List tasks, optionally filtered by list/project.

        Returns:
            {"success": bool, "data": [task_dict, ...], "error": str}
        """
        pass

    @abstractmethod
    async def search_tasks(
        self,
        query: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search tasks by keyword.

        Returns:
            {"success": bool, "data": [task_dict, ...], "error": str}
        """
        pass

    @abstractmethod
    async def create_task(
        self,
        title: str,
        due: Optional[str] = None,
        priority: Optional[str] = None,
        description: Optional[str] = None,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new task.

        Args:
            title: Task title/content
            due: Due date string (e.g., "2026-02-10", "tomorrow")
            priority: Priority level ("low", "medium", "high", "urgent")
            description: Task description/notes
            list_id: Target list/project ID

        Returns:
            {"success": bool, "data": task_dict, "error": str}
        """
        pass

    @abstractmethod
    async def complete_task(
        self,
        task_id: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Mark a task as completed.

        Returns:
            {"success": bool, "error": str}
        """
        pass

    @abstractmethod
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
        """
        Update an existing task.

        Returns:
            {"success": bool, "data": task_dict, "error": str}
        """
        pass

    @abstractmethod
    async def delete_task(
        self,
        task_id: str,
        list_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Delete a task.

        Returns:
            {"success": bool, "error": str}
        """
        pass

    @abstractmethod
    async def list_task_lists(self) -> Dict[str, Any]:
        """
        List all task lists/projects.

        Returns:
            {"success": bool, "data": [{"id": str, "name": str}, ...], "error": str}
        """
        pass

    @abstractmethod
    async def refresh_access_token(self) -> Dict[str, Any]:
        """
        Refresh expired OAuth access token.

        Returns:
            {
                "success": bool,
                "access_token": str (new token),
                "expires_in": int (seconds),
                "token_expiry": datetime (calculated expiry time),
                "error": str (if failed)
            }
        """
        pass

    # ===== Common helper methods =====

    async def ensure_valid_token(self, force_refresh: bool = False) -> bool:
        """
        Check if access token is valid, refresh if needed.

        After a successful refresh the provider updates its own fields and
        invokes the on_token_refreshed callback so the caller can persist.
        """
        if force_refresh:
            logger.info(f"Forcing token refresh for {self.account_name}...")
            return await self._do_refresh()

        if not self.token_expiry:
            logger.debug(f"No token_expiry info for {self.account_name}, assuming token is valid")
            return True

        # Parse token_expiry if it's a string
        if isinstance(self.token_expiry, str):
            from dateutil import parser
            self.token_expiry = parser.parse(self.token_expiry)

        now = datetime.now(timezone.utc)
        if self.token_expiry <= now + timedelta(minutes=5):
            logger.info(f"Token expiring soon for {self.account_name}, refreshing...")
            return await self._do_refresh()

        return True

    async def _do_refresh(self) -> bool:
        """Perform token refresh and notify callback."""
        result = await self.refresh_access_token()
        if result.get("success"):
            self.access_token = result["access_token"]
            self.token_expiry = result.get("token_expiry")

            # Update internal credentials dict
            self.credentials["access_token"] = self.access_token
            if self.token_expiry:
                self.credentials["token_expiry"] = (
                    self.token_expiry.isoformat()
                    if isinstance(self.token_expiry, datetime)
                    else self.token_expiry
                )

            if self._on_token_refreshed:
                self._on_token_refreshed(self.credentials)

            logger.info(f"Token refreshed successfully for {self.account_name}")
            return True
        else:
            logger.error(f"Token refresh failed for {self.account_name}: {result.get('error')}")
            return False

    def __repr__(self):
        return f"<{self.__class__.__name__} account={self.account_name} email={self.email}>"

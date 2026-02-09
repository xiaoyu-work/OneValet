"""
TaskRepository - Data access for the tasks table.

Used by TaskManagementAgent to list, update, and delete user tasks.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from onevalet.db import Repository

logger = logging.getLogger(__name__)


class TaskRepository(Repository):
    TABLE_NAME = "tasks"
    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS tasks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            name TEXT,
            description TEXT,
            status TEXT DEFAULT 'active',
            trigger_type TEXT,
            trigger_config JSONB,
            action_type TEXT,
            action_config JSONB,
            run_count INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """

    async def get_user_tasks(
        self, tenant_id: str, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get tasks for a tenant, optionally filtered by status."""
        if status:
            rows = await self.db.fetch(
                "SELECT * FROM tasks WHERE tenant_id = $1 AND status = $2 ORDER BY created_at DESC",
                tenant_id,
                status,
            )
        else:
            rows = await self.db.fetch(
                "SELECT * FROM tasks WHERE tenant_id = $1 ORDER BY created_at DESC",
                tenant_id,
            )
        results = []
        for r in rows:
            d = dict(r)
            # Ensure JSONB columns are dicts
            for col in ("trigger_config", "action_config"):
                val = d.get(col)
                if isinstance(val, str):
                    try:
                        d[col] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append(d)
        return results

    async def update_task(
        self, task_id: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a task by id. Returns the updated row."""
        # Serialize JSONB values
        update_data = {}
        for k, v in data.items():
            if k in ("trigger_config", "action_config") and isinstance(v, dict):
                update_data[k] = json.dumps(v)
            else:
                update_data[k] = v
        return await self._update("id", task_id, update_data)

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task by id."""
        return await self._delete("id", task_id)

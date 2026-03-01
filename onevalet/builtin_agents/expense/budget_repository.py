"""
BudgetRepository - Data access for the budgets table.

Manages per-category and total monthly budget limits for expense tracking.
"""

import logging
from typing import Any, Dict, List, Optional

from onevalet.db.repository import Repository

logger = logging.getLogger(__name__)


class BudgetRepository(Repository):
    TABLE_NAME = "budgets"

    async def set_budget(
        self,
        tenant_id: str,
        category: str,
        monthly_limit: float,
        currency: str = "USD",
    ) -> dict:
        """Set or update a budget for a tenant/category pair (upsert)."""
        row = await self._db.fetchrow(
            "INSERT INTO budgets (tenant_id, category, monthly_limit, currency) "
            "VALUES ($1, $2, $3, $4) "
            "ON CONFLICT (tenant_id, category) "
            "DO UPDATE SET monthly_limit = EXCLUDED.monthly_limit, "
            "             currency = EXCLUDED.currency "
            "RETURNING *",
            tenant_id,
            category,
            monthly_limit,
            currency,
        )
        return dict(row) if row else None

    async def get_budget(
        self, tenant_id: str, category: str = "_total"
    ) -> Optional[dict]:
        """Get a single budget for a tenant/category pair."""
        row = await self._db.fetchrow(
            "SELECT * FROM budgets WHERE tenant_id = $1 AND category = $2",
            tenant_id,
            category,
        )
        return dict(row) if row else None

    async def get_all_budgets(self, tenant_id: str) -> list[dict]:
        """Get all budgets for a tenant."""
        return await self._fetch_many(
            where="tenant_id = $1",
            args=(tenant_id,),
            order_by="category ASC",
        )

    async def delete_budget(self, tenant_id: str, category: str) -> bool:
        """Delete a budget for a tenant/category pair."""
        result = await self._db.execute(
            "DELETE FROM budgets WHERE tenant_id = $1 AND category = $2",
            tenant_id,
            category,
        )
        return result == "DELETE 1"

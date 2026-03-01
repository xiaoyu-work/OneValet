"""
ExpenseRepository - Data access for the expenses table.

Used by the expense tracking agent to record, query, and analyze expenses.
"""

import logging
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from onevalet.db.repository import Repository

logger = logging.getLogger(__name__)


class ExpenseRepository(Repository):
    TABLE_NAME = "expenses"

    async def add(
        self,
        tenant_id: str,
        amount: float,
        category: str,
        description: str = "",
        merchant: str = "",
        date: Optional[date] = None,
        currency: str = "USD",
        receipt_id: Optional[str] = None,
    ) -> dict:
        """Insert a new expense and return the created row."""
        data: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "amount": amount,
            "category": category,
            "description": description,
            "merchant": merchant,
            "currency": currency,
        }
        if date is not None:
            data["date"] = date
        if receipt_id is not None:
            data["receipt_id"] = receipt_id
        return await self._insert(data)

    async def query(
        self,
        tenant_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        category: Optional[str] = None,
        merchant: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Fetch expenses with optional filters, ordered by date descending."""
        conditions = ["tenant_id = $1"]
        args: list[Any] = [tenant_id]
        idx = 2

        if start_date is not None:
            conditions.append(f"date >= ${idx}")
            args.append(start_date)
            idx += 1

        if end_date is not None:
            conditions.append(f"date <= ${idx}")
            args.append(end_date)
            idx += 1

        if category is not None:
            conditions.append(f"category = ${idx}")
            args.append(category)
            idx += 1

        if merchant is not None:
            conditions.append(f"merchant = ${idx}")
            args.append(merchant)
            idx += 1

        where = " AND ".join(conditions)
        return await self._fetch_many(
            where=where,
            args=tuple(args),
            order_by="date DESC",
            limit=limit,
        )

    async def update_receipt_id(
        self, expense_id: str, receipt_id: str
    ) -> dict:
        """Attach a receipt to an expense by updating its receipt_id."""
        return await self._update("id", expense_id, {"receipt_id": receipt_id})

    async def delete(self, tenant_id: str, expense_id: str) -> bool:
        """Delete an expense, verifying it belongs to the given tenant."""
        result = await self._db.execute(
            "DELETE FROM expenses WHERE id = $1 AND tenant_id = $2",
            expense_id,
            tenant_id,
        )
        return result == "DELETE 1"

    async def summary_by_category(
        self, tenant_id: str, start_date: date, end_date: date
    ) -> list[dict]:
        """Return per-category totals for the given date range."""
        rows = await self._db.fetch(
            "SELECT category, SUM(amount) AS total_amount, COUNT(*) AS count "
            "FROM expenses "
            "WHERE tenant_id = $1 AND date >= $2 AND date <= $3 "
            "GROUP BY category "
            "ORDER BY total_amount DESC",
            tenant_id,
            start_date,
            end_date,
        )
        return [dict(r) for r in rows]

    async def monthly_total(
        self, tenant_id: str, year: int, month: int,
        category: Optional[str] = None,
    ) -> float:
        """Return the total expense amount for a given month.

        If *category* is provided, returns the total for that category only.
        """
        if category is not None:
            row = await self._db.fetchrow(
                "SELECT COALESCE(SUM(amount), 0) AS total "
                "FROM expenses "
                "WHERE tenant_id = $1 "
                "AND EXTRACT(YEAR FROM date) = $2 "
                "AND EXTRACT(MONTH FROM date) = $3 "
                "AND category = $4",
                tenant_id, year, month, category,
            )
        else:
            row = await self._db.fetchrow(
                "SELECT COALESCE(SUM(amount), 0) AS total "
                "FROM expenses "
                "WHERE tenant_id = $1 "
                "AND EXTRACT(YEAR FROM date) = $2 "
                "AND EXTRACT(MONTH FROM date) = $3",
                tenant_id, year, month,
            )
        return float(row["total"]) if row else 0.0

    async def search(
        self, tenant_id: str, query: str, limit: int = 20
    ) -> list[dict]:
        """Search expenses by description or merchant using ILIKE."""
        pattern = f"%{query}%"
        rows = await self._db.fetch(
            "SELECT * FROM expenses "
            "WHERE tenant_id = $1 "
            "AND (description ILIKE $2 OR merchant ILIKE $2) "
            "ORDER BY date DESC "
            "LIMIT $3",
            tenant_id,
            pattern,
            limit,
        )
        return [dict(r) for r in rows]

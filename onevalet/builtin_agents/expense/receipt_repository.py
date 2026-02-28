"""
ReceiptRepository - Data access for the receipts table.

Stores receipt metadata, cloud storage references, and OCR text
for the expense tracking feature.
"""

import logging
from typing import Any, Dict, List, Optional

from onevalet.db.repository import Repository

logger = logging.getLogger(__name__)


class ReceiptRepository(Repository):
    TABLE_NAME = "receipts"
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS receipts (
        id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id         TEXT NOT NULL,
        expense_id        UUID,
        file_name         TEXT NOT NULL,
        storage_provider  TEXT DEFAULT 'google_drive',
        storage_file_id   TEXT,
        storage_url       TEXT,
        thumbnail_base64  TEXT,
        ocr_text          TEXT DEFAULT '',
        created_at        TIMESTAMPTZ DEFAULT NOW()
    )
    """
    SETUP_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_receipts_tenant_id ON receipts (tenant_id)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_expense_id ON receipts (expense_id)",
    ]

    async def add(
        self,
        tenant_id: str,
        file_name: str,
        expense_id: Optional[str] = None,
        storage_provider: str = "google_drive",
        storage_file_id: Optional[str] = None,
        storage_url: Optional[str] = None,
        thumbnail_base64: Optional[str] = None,
        ocr_text: str = "",
    ) -> dict:
        """Insert a new receipt record and return the created row."""
        data: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "file_name": file_name,
            "storage_provider": storage_provider,
            "ocr_text": ocr_text,
        }
        if expense_id is not None:
            data["expense_id"] = expense_id
        if storage_file_id is not None:
            data["storage_file_id"] = storage_file_id
        if storage_url is not None:
            data["storage_url"] = storage_url
        if thumbnail_base64 is not None:
            data["thumbnail_base64"] = thumbnail_base64
        return await self._insert(data)

    async def get_by_expense(self, expense_id: str) -> Optional[dict]:
        """Get the receipt associated with a given expense."""
        row = await self._db.fetchrow(
            "SELECT * FROM receipts WHERE expense_id = $1",
            expense_id,
        )
        return dict(row) if row else None

    async def search_by_text(
        self, tenant_id: str, query: str, limit: int = 20
    ) -> list[dict]:
        """Search receipts by OCR text or file name using ILIKE."""
        pattern = f"%{query}%"
        rows = await self._db.fetch(
            "SELECT * FROM receipts "
            "WHERE tenant_id = $1 "
            "AND (ocr_text ILIKE $2 OR file_name ILIKE $2) "
            "ORDER BY created_at DESC "
            "LIMIT $3",
            tenant_id,
            pattern,
            limit,
        )
        return [dict(r) for r in rows]

    async def delete(self, receipt_id: str) -> bool:
        """Delete a receipt by its ID."""
        return await self._delete("id", receipt_id)

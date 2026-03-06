"""Profile Repository - Data access for tenant_profiles and tenant_extractions."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from onevalet.db import Repository

logger = logging.getLogger(__name__)


class ProfileRepository(Repository):
    TABLE_NAME = "tenant_profiles"

    # ── tenant_profiles (merged final profile) ──

    async def get_profile(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get the full merged profile for a tenant."""
        row = await self.db.fetchrow(
            "SELECT profile FROM tenant_profiles WHERE tenant_id = $1",
            tenant_id,
        )
        if not row:
            return None
        val = row["profile"]
        return json.loads(val) if isinstance(val, str) else val

    async def get_profile_section(
        self, tenant_id: str, section: str
    ) -> Optional[Any]:
        """Get a specific section (e.g. 'identity', 'work') of the profile."""
        val = await self.db.fetchval(
            "SELECT profile->$2 FROM tenant_profiles WHERE tenant_id = $1",
            tenant_id,
            section,
        )
        if val is None:
            return None
        return json.loads(val) if isinstance(val, str) else val

    async def upsert_profile(
        self, tenant_id: str, profile: Dict[str, Any]
    ) -> None:
        """Insert or update the merged profile for a tenant."""
        now = datetime.now(timezone.utc)
        await self.db.execute(
            """
            INSERT INTO tenant_profiles (tenant_id, profile, extracted_at, updated_at)
            VALUES ($1, $2::jsonb, $3, $3)
            ON CONFLICT (tenant_id)
            DO UPDATE SET profile = $2::jsonb, extracted_at = $3, updated_at = $3
            """,
            tenant_id,
            json.dumps(profile),
            now,
        )

    async def delete_profile(self, tenant_id: str) -> bool:
        """Delete a tenant's profile. Returns True if deleted."""
        result = await self.db.execute(
            "DELETE FROM tenant_profiles WHERE tenant_id = $1",
            tenant_id,
        )
        return result == "DELETE 1"

    # ── tenant_extractions (per-account raw extractions) ──

    async def save_extraction(
        self, tenant_id: str, email_account: str, raw_profile: Dict[str, Any]
    ) -> None:
        """Save a raw extraction result for a specific email account."""
        await self.db.execute(
            """
            INSERT INTO tenant_extractions (tenant_id, email_account, raw_profile)
            VALUES ($1, $2, $3::jsonb)
            """,
            tenant_id,
            email_account,
            json.dumps(raw_profile),
        )

    async def get_extractions(
        self, tenant_id: str
    ) -> List[Dict[str, Any]]:
        """Get all raw extractions for a tenant, ordered by time."""
        rows = await self.db.fetch(
            """
            SELECT email_account, raw_profile, extracted_at
            FROM tenant_extractions
            WHERE tenant_id = $1
            ORDER BY extracted_at ASC
            """,
            tenant_id,
        )
        results = []
        for row in rows:
            val = row["raw_profile"]
            results.append({
                "email_account": row["email_account"],
                "raw_profile": json.loads(val) if isinstance(val, str) else val,
                "extracted_at": row["extracted_at"].isoformat() if row["extracted_at"] else None,
            })
        return results

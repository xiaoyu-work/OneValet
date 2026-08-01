"""PostgresCronRunLog — database-backed cron run history.

Drop-in replacement for the file-based CronRunLog. Uses the cron_runs
table in PostgreSQL.
"""

import json
import logging
from typing import List, Optional

from .models import CronRunEntry

logger = logging.getLogger(__name__)


class PostgresCronRunLog:
    """PostgreSQL persistence for cron run history.

    Implements the same interface as CronRunLog (append, get_runs, prune,
    delete_log, list_job_ids).
    """

    def __init__(self, db):
        self._db = db

    async def append(
        self,
        entry: CronRunEntry,
        max_bytes: int = 0,
        keep_lines: int = 0,
    ) -> None:
        """Append a run entry to the database."""
        data = entry.to_dict()
        try:
            await self._db.execute(
                """
                INSERT INTO cron_runs (job_id, status, summary, error, delivered, duration_ms, data, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, to_timestamp($8 / 1000.0))
                """,
                entry.job_id,
                entry.status,
                entry.summary[:2000] if entry.summary else None,
                entry.error[:2000] if entry.error else None,
                entry.delivered,
                entry.duration_ms,
                json.dumps(data, ensure_ascii=False),
                entry.ts,
            )
        except Exception as e:
            logger.warning(f"Failed to append run log for {entry.job_id}: {e}")

    async def get_runs(
        self,
        job_id: str,
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = None,
    ) -> List[CronRunEntry]:
        """Read run entries for a job, newest first."""
        if status_filter:
            rows = await self._db.fetch(
                """
                SELECT data FROM cron_runs
                WHERE job_id = $1 AND status = $2
                ORDER BY created_at DESC
                LIMIT $3 OFFSET $4
                """,
                job_id,
                status_filter,
                limit,
                offset,
            )
        else:
            rows = await self._db.fetch(
                """
                SELECT data FROM cron_runs
                WHERE job_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                job_id,
                limit,
                offset,
            )

        entries = []
        for row in rows:
            try:
                d = row["data"] if isinstance(row["data"], dict) else json.loads(row["data"])
                entries.append(CronRunEntry.from_dict(d))
            except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed run-log row for job {job_id}: {e}")
                continue
        return entries

    async def prune(
        self,
        job_id: str,
        max_bytes: int = 0,
        keep_lines: int = 500,
    ) -> int:
        """Delete old entries beyond keep_lines for a job."""
        try:
            result = await self._db.execute(
                """
                DELETE FROM cron_runs
                WHERE job_id = $1 AND id NOT IN (
                    SELECT id FROM cron_runs
                    WHERE job_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                )
                """,
                job_id,
                keep_lines,
            )
            # asyncpg returns "DELETE N"
            if result and result.startswith("DELETE "):
                return int(result.split(" ")[1])
            return 0
        except Exception as e:
            logger.warning(f"Failed to prune run log for {job_id}: {e}")
            return 0

    async def delete_log(self, job_id: str) -> None:
        """Delete all run entries for a job."""
        try:
            await self._db.execute(
                "DELETE FROM cron_runs WHERE job_id = $1",
                job_id,
            )
        except Exception as e:
            logger.warning(f"Failed to delete run log for {job_id}: {e}")

    async def list_job_ids(self) -> List[str]:
        """List all job IDs that have run logs."""
        rows = await self._db.fetch("SELECT DISTINCT job_id FROM cron_runs")
        return [row["job_id"] for row in rows]

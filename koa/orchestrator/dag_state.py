"""Durable state for a DAG paused behind one or more approval sub-runs."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DagContinuation:
    parent_run_id: str
    tenant_id: str
    intent: Dict[str, Any]
    metadata: Dict[str, Any]
    results: Dict[str, Any]
    claim_token: Optional[str] = None


def _json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default
    return value


def _row(row: Any) -> DagContinuation:
    return DagContinuation(
        parent_run_id=row["parent_run_id"],
        tenant_id=row["tenant_id"],
        intent=_json(row["intent"], {}),
        metadata=_json(row["metadata"], {}),
        results=_json(row["results"], {}),
        claim_token=row["claim_token"],
    )


class DagContinuationStore:
    def __init__(self, database: Optional[Any] = None) -> None:
        self._db = database

    @property
    def enabled(self) -> bool:
        return self._db is not None

    async def start(
        self,
        parent_run_id: str,
        tenant_id: str,
        intent: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        if not self._db:
            return
        await self._db.execute(
            """
            INSERT INTO dag_continuations
                (parent_run_id, tenant_id, intent, metadata)
            VALUES ($1, $2, $3::jsonb, $4::jsonb)
            ON CONFLICT (parent_run_id) DO NOTHING
            """,
            parent_run_id,
            tenant_id,
            json.dumps(intent, default=str),
            json.dumps(metadata, default=str),
        )

    async def record_result(
        self,
        parent_run_id: str,
        sub_task_id: int,
        result: Dict[str, Any],
        claim_token: Optional[str] = None,
    ) -> bool:
        if not self._db:
            return False
        if claim_token:
            query = """
                UPDATE dag_continuations
                   SET results = results || jsonb_build_object($2::text, $3::jsonb),
                       updated_at = NOW()
                 WHERE parent_run_id = $1 AND claim_token = $4
            """
            args = (
                parent_run_id,
                sub_task_id,
                json.dumps(result, default=str),
                claim_token,
            )
        else:
            query = """
                UPDATE dag_continuations
                   SET results = results || jsonb_build_object($2::text, $3::jsonb),
                       updated_at = NOW()
                 WHERE parent_run_id = $1 AND claim_token IS NULL
            """
            args = (parent_run_id, sub_task_id, json.dumps(result, default=str))
        result_status = await self._db.execute(query, *args)
        return str(result_status).endswith(" 1")

    async def record_result_and_wait(
        self,
        parent_run_id: str,
        sub_run_id: str,
        sub_task_id: int,
        result: Dict[str, Any],
        claim_token: Optional[str] = None,
    ) -> bool:
        """Persist the waiting result and barrier as one fenced transaction."""
        if not self._db:
            return False
        row = await self._db.fetchrow(
            """
            WITH owner AS MATERIALIZED (
                SELECT parent_run_id FROM dag_continuations
                 WHERE parent_run_id = $1
                   AND (
                       ($5::text IS NULL AND claim_token IS NULL)
                       OR claim_token = $5
                   )
                FOR UPDATE
            ),
            updated AS (
                UPDATE dag_continuations AS dag
                   SET results = dag.results
                                 || jsonb_build_object($3::text, $4::jsonb),
                       updated_at = NOW()
                  FROM owner
                 WHERE dag.parent_run_id = owner.parent_run_id
                RETURNING dag.parent_run_id
            ),
            waiting AS (
                INSERT INTO dag_waiting_subruns
                    (sub_run_id, parent_run_id, sub_task_id)
                SELECT $2, updated.parent_run_id, $3 FROM updated
                ON CONFLICT (sub_run_id) DO UPDATE
                    SET parent_run_id = EXCLUDED.parent_run_id,
                        sub_task_id = EXCLUDED.sub_task_id
                RETURNING parent_run_id
            )
            UPDATE dag_continuations AS dag
               SET status = 'waiting', updated_at = NOW()
              FROM waiting
             WHERE dag.parent_run_id = waiting.parent_run_id
            RETURNING dag.parent_run_id
            """,
            parent_run_id,
            sub_run_id,
            sub_task_id,
            json.dumps(result, default=str),
            claim_token,
        )
        return row is not None

    async def touch(
        self,
        parent_run_id: str,
        claim_token: Optional[str] = None,
    ) -> bool:
        if not self._db:
            return False
        if claim_token:
            result = await self._db.execute(
                """
                UPDATE dag_continuations SET updated_at = NOW()
                 WHERE parent_run_id = $1 AND claim_token = $2
                """,
                parent_run_id,
                claim_token,
            )
        else:
            result = await self._db.execute(
                """
                UPDATE dag_continuations SET updated_at = NOW()
                 WHERE parent_run_id = $1 AND claim_token IS NULL
                """,
                parent_run_id,
            )
        return str(result).endswith(" 1")

    async def wait_for(
        self,
        parent_run_id: str,
        sub_run_id: str,
        sub_task_id: int,
        claim_token: Optional[str] = None,
    ) -> bool:
        if not self._db:
            return False
        row = await self._db.fetchrow(
            """
            WITH owner AS MATERIALIZED (
                SELECT parent_run_id FROM dag_continuations
                 WHERE parent_run_id = $1
                   AND (
                       ($4::text IS NULL AND claim_token IS NULL)
                       OR claim_token = $4
                   )
                FOR UPDATE
            ),
            waiting AS (
                INSERT INTO dag_waiting_subruns
                    (sub_run_id, parent_run_id, sub_task_id)
                SELECT $2, owner.parent_run_id, $3 FROM owner
                ON CONFLICT (sub_run_id) DO UPDATE
                    SET parent_run_id = EXCLUDED.parent_run_id,
                        sub_task_id = EXCLUDED.sub_task_id
                RETURNING parent_run_id
            )
            UPDATE dag_continuations AS dag
               SET status = 'waiting', updated_at = NOW()
              FROM waiting
             WHERE dag.parent_run_id = waiting.parent_run_id
            RETURNING dag.parent_run_id
            """,
            parent_run_id,
            sub_run_id,
            sub_task_id,
            claim_token,
        )
        return row is not None

    async def resolve_subrun(
        self,
        sub_run_id: str,
        result: Dict[str, Any],
    ) -> Optional[str]:
        """Merge the resumed child result and remove its wait marker atomically."""
        if not self._db:
            return None
        row = await self._db.fetchrow(
            """
            WITH owner AS MATERIALIZED (
                SELECT dag.parent_run_id
                FROM dag_continuations AS dag
                JOIN dag_waiting_subruns AS waiting
                  ON waiting.parent_run_id = dag.parent_run_id
                WHERE waiting.sub_run_id = $1
                FOR UPDATE OF dag
            ),
            removed AS (
                DELETE FROM dag_waiting_subruns AS waiting
                USING owner
                 WHERE waiting.sub_run_id = $1
                   AND waiting.parent_run_id = owner.parent_run_id
                RETURNING waiting.parent_run_id, waiting.sub_task_id
            ),
            updated AS (
                UPDATE dag_continuations AS dag
                   SET results = dag.results || jsonb_build_object(
                           removed.sub_task_id::text,
                           COALESCE(
                               dag.results -> removed.sub_task_id::text,
                               '{}'::jsonb
                           ) || $2::jsonb
                       ),
                       recovery_attempts = 0,
                       next_recovery_at = NULL,
                       updated_at = NOW()
                  FROM removed
                 WHERE dag.parent_run_id = removed.parent_run_id
                RETURNING dag.parent_run_id
            )
            SELECT parent_run_id FROM updated
            """,
            sub_run_id,
            json.dumps(result, default=str),
        )
        return row["parent_run_id"] if row else None

    async def reserve_recovery(
        self,
        parent_run_id: str,
        stale_after_seconds: int,
        max_attempts: int,
        base_delay_seconds: int,
    ) -> bool:
        if not self._db:
            return False
        row = await self._db.fetchrow(
            """
            UPDATE dag_continuations AS dag
               SET recovery_attempts = recovery_attempts + 1,
                   next_recovery_at = NOW() + make_interval(
                       secs => $4::integer * power(2, recovery_attempts)::integer
                   )
             WHERE dag.parent_run_id = $1
               AND recovery_attempts < $3
               AND (next_recovery_at IS NULL OR next_recovery_at <= NOW())
               AND (
                   (
                       status = 'waiting'
                       AND NOT EXISTS (
                           SELECT 1 FROM dag_waiting_subruns AS waiting
                           WHERE waiting.parent_run_id = dag.parent_run_id
                       )
                   )
                   OR (
                       status = 'running'
                       AND updated_at
                           < NOW() - ($2 || ' seconds')::interval
                   )
               )
            RETURNING parent_run_id
            """,
            parent_run_id,
            str(int(stale_after_seconds)),
            max_attempts,
            base_delay_seconds,
        )
        return row is not None

    async def claim_ready(
        self,
        parent_run_id: str,
        stale_after_seconds: int,
    ) -> Optional[DagContinuation]:
        if not self._db:
            return None
        token = uuid.uuid4().hex
        row = await self._db.fetchrow(
            """
            UPDATE dag_continuations AS dag
               SET status = 'running', claim_token = $3, updated_at = NOW()
             WHERE dag.parent_run_id = $1
               AND (
                   (
                       dag.status = 'waiting'
                       AND NOT EXISTS (
                           SELECT 1 FROM dag_waiting_subruns AS waiting
                           WHERE waiting.parent_run_id = dag.parent_run_id
                       )
                   )
                   OR (
                       dag.status = 'running'
                       AND dag.updated_at
                           < NOW() - ($2 || ' seconds')::interval
                   )
               )
            RETURNING parent_run_id, tenant_id, intent, metadata, results, claim_token
            """,
            parent_run_id,
            str(int(stale_after_seconds)),
            token,
        )
        return _row(row) if row else None

    async def ready_parents(
        self,
        stale_after_seconds: int,
        max_attempts: int,
        limit: int = 100,
    ) -> List[str]:
        if not self._db:
            return []
        rows = await self._db.fetch(
            """
            SELECT dag.parent_run_id
            FROM dag_continuations AS dag
            WHERE (
                (
                    dag.status = 'waiting'
                    AND NOT EXISTS (
                        SELECT 1 FROM dag_waiting_subruns AS waiting
                        WHERE waiting.parent_run_id = dag.parent_run_id
                    )
                )
                OR (
                    dag.status = 'running'
                    AND dag.updated_at
                        < NOW() - ($1 || ' seconds')::interval
                )
              )
              AND dag.recovery_attempts < $2
              AND (dag.next_recovery_at IS NULL OR dag.next_recovery_at <= NOW())
            ORDER BY dag.updated_at
            LIMIT $3
            """,
            str(int(stale_after_seconds)),
            max_attempts,
            limit,
        )
        return [row["parent_run_id"] for row in rows]

    async def list_resumable(
        self,
        tenant_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if not self._db:
            return []
        rows = await self._db.fetch(
            """
            SELECT parent_run_id, status, recovery_attempts, next_recovery_at,
                   updated_at
            FROM dag_continuations
            WHERE tenant_id = $1 AND status IN ('waiting', 'running')
            ORDER BY updated_at DESC
            LIMIT $2
            """,
            tenant_id,
            limit,
        )
        return [
            {
                "run_id": row["parent_run_id"],
                "status": f"dag_{row['status']}",
                "recovery_attempts": row["recovery_attempts"],
                "next_recovery_at": (
                    row["next_recovery_at"].isoformat()
                    if row["next_recovery_at"]
                    else None
                ),
                "updated_at": (
                    row["updated_at"].isoformat() if row["updated_at"] else None
                ),
            }
            for row in rows
        ]

    async def is_unfinished(self, parent_run_id: str) -> bool:
        if not self._db:
            return False
        row = await self._db.fetchrow(
            """
            SELECT 1 FROM dag_continuations
            WHERE parent_run_id = $1 AND status IN ('waiting', 'running')
            """,
            parent_run_id,
        )
        return row is not None

    async def complete(
        self,
        parent_run_id: str,
        claim_token: Optional[str] = None,
    ) -> bool:
        if self._db:
            if claim_token:
                result = await self._db.execute(
                    """
                    DELETE FROM dag_continuations
                     WHERE parent_run_id = $1 AND claim_token = $2
                    """,
                    parent_run_id,
                    claim_token,
                )
            else:
                result = await self._db.execute(
                    """
                    DELETE FROM dag_continuations
                     WHERE parent_run_id = $1 AND claim_token IS NULL
                    """,
                    parent_run_id,
                )
            return str(result).endswith(" 1")
        return False

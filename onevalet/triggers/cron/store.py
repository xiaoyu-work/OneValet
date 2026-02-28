"""CronJobStore — JSON file persistence with atomic writes and backup."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from .models import CronJob

logger = logging.getLogger(__name__)

STORE_VERSION = 1


class CronJobStore:
    """JSON file persistence for cron jobs.

    Stores all jobs in a single JSON file. Uses atomic writes
    (temp file + rename) with automatic .bak backup, matching
    OpenClaw's store.ts pattern.
    """

    def __init__(self, store_path: str = "~/.onevalet/cron/jobs.json"):
        self._store_path = Path(os.path.expanduser(store_path))
        self._jobs: Dict[str, CronJob] = {}

    @property
    def store_path(self) -> Path:
        return self._store_path

    async def load(self) -> None:
        """Load jobs from disk. Creates empty store if file doesn't exist."""
        if not self._store_path.exists():
            self._jobs = {}
            logger.info(f"Cron store not found at {self._store_path}, starting empty")
            return

        try:
            raw = self._store_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            version = data.get("version", 1)
            if version != STORE_VERSION:
                logger.warning(f"Cron store version mismatch: expected {STORE_VERSION}, got {version}")

            self._jobs = {}
            for job_dict in data.get("jobs", []):
                try:
                    job = CronJob.from_dict(job_dict)
                    self._jobs[job.id] = job
                except Exception as e:
                    logger.warning(f"Skipping invalid job entry: {e}")

            logger.info(f"Loaded {len(self._jobs)} cron jobs from {self._store_path}")
        except Exception as e:
            logger.error(f"Failed to load cron store from {self._store_path}: {e}")
            self._jobs = {}

    async def save(self) -> None:
        """Persist jobs to disk with atomic write and backup."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

        store_data = {
            "version": STORE_VERSION,
            "jobs": [job.to_dict() for job in self._jobs.values()],
        }
        content = json.dumps(store_data, indent=2, ensure_ascii=False)

        # Create backup of existing file
        if self._store_path.exists():
            bak_path = self._store_path.with_suffix(".json.bak")
            try:
                import shutil
                shutil.copy2(str(self._store_path), str(bak_path))
            except Exception as e:
                logger.debug(f"Backup creation failed (non-fatal): {e}")

        # Atomic write: temp file in same directory, then rename
        dir_path = self._store_path.parent
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=".jobs-", suffix=".tmp", dir=str(dir_path)
            )
            try:
                os.write(fd, content.encode("utf-8"))
            finally:
                os.close(fd)
            os.replace(tmp_path, str(self._store_path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise

    def get(self, job_id: str) -> Optional[CronJob]:
        return self._jobs.get(job_id)

    def list(
        self,
        user_id: Optional[str] = None,
        include_disabled: bool = False,
    ) -> List[CronJob]:
        """List jobs, optionally filtered by user_id and enabled status."""
        jobs = list(self._jobs.values())
        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]
        if not include_disabled:
            jobs = [j for j in jobs if j.enabled]
        # Sort by next_run_at_ms (None last)
        jobs.sort(key=lambda j: j.state.next_run_at_ms or float("inf"))
        return jobs

    def add(self, job: CronJob) -> None:
        self._jobs[job.id] = job

    def update(self, job: CronJob) -> None:
        self._jobs[job.id] = job

    def remove(self, job_id: str) -> bool:
        return self._jobs.pop(job_id, None) is not None

    def get_next_due_time(self) -> Optional[int]:
        """Return the earliest next_run_at_ms across all enabled, non-running jobs."""
        earliest: Optional[int] = None
        for job in self._jobs.values():
            if not job.enabled:
                continue
            if job.state.running_at_ms is not None:
                continue
            nra = job.state.next_run_at_ms
            if nra is not None and (earliest is None or nra < earliest):
                earliest = nra
        return earliest

    def find_by_name(self, name: str, user_id: Optional[str] = None) -> Optional[CronJob]:
        """Find a job by exact or partial name match."""
        name_lower = name.lower()
        for job in self._jobs.values():
            if user_id and job.user_id != user_id:
                continue
            if job.name.lower() == name_lower:
                return job
        # Partial match fallback
        for job in self._jobs.values():
            if user_id and job.user_id != user_id:
                continue
            if name_lower in job.name.lower():
                return job
        return None

    def find_by_hint(self, hint: str, user_id: Optional[str] = None) -> Optional[CronJob]:
        """Find a job by ID or name hint."""
        # Try exact ID first
        job = self.get(hint)
        if job and (user_id is None or job.user_id == user_id):
            return job
        # Try name match
        return self.find_by_name(hint, user_id)

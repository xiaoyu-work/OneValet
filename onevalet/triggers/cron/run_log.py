"""CronRunLog — per-job JSONL run history with pruning."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from .models import CronRunEntry

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 2_000_000  # 2 MB
DEFAULT_KEEP_LINES = 2000


class CronRunLog:
    """Per-job JSONL run history.

    Stores one ``{job_id}.jsonl`` file per job in ``{data_dir}/runs/``.
    Supports automatic pruning when files exceed size limits.
    """

    def __init__(self, data_dir: str = "~/.onevalet/cron"):
        self._runs_dir = Path(os.path.expanduser(data_dir)) / "runs"

    def _job_log_path(self, job_id: str) -> Path:
        """Return path for a job's log file, preventing directory traversal."""
        safe_id = job_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self._runs_dir / f"{safe_id}.jsonl"

    async def append(
        self,
        entry: CronRunEntry,
        max_bytes: int = DEFAULT_MAX_BYTES,
        keep_lines: int = DEFAULT_KEEP_LINES,
    ) -> None:
        """Append a run entry and prune if file is too large."""
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        path = self._job_log_path(entry.job_id)

        line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.warning(f"Failed to append run log for {entry.job_id}: {e}")
            return

        # Prune if file exceeds max_bytes
        try:
            if path.stat().st_size > max_bytes:
                await self.prune(entry.job_id, max_bytes=max_bytes, keep_lines=keep_lines)
        except Exception as e:
            logger.debug(f"Prune check failed (non-fatal): {e}")

    async def get_runs(
        self,
        job_id: str,
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = None,
    ) -> List[CronRunEntry]:
        """Read run entries for a job, newest first."""
        path = self._job_log_path(job_id)
        if not path.exists():
            return []

        entries: List[CronRunEntry] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        entry = CronRunEntry.from_dict(d)
                        if status_filter and entry.status != status_filter:
                            continue
                        entries.append(entry)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Failed to read run log for {job_id}: {e}")
            return []

        # Newest first
        entries.reverse()

        # Apply offset + limit
        return entries[offset : offset + limit]

    async def prune(
        self,
        job_id: str,
        max_bytes: int = DEFAULT_MAX_BYTES,
        keep_lines: int = DEFAULT_KEEP_LINES,
    ) -> int:
        """Prune old entries, keeping the most recent keep_lines. Returns lines removed."""
        path = self._job_log_path(job_id)
        if not path.exists():
            return 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return 0

        if len(lines) <= keep_lines:
            return 0

        removed = len(lines) - keep_lines
        kept = lines[-keep_lines:]

        # Atomic rewrite
        try:
            dir_path = path.parent
            fd, tmp_path = tempfile.mkstemp(prefix=".run-", suffix=".tmp", dir=str(dir_path))
            try:
                os.write(fd, "".join(kept).encode("utf-8"))
            finally:
                os.close(fd)
            os.replace(tmp_path, str(path))
        except Exception as e:
            logger.warning(f"Failed to prune run log for {job_id}: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return 0

        return removed

    async def delete_log(self, job_id: str) -> None:
        """Delete the run log file for a job."""
        path = self._job_log_path(job_id)
        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete run log for {job_id}: {e}")

    async def list_job_ids(self) -> List[str]:
        """List all job IDs that have run logs."""
        if not self._runs_dir.exists():
            return []
        return [
            p.stem for p in self._runs_dir.glob("*.jsonl")
        ]

"""In-process background manager for asynchronous interactions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from .interaction_job_record import InteractionJobRecord


class InteractionJobManager:
    """Runs interaction tasks in a worker pool and tracks task state in-memory."""

    def __init__(self, *, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ai-interaction")
        self._jobs: dict[str, InteractionJobRecord] = {}
        self._lock = Lock()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def submit(self, work: Callable[[], dict[str, Any]]) -> str:
        job_id = str(uuid4())
        now = self._now()

        with self._lock:
            self._jobs[job_id] = InteractionJobRecord(
                id=job_id,
                status="queued",
                created_at=now,
                updated_at=now,
            )

        def _wrapped() -> None:
            with self._lock:
                record = self._jobs[job_id]
                record.status = "running"
                record.updated_at = self._now()

            try:
                result = work()
            except Exception as exc:
                with self._lock:
                    record = self._jobs[job_id]
                    record.status = "failed"
                    record.updated_at = self._now()
                    record.error = {
                        "code": "runtime_unavailable",
                        "message": "There was a problem with the model contact the administrator.",
                        "details": {"exception": exc.__class__.__name__},
                    }
                return

            with self._lock:
                record = self._jobs[job_id]
                record.status = "succeeded"
                record.updated_at = self._now()
                record.result = result

        self._executor.submit(_wrapped)
        return job_id

    def get(self, job_id: str) -> InteractionJobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

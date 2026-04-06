"""Minimal experimental alternative to Celery using Huey."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .interaction_job_record import InteractionJobRecord


class InteractionJobManagerHuey:
    """Small optional alternative manager for comparison in college projects.

    Requires Huey and a configured Huey instance (typically RedisHuey).
    """

    def __init__(self, *, huey: Any, task_callable: Any) -> None:
        self._huey = huey
        self._task_callable = task_callable
        self._submitted_at: dict[str, datetime] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def submit_work_ref(self, *, work_ref: str, kwargs: dict[str, Any] | None = None) -> str:
        job_id = str(uuid4())
        self._submitted_at[job_id] = self._now()
        self._task_callable(work_ref=work_ref, kwargs=kwargs or {}, task_id=job_id)
        return job_id

    def get(self, job_id: str) -> InteractionJobRecord | None:
        created_at = self._submitted_at.get(job_id)
        if created_at is None:
            return None
        # Huey result polling varies by backend; this is intentionally minimal.
        return InteractionJobRecord(id=job_id, status="queued", created_at=created_at, updated_at=self._now())

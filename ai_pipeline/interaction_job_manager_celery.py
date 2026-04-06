"""Experimental Celery-backed interaction job manager."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from celery.result import AsyncResult

from .interaction_job_record import InteractionJobRecord
from .interaction_tasks_celery import (
    InteractionTasksCelery,
    execute_local_callable_task,
    execute_work_ref_task,
)


class InteractionJobManagerCelery:
    """Celery-based variant of the interaction job manager.

    This class is additive and does not replace InteractionJobManager.
    """

    def __init__(self, *, celery_app: Any) -> None:
        self._celery_app = celery_app
        self._submitted_at: dict[str, datetime] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def submit(self, work: Callable[[], dict[str, Any]]) -> str:
        """Submit a Python callable using local registration.

        Best for local experiments, especially with Celery eager mode enabled.
        """

        job_id = str(uuid4())
        self._submitted_at[job_id] = self._now()
        InteractionTasksCelery.register_local_work(job_id, work)
        execute_local_callable_task.apply_async(args=[job_id], task_id=job_id, app=self._celery_app)
        return job_id

    def submit_work_ref(self, *, work_ref: str, kwargs: dict[str, Any] | None = None) -> str:
        """Submit a worker-safe import reference: ``module.path:function_name``."""

        job_id = str(uuid4())
        self._submitted_at[job_id] = self._now()
        execute_work_ref_task.apply_async(args=[work_ref, kwargs], task_id=job_id, app=self._celery_app)
        return job_id

    def get(self, job_id: str) -> InteractionJobRecord | None:
        created_at = self._submitted_at.get(job_id)
        if created_at is None:
            return None

        async_result = AsyncResult(job_id, app=self._celery_app)
        status = self._map_status(async_result.state)
        now = self._now()

        if status == "succeeded":
            return InteractionJobRecord(
                id=job_id,
                status=status,
                created_at=created_at,
                updated_at=now,
                result=async_result.result if isinstance(async_result.result, dict) else None,
            )

        if status == "failed":
            error = {
                "code": "runtime_unavailable",
                "message": "There was a problem with the model contact the administrator.",
                "details": {"exception": async_result.result.__class__.__name__},
            }
            return InteractionJobRecord(
                id=job_id,
                status=status,
                created_at=created_at,
                updated_at=now,
                error=error,
            )

        return InteractionJobRecord(
            id=job_id,
            status=status,
            created_at=created_at,
            updated_at=now,
        )

    @staticmethod
    def _map_status(celery_state: str) -> str:
        if celery_state in {"PENDING", "RECEIVED"}:
            return "queued"
        if celery_state in {"STARTED", "RETRY"}:
            return "running"
        if celery_state == "SUCCESS":
            return "succeeded"
        if celery_state in {"FAILURE", "REVOKED"}:
            return "failed"
        return "queued"

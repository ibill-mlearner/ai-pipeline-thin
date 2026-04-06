"""Experimental Celery app factory for interaction jobs.

This is intentionally additive and does not replace the in-memory manager.
"""

from __future__ import annotations

import os
from typing import Any


def create_celery_app(**overrides: Any):
    """Create a Celery app for experimental interaction job processing.

    Defaults assume Redis for both broker and result backend.
    Set CELERY_BROKER_URL/CELERY_RESULT_BACKEND to change this.
    """

    from celery import Celery

    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    app = Celery(
        "ai_pipeline_experimental",
        broker=broker_url,
        backend=result_backend,
        include=["ai_pipeline.interaction_tasks_celery"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        timezone="UTC",
    )
    app.conf.update(**overrides)
    return app

"""Experimental registry for Celery-backed interaction job manager."""

from __future__ import annotations

from typing import Any

from .interaction_job_manager_celery import InteractionJobManagerCelery


class InteractionJobRegistryCelery:
    """Flask-extension registry for experimental Celery manager."""

    EXTENSION_KEY = "ai_interaction_job_manager_celery"

    @classmethod
    def get_or_create(cls, app: Any, *, celery_app: Any) -> InteractionJobManagerCelery:
        manager = app.extensions.get(cls.EXTENSION_KEY)
        if isinstance(manager, InteractionJobManagerCelery):
            return manager

        manager = InteractionJobManagerCelery(celery_app=celery_app)
        app.extensions[cls.EXTENSION_KEY] = manager
        return manager

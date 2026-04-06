"""Factory/registry for interaction job manager instances."""

from __future__ import annotations

from typing import Any

from .interaction_job_manager import InteractionJobManager


class InteractionJobManagerRegistry:
    """Resolves a singleton interaction manager from Flask-style app extensions."""

    @staticmethod
    def get_or_create(app: Any, *, max_workers: int = 2) -> InteractionJobManager:
        manager = app.extensions.get("ai_interaction_job_manager")
        if isinstance(manager, InteractionJobManager):
            return manager

        manager = InteractionJobManager(max_workers=max_workers)
        app.extensions["ai_interaction_job_manager"] = manager
        return manager

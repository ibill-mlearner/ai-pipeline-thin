"""Persistence-aware wrapper for interaction execution flow."""

from __future__ import annotations

from typing import Any, Callable


class InteractionPersistenceRunner:
    """Runs interaction execution and validates persistence completion."""

    @staticmethod
    def run_and_persist(
        *,
        run_fn: Callable[[], dict[str, Any]],
        warn_if_empty_fn: Callable[[str, dict[str, Any]], None],
        persist_fn: Callable[[], Any],
        request_id: str,
    ) -> dict[str, Any]:
        normalized_result = run_fn()
        warn_if_empty_fn(request_id, normalized_result)
        persistence_error = persist_fn()
        if persistence_error is not None:
            raise RuntimeError("Interaction persistence failed")
        return normalized_result

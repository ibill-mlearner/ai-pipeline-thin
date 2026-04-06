"""Experimental Celery tasks for interaction job execution."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from celery import shared_task

_LOCAL_WORK_REGISTRY: dict[str, Callable[[], dict[str, Any]]] = {}


class InteractionTasksCelery:
    """Namespace for helper methods used by experimental Celery tasks."""

    @staticmethod
    def register_local_work(job_id: str, work: Callable[[], dict[str, Any]]) -> None:
        """Register a callable for local dev/eager-mode experiments.

        This is not suitable for distributed workers because it uses in-process memory.
        """

        _LOCAL_WORK_REGISTRY[job_id] = work

    @staticmethod
    def load_work_ref(work_ref: str) -> Callable[..., dict[str, Any]]:
        module_name, _, attr_name = work_ref.partition(":")
        if not module_name or not attr_name:
            raise ValueError("work_ref must look like 'module.path:function_name'")
        module = import_module(module_name)
        fn = getattr(module, attr_name)
        if not callable(fn):
            raise TypeError(f"Work reference {work_ref!r} did not resolve to a callable")
        return fn


@shared_task(name="ai_pipeline.interaction.execute_work_ref")
def execute_work_ref_task(work_ref: str, kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute an importable function reference (worker-safe approach)."""

    fn = InteractionTasksCelery.load_work_ref(work_ref)
    payload = kwargs or {}
    return fn(**payload)


@shared_task(name="ai_pipeline.interaction.execute_local_callable")
def execute_local_callable_task(job_id: str) -> dict[str, Any]:
    """Execute a locally registered callable.

    Useful for side-by-side local experiments with task_always_eager=True.
    """

    work = _LOCAL_WORK_REGISTRY.pop(job_id)
    return work()

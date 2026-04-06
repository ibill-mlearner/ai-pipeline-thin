"""Unified gateway object for accessing exported ai_pipeline symbols."""

from __future__ import annotations

from importlib import import_module
from typing import Any


class AIPipelineInterface:
    """Single-object gateway to package exports.

    Example:
        interface = AIPipelineInterface()
        service = interface.AIPipelineInteractionService()
        manager = interface.InteractionJobManager(max_workers=2)
    """

    def __init__(self) -> None:
        self._module = import_module("ai_pipeline")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def build(self, class_name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate a class exported by ``ai_pipeline`` by name."""

        target = getattr(self._module, class_name)
        return target(*args, **kwargs)

"""Request contract objects for AI interaction execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AIPipelineRequest:
    """Minimal request shape expected by service adapters."""

    prompt: str
    system_prompt: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    provider: str | None = None
    model_id: str | None = None

    def resolved_system_prompt(self, default_system_prompt: str) -> str:
        """Return explicit system prompt when provided, otherwise default."""
        if self.system_prompt is None:
            return default_system_prompt
        return self.system_prompt

"""Job record model for asynchronous interaction execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class InteractionJobRecord:
    """Represents an interaction task lifecycle and payload state."""

    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

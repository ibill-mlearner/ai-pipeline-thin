"""Error type raised when pipeline invocation fails."""

from __future__ import annotations


class AIPipelineUpstreamError(Exception):
    """Wrapper error preserving upstream failure metadata for adapters."""

    def __init__(self, message: str, *, details: dict[str, str]):
        super().__init__(message)
        self.details = details

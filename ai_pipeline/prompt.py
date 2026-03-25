"""Prompt abstraction for the AI pipeline."""

from dataclasses import dataclass


@dataclass
class Prompt:
    """Represents the user prompt text."""

    value: str = "Give me a short introduction to large language model."

    def get(self) -> str:
        """Return the prompt text."""
        return self.value

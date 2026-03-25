"""Model name abstraction for the AI pipeline."""

from dataclasses import dataclass


@dataclass
class ModelName:
    """Stores and validates the Hugging Face model identifier."""

    value: str = "Qwen/Qwen2.5-3B-Instruct"

    def get(self) -> str:
        """Return the model identifier string."""
        return self.value

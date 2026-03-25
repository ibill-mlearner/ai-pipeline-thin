"""Tokenizer loading abstraction for the AI pipeline."""

from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer


@dataclass
class TokenizerLoader:
    """Loads a tokenizer for the selected model."""

    model_name: str
    download_locally: bool = True

    def _cache_dir(self) -> str | None:
        """Return local models directory if local download is enabled."""
        if not self.download_locally:
            return None

        local_models_dir = Path(__file__).resolve().parent / "models"
        local_models_dir.mkdir(parents=True, exist_ok=True)
        return str(local_models_dir)

    def build(self):
        """Load and return the tokenizer instance."""
        return AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self._cache_dir(),
        )

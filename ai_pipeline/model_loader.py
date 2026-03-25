"""Model loading abstraction for the AI pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelLoader:
    """Loads a causal language model from Hugging Face Transformers."""

    model_name: str
    torch_dtype: str | Any = "auto"
    device_map: str | dict[str, Any] = "auto"
    download_locally: bool = True

    def _cache_dir(self) -> str | None:
        """Return local models directory if local download is enabled."""
        if not self.download_locally:
            return None

        local_models_dir = Path(__file__).resolve().parent / "models"
        local_models_dir.mkdir(parents=True, exist_ok=True)
        return str(local_models_dir)

    def build(self):
        """Load and return the model instance."""
        from transformers import AutoModelForCausalLM

        cache_dir = self._cache_dir()
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            cache_dir=cache_dir,
        )

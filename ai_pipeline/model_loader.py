"""Model loading abstraction for the AI pipeline."""

from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any

from .dependency_diagnostics import (
    build_transformers_import_error_details,
    ensure_stdlib_logging_available,
)


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
        try:
            ensure_stdlib_logging_available()
            from transformers import AutoModelForCausalLM
        except Exception as exc:  # pragma: no cover - environment-dependent imports.
            raise ImportError(build_transformers_import_error_details(exc)) from exc

        cache_dir = self._cache_dir()
        from_pretrained_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "cache_dir": cache_dir,
        }

        from_pretrained_signature = signature(AutoModelForCausalLM.from_pretrained)
        if "dtype" in from_pretrained_signature.parameters:
            from_pretrained_kwargs["dtype"] = self.torch_dtype
        else:
            from_pretrained_kwargs["torch_dtype"] = self.torch_dtype

        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **from_pretrained_kwargs,
        )

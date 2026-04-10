"""Service wrapper dedicated to model/tokenizer artifact prefetching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model_loader import ModelLoader
from .tokenizer_loader import TokenizerLoader
from .upstream_error import AIPipelineUpstreamError


@dataclass
class AIPipelineModelDownloadService:
    """Downloads and caches model assets without running generation."""

    default_provider: str = "huggingface"

    def download(self, model_id: str, *, provider: str | None = None) -> dict[str, Any]:
        """Download model and tokenizer files for the given Hugging Face model id."""
        resolved_provider = provider or self.default_provider

        try:
            model_loader = ModelLoader(
                model_name=model_id,
                device_map=None,
                torch_dtype="auto",
                download_locally=True,
            )
            tokenizer_loader = TokenizerLoader(
                model_name=model_id,
                download_locally=True,
            )

            model_loader.build()
            tokenizer_loader.build()
        except Exception as exc:
            raise AIPipelineUpstreamError(
                "Model download failed.",
                details={
                    "exception_class": exc.__class__.__name__,
                    "message": str(exc),
                    "model_id": model_id,
                },
            ) from exc

        return {
            "provider": resolved_provider,
            "model_id": model_id,
            "status": "downloaded",
        }

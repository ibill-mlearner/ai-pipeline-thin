"""Thin service wrapper for interaction execution against the AI pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model_loader import ModelLoader
from .pipeline import AIPipeline
from .tokenizer_loader import TokenizerLoader
from .request import AIPipelineRequest
from .upstream_error import AIPipelineUpstreamError


@dataclass
class AIPipelineInteractionService:
    """Executes AI interactions while preserving API-layer boundaries."""

    default_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    default_system_prompt: str = "You are a helpful assistant."

    @staticmethod
    def resolve_model_id(
        request: AIPipelineRequest,
        session_model_id: str | None,
        configured_default_model_id: str,
    ) -> str:
        """Resolve model selection source without mutating model identifiers."""
        if request.model_id:
            return request.model_id
        if session_model_id:
            return session_model_id
        return configured_default_model_id

    @staticmethod
    def prompt_from_request(request: AIPipelineRequest) -> str:
        """Use explicit prompt, or fallback to latest user message if prompt is empty."""
        if request.prompt:
            return request.prompt

        for message in reversed(request.messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]

        return ""

    def execute(
        self,
        request: AIPipelineRequest,
        *,
        session_model_id: str | None = None,
        configured_default_model_id: str | None = None,
        download_locally: bool = True,
    ) -> dict[str, Any]:
        """Run one interaction through the pipeline with top-level error wrapping."""
        model_id = self.resolve_model_id(
            request,
            session_model_id=session_model_id,
            configured_default_model_id=configured_default_model_id or self.default_model_id,
        )

        system_prompt = request.resolved_system_prompt(self.default_system_prompt)
        prompt = self.prompt_from_request(request)

        try:
            pipeline = AIPipeline(
                model_name_value=model_id,
                system_content=system_prompt,
                prompt_value=prompt,
                download_locally=download_locally,
            )

            model = pipeline.build_model()
            tokenizer = pipeline.build_tokenizer()
            text = pipeline.build_text(tokenizer=tokenizer)
            model_inputs = pipeline.build_model_inputs(tokenizer=tokenizer, text=text, model=model)
            raw_ids = pipeline.build_raw_generated_ids(model=model, model_inputs=model_inputs)
            generated_ids = pipeline.build_generated_ids(model_inputs=model_inputs, raw_generated_ids=raw_ids)
            response_text = pipeline.build_response(tokenizer=tokenizer, generated_ids=generated_ids)
        except Exception as exc:
            raise AIPipelineUpstreamError(
                "Pipeline invocation failed.",
                details={
                    "exception_class": exc.__class__.__name__,
                    "message": str(exc),
                },
            ) from exc

        return {
            "provider": request.provider,
            "model_id": model_id,
            "response": response_text,
            "context": request.context,
        }

    def download_model(
        self,
        model_id: str,
        *,
        provider: str = "huggingface",
    ) -> dict[str, Any]:
        """Download and cache model/tokenizer artifacts without running generation."""
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
            "provider": provider,
            "model_id": model_id,
            "status": "downloaded",
        }

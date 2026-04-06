"""Ollama-backed interaction service.

This service delegates model execution to a running Ollama server. GPU selection and
runtime details are managed by Ollama itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib import error, request as urllib_request

from .request import AIPipelineRequest
from .upstream_error import AIPipelineUpstreamError


@dataclass
class OllamaInteractionService:
    """Execute AI interactions against an Ollama server endpoint."""

    base_url: str = "http://127.0.0.1:11434"
    default_model_id: str = "ollama/llama3.2:1b"
    default_system_prompt: str = "You are a helpful assistant."

    @staticmethod
    def _resolve_model_id(
        request_payload: AIPipelineRequest,
        session_model_id: str | None,
        configured_default_model_id: str,
    ) -> str:
        if request_payload.model_id:
            return request_payload.model_id
        if session_model_id:
            return session_model_id
        return configured_default_model_id

    @staticmethod
    def _normalize_ollama_model(model_id: str) -> str:
        if model_id.lower().startswith("ollama/"):
            return model_id.split("/", 1)[1]
        return model_id

    @staticmethod
    def _prompt_from_request(request_payload: AIPipelineRequest) -> str:
        if request_payload.prompt:
            return request_payload.prompt

        for message in reversed(request_payload.messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]

        return ""

    def execute(
        self,
        request_payload: AIPipelineRequest,
        *,
        session_model_id: str | None = None,
        configured_default_model_id: str | None = None,
    ) -> dict[str, Any]:
        model_id = self._resolve_model_id(
            request_payload,
            session_model_id=session_model_id,
            configured_default_model_id=configured_default_model_id or self.default_model_id,
        )
        ollama_model = self._normalize_ollama_model(model_id)
        system_prompt = request_payload.resolved_system_prompt(self.default_system_prompt)
        prompt = self._prompt_from_request(request_payload)

        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
        }

        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url.rstrip('/')}/api/generate"
        req = urllib_request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")

        try:
            with urllib_request.urlopen(req, timeout=120) as response:
                raw = response.read().decode("utf-8")
                decoded = json.loads(raw)
        except (OSError, error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise AIPipelineUpstreamError(
                "Ollama invocation failed.",
                details={
                    "exception_class": exc.__class__.__name__,
                    "message": str(exc),
                    "url": url,
                },
            ) from exc

        return {
            "provider": request_payload.provider or "ollama",
            "model_id": f"ollama/{ollama_model}",
            "response": decoded.get("response", ""),
            "context": request_payload.context,
            "raw": decoded,
        }

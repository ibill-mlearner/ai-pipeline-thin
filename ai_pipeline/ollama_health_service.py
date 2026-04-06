"""Ollama server health and observability helper."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib import error, request as urllib_request


@dataclass
class OllamaHealthService:
    """Read-only health checks for Ollama service status and model visibility."""

    base_url: str = "http://127.0.0.1:11434"

    def is_running(self) -> bool:
        url = f"{self.base_url.rstrip('/')}/api/tags"
        req = urllib_request.Request(url, method="GET")
        try:
            with urllib_request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except (OSError, error.URLError, error.HTTPError, TimeoutError):
            return False

    def status(self) -> dict[str, Any]:
        if not self.is_running():
            return {
                "status": "error",
                "running": False,
                "message": "Ollama service is not reachable.",
                "base_url": self.base_url,
            }

        models_payload = self.list_models()
        return {
            "status": "ok",
            "running": True,
            "message": "Ollama service is reachable.",
            "base_url": self.base_url,
            "models": models_payload.get("models", []),
        }

    def list_models(self) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/api/tags"
        req = urllib_request.Request(url, method="GET")
        try:
            with urllib_request.urlopen(req, timeout=5) as response:
                raw = response.read().decode("utf-8")
                decoded = json.loads(raw)
                models = decoded.get("models", []) if isinstance(decoded, dict) else []
                return {
                    "status": "ok",
                    "models": models,
                    "count": len(models),
                }
        except (OSError, error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            return {
                "status": "error",
                "models": [],
                "count": 0,
                "error": {"exception_class": exc.__class__.__name__, "message": str(exc)},
            }

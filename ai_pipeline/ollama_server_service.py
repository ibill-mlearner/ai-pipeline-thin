"""Local Ollama server lifecycle helper."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import subprocess
import time
from typing import Any
from urllib import error, request as urllib_request


@dataclass
class OllamaServerService:
    """Start and health-check a local Ollama server process when needed."""

    base_url: str = "http://127.0.0.1:11434"
    command: tuple[str, ...] = ("ollama", "serve")
    startup_timeout_seconds: float = 20.0
    _process: subprocess.Popen[Any] | None = field(default=None, init=False, repr=False)

    def is_running(self) -> bool:
        url = f"{self.base_url.rstrip('/')}/api/tags"
        req = urllib_request.Request(url, method="GET")
        try:
            with urllib_request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except (error.URLError, error.HTTPError, TimeoutError):
            return False

    def start(self) -> subprocess.Popen[Any]:
        if self._process and self._process.poll() is None:
            return self._process

        self._process = subprocess.Popen(self.command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self._process

    def ensure_running(self, *, start_if_needed: bool = True) -> dict[str, Any]:
        if self.is_running():
            return {"status": "ok", "message": "Ollama server is already running."}

        if not start_if_needed:
            return {"status": "error", "message": "Ollama server is not reachable."}

        self.start()
        deadline = time.time() + self.startup_timeout_seconds
        while time.time() < deadline:
            if self.is_running():
                return {"status": "ok", "message": "Ollama server started.", "started": True}
            time.sleep(0.25)

        return {
            "status": "error",
            "message": "Ollama server failed to become ready before timeout.",
            "started": True,
            "timeout_seconds": self.startup_timeout_seconds,
        }

    def stop(self) -> dict[str, Any]:
        if self._process is None:
            return {"status": "ok", "message": "No managed Ollama process to stop."}

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

        self._process = None
        return {"status": "ok", "message": "Managed Ollama process stopped."}

    def pull_model(self, model_id: str) -> dict[str, Any]:
        normalized = model_id.split("/", 1)[1] if model_id.lower().startswith("ollama/") else model_id
        url = f"{self.base_url.rstrip('/')}/api/pull"
        body = json.dumps({"name": normalized, "stream": False}).encode("utf-8")
        req = urllib_request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=300) as response:
                raw = response.read().decode("utf-8")
                return {"status": "ok", "model": normalized, "raw": json.loads(raw)}
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            return {
                "status": "error",
                "model": normalized,
                "error": {"exception_class": exc.__class__.__name__, "message": str(exc)},
            }

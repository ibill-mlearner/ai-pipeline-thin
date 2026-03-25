"""Hardware acceleration helpers for GPU-backed model execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, util
import json
import subprocess
import sys
from typing import Any


@dataclass
class HardwareAcceleration:
    """Detect, validate, and troubleshoot available GPU hardware acceleration."""

    preferred_vendor: str = "nvidia"
    last_error: str | None = field(default=None, init=False)

    def find_gpu(self) -> str | None:
        """Return the current CUDA GPU name when available, otherwise ``None``."""
        torch_spec = util.find_spec("torch")
        if torch_spec is None:
            self.last_error = "PyTorch is not installed."
            return None

        torch = import_module("torch")

        if not torch.cuda.is_available():
            self.last_error = "CUDA runtime is unavailable."
            return None

        device_index = torch.cuda.current_device()
        self.last_error = None
        return torch.cuda.get_device_name(device_index)

    def is_valid_gpu(self, gpu_name: str | None) -> bool:
        """Return whether the detected GPU matches the preferred vendor."""
        if not gpu_name:
            self.last_error = "No GPU name was detected."
            return False

        is_valid = self.preferred_vendor.lower() in gpu_name.lower()
        if not is_valid:
            self.last_error = (
                f"Detected GPU '{gpu_name}' is not a supported {self.preferred_vendor.upper()} device."
            )
        else:
            self.last_error = None

        return is_valid

    def try_use_gpu(self) -> bool:
        """Try to allocate a small tensor on CUDA and report success/failure."""
        gpu_name = self.find_gpu()
        if not self.is_valid_gpu(gpu_name):
            return False

        # Future work: add a parallel path for AMD Radeon/ROCm acceleration.
        torch = import_module("torch")

        try:
            _ = torch.tensor([1.0], device="cuda")
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = f"GPU allocation failed: {exc}"
            return False

    def missing_requirements(self) -> list[str]:
        """Return missing runtime requirements for CUDA-backed execution."""
        missing: list[str] = []

        if util.find_spec("torch") is None:
            missing.append("torch")

        return missing

    def attempt_dependency_install(self, package: str = "torch") -> dict[str, Any]:
        """Attempt to install a missing dependency with pip and return command results."""
        command = [sys.executable, "-m", "pip", "install", package]
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        return {
            "attempted": True,
            "package": package,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "success": result.returncode == 0,
        }

    def build_rest_response(
        self,
        message: str,
        *,
        status_code: int,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an HTTP/REST-style JSON-compatible payload for integrations."""
        return {
            "content_type": "application/json",
            "status_code": status_code,
            "body": {
                "status": status,
                "message": message,
                "details": details or {},
            },
        }

    def build_rest_response_json(self, payload: dict[str, Any]) -> str:
        """Serialize a REST payload to a JSON string for transport."""
        return json.dumps(payload, indent=2)

    def troubleshoot_gpu(self, attempt_install: bool = False) -> dict[str, Any]:
        """Return a REST-style troubleshooting response for GPU readiness checks."""
        missing = self.missing_requirements()
        if missing:
            details: dict[str, Any] = {
                "missing_requirements": missing,
                "suggested_action": "Install missing Python dependencies before enabling GPU inference.",
            }
            if attempt_install:
                details["install_result"] = self.attempt_dependency_install(missing[0])

            return self.build_rest_response(
                "GPU acceleration is unavailable because required modules are missing.",
                status_code=503,
                status="error",
                details=details,
            )

        gpu_name = self.find_gpu()
        if not self.is_valid_gpu(gpu_name):
            return self.build_rest_response(
                "GPU acceleration is unavailable on this system.",
                status_code=422,
                status="error",
                details={
                    "detected_gpu": gpu_name,
                    "error": self.last_error,
                    "suggested_action": "Verify NVIDIA GPU drivers and CUDA runtime installation.",
                },
            )

        if self.try_use_gpu():
            return self.build_rest_response(
                "GPU acceleration is ready.",
                status_code=200,
                status="ok",
                details={"detected_gpu": gpu_name},
            )

        return self.build_rest_response(
            "GPU detected but initialization failed.",
            status_code=500,
            status="error",
            details={
                "detected_gpu": gpu_name,
                "error": self.last_error,
                "suggested_action": "Reinstall CUDA-compatible torch and validate CUDA toolkit configuration.",
            },
        )

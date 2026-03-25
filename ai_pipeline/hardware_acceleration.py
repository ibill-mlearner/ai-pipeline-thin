"""Hardware acceleration helpers for GPU-backed model execution."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util


@dataclass
class HardwareAcceleration:
    """Detect and validate available GPU hardware for acceleration."""

    preferred_vendor: str = "nvidia"

    def find_gpu(self) -> str | None:
        """Return the current CUDA GPU name when available, otherwise ``None``."""
        torch_spec = util.find_spec("torch")
        if torch_spec is None:
            return None

        import torch

        if not torch.cuda.is_available():
            return None

        device_index = torch.cuda.current_device()
        return torch.cuda.get_device_name(device_index)

    def is_valid_gpu(self, gpu_name: str | None) -> bool:
        """Return whether the detected GPU matches the preferred vendor."""
        if not gpu_name:
            return False

        return self.preferred_vendor.lower() in gpu_name.lower()

    def try_use_gpu(self) -> bool:
        """Try to allocate a small tensor on CUDA and report success/failure."""
        gpu_name = self.find_gpu()
        if not self.is_valid_gpu(gpu_name):
            return False

        # Future work: add a parallel path for AMD Radeon/ROCm acceleration.
        try:
            import torch

            _ = torch.tensor([1.0], device="cuda")
            return True
        except Exception:
            return False

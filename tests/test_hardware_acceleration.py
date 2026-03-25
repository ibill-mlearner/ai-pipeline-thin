"""Tests for hardware acceleration helpers."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from ai_pipeline.hardware_acceleration import HardwareAcceleration


class HardwareAccelerationTests(TestCase):
    def test_is_valid_gpu_requires_preferred_vendor(self) -> None:
        accelerator = HardwareAcceleration()

        self.assertTrue(accelerator.is_valid_gpu("NVIDIA GeForce RTX 4090"))
        self.assertFalse(accelerator.is_valid_gpu("AMD Radeon RX 7900 XTX"))
        self.assertFalse(accelerator.is_valid_gpu(None))

    def test_try_use_gpu_returns_false_when_not_valid(self) -> None:
        accelerator = HardwareAcceleration()

        with patch.object(accelerator, "find_gpu", return_value="AMD Radeon RX 6800"):
            self.assertFalse(accelerator.try_use_gpu())

    def test_try_use_gpu_returns_false_when_tensor_allocation_fails(self) -> None:
        accelerator = HardwareAcceleration()
        fake_torch = SimpleNamespace(tensor=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("CUDA error")))

        with (
            patch.object(accelerator, "find_gpu", return_value="NVIDIA A10"),
            patch.dict("sys.modules", {"torch": fake_torch}),
        ):
            self.assertFalse(accelerator.try_use_gpu())

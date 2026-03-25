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
            patch("ai_pipeline.hardware_acceleration.import_module", return_value=fake_torch),
        ):
            self.assertFalse(accelerator.try_use_gpu())

    def test_troubleshoot_gpu_returns_rest_payload_when_torch_missing(self) -> None:
        accelerator = HardwareAcceleration()

        with patch.object(accelerator, "missing_requirements", return_value=["torch"]):
            payload = accelerator.troubleshoot_gpu(attempt_install=False)

        self.assertEqual(payload["content_type"], "application/json")
        self.assertEqual(payload["status_code"], 503)
        self.assertEqual(payload["body"]["status"], "error")

    def test_build_rest_response_json_serializes_payload(self) -> None:
        accelerator = HardwareAcceleration()

        payload = accelerator.build_rest_response(
            "GPU ready",
            status_code=200,
            status="ok",
            details={"detected_gpu": "NVIDIA A10"},
        )
        serialized = accelerator.build_rest_response_json(payload)

        self.assertIn('"content_type": "application/json"', serialized)
        self.assertIn('"status": "ok"', serialized)

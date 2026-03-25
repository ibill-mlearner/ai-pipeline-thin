"""Tests for model loader argument compatibility."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest import TestCase
from unittest.mock import patch

from ai_pipeline.model_loader import ModelLoader


class ModelLoaderTests(TestCase):
    def test_uses_dtype_when_transformers_supports_it(self) -> None:
        captured_kwargs: dict[str, object] = {}

        class FakeAutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, model_name, dtype=None, device_map=None, cache_dir=None):
                captured_kwargs.update(
                    {
                        "model_name": model_name,
                        "dtype": dtype,
                        "device_map": device_map,
                        "cache_dir": cache_dir,
                    }
                )
                return "fake-model"

        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM

        with (
            patch("ai_pipeline.model_loader.ensure_stdlib_logging_available", return_value=False),
            patch.dict(sys.modules, {"transformers": fake_transformers}),
        ):
            loader = ModelLoader(model_name="demo/model", torch_dtype="auto", device_map="auto")
            result = loader.build()

        self.assertEqual(result, "fake-model")
        self.assertEqual(captured_kwargs["dtype"], "auto")
        self.assertEqual(captured_kwargs["device_map"], "auto")

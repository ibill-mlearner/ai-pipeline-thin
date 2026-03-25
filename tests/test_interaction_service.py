"""Tests for the request contract and interaction service boundaries."""

from unittest import TestCase
from unittest.mock import patch

from ai_pipeline.interaction_service import AIPipelineInteractionService
from ai_pipeline.request import AIPipelineRequest
from ai_pipeline.upstream_error import AIPipelineUpstreamError


class InteractionServiceTests(TestCase):
    def test_resolve_model_id_keeps_request_value_opaque(self) -> None:
        service = AIPipelineInteractionService(default_model_id="provider/default-model")
        request = AIPipelineRequest(prompt="hi", model_id="OpenAI/GPT-4.1-Mini")

        resolved = service.resolve_model_id(
            request,
            session_model_id="session/model",
            configured_default_model_id="config/model",
        )

        self.assertEqual(resolved, "OpenAI/GPT-4.1-Mini")

    def test_execute_wraps_upstream_exception(self) -> None:
        service = AIPipelineInteractionService()
        request = AIPipelineRequest(prompt="hello")

        with patch("ai_pipeline.interaction_service.AIPipeline", side_effect=RuntimeError("boom")):
            with self.assertRaises(AIPipelineUpstreamError) as raised:
                service.execute(request)

        self.assertEqual(raised.exception.details["exception_class"], "RuntimeError")
        self.assertEqual(raised.exception.details["message"], "boom")

    def test_execute_returns_response_payload(self) -> None:
        service = AIPipelineInteractionService(default_system_prompt="sys default")
        request = AIPipelineRequest(
            prompt="",
            system_prompt=None,
            messages=[{"role": "user", "content": "message fallback"}],
            context={"session_id": "abc123"},
            provider="huggingface",
            model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        )

        class FakePipeline:
            def __init__(self, model_name_value, system_content, prompt_value, download_locally):
                self.model_name_value = model_name_value
                self.system_content = system_content
                self.prompt_value = prompt_value
                self.download_locally = download_locally

            @staticmethod
            def build_model():
                return "model"

            @staticmethod
            def build_tokenizer():
                return "tokenizer"

            @staticmethod
            def build_text(tokenizer):
                return f"text:{tokenizer}"

            @staticmethod
            def build_model_inputs(tokenizer, text, model):
                return {"tokenizer": tokenizer, "text": text, "model": model}

            @staticmethod
            def build_raw_generated_ids(model, model_inputs):
                return [1, 2, 3]

            @staticmethod
            def build_generated_ids(model_inputs, raw_generated_ids):
                return raw_generated_ids

            @staticmethod
            def build_response(tokenizer, generated_ids):
                return "ok"

        with patch("ai_pipeline.interaction_service.AIPipeline", FakePipeline):
            payload = service.execute(request)

        self.assertEqual(payload["provider"], "huggingface")
        self.assertEqual(payload["model_id"], "HuggingFaceTB/SmolLM2-360M-Instruct")
        self.assertEqual(payload["response"], "ok")
        self.assertEqual(payload["context"], {"session_id": "abc123"})

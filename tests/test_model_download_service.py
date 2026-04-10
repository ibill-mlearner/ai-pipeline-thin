"""Tests for dedicated model download service behavior."""

from unittest import TestCase
from unittest.mock import patch

from ai_pipeline.model_download_service import AIPipelineModelDownloadService
from ai_pipeline.upstream_error import AIPipelineUpstreamError


class ModelDownloadServiceTests(TestCase):
    def test_download_returns_download_payload(self) -> None:
        service = AIPipelineModelDownloadService()

        class FakeModelLoader:
            def __init__(self, model_name, device_map, torch_dtype, download_locally):
                self.model_name = model_name
                self.device_map = device_map
                self.torch_dtype = torch_dtype
                self.download_locally = download_locally

            @staticmethod
            def build():
                return "model"

        class FakeTokenizerLoader:
            def __init__(self, model_name, download_locally):
                self.model_name = model_name
                self.download_locally = download_locally

            @staticmethod
            def build():
                return "tokenizer"

        with (
            patch("ai_pipeline.model_download_service.ModelLoader", FakeModelLoader),
            patch("ai_pipeline.model_download_service.TokenizerLoader", FakeTokenizerLoader),
        ):
            payload = service.download("Qwen/Qwen2.5-3B-Instruct")

        self.assertEqual(payload["provider"], "huggingface")
        self.assertEqual(payload["model_id"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(payload["status"], "downloaded")

    def test_download_wraps_exception(self) -> None:
        service = AIPipelineModelDownloadService()

        class FailingModelLoader:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("download boom")

        with patch("ai_pipeline.model_download_service.ModelLoader", FailingModelLoader):
            with self.assertRaises(AIPipelineUpstreamError) as raised:
                service.download("broken/model")

        self.assertEqual(raised.exception.details["exception_class"], "RuntimeError")
        self.assertEqual(raised.exception.details["message"], "download boom")
        self.assertEqual(raised.exception.details["model_id"], "broken/model")

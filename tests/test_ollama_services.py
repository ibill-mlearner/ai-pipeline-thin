from __future__ import annotations

import json
from unittest import TestCase
from unittest.mock import MagicMock, patch

from ai_pipeline.ollama_interaction_service import OllamaInteractionService
from ai_pipeline.ollama_health_service import OllamaHealthService
from ai_pipeline.ollama_server_service import OllamaServerService
from ai_pipeline.request import AIPipelineRequest
from ai_pipeline.upstream_error import AIPipelineUpstreamError


class OllamaInteractionServiceTests(TestCase):
    @patch("ai_pipeline.ollama_interaction_service.urllib_request.urlopen")
    def test_execute_returns_normalized_payload(self, mocked_urlopen: MagicMock) -> None:
        fake_response = MagicMock()
        fake_response.read.return_value = json.dumps({"response": "hello"}).encode("utf-8")
        fake_response.__enter__.return_value = fake_response
        mocked_urlopen.return_value = fake_response

        service = OllamaInteractionService(base_url="http://localhost:11434")
        request_payload = AIPipelineRequest(prompt="Hi", provider="ollama")

        result = service.execute(request_payload, configured_default_model_id="ollama/tiny")

        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["model_id"], "ollama/tiny")
        self.assertEqual(result["response"], "hello")

    @patch("ai_pipeline.ollama_interaction_service.urllib_request.urlopen", side_effect=OSError("offline"))
    def test_execute_wraps_transport_errors(self, _: MagicMock) -> None:
        service = OllamaInteractionService(base_url="http://localhost:11434")
        request_payload = AIPipelineRequest(prompt="Hi")

        with self.assertRaises(AIPipelineUpstreamError):
            service.execute(request_payload)


class OllamaServerServiceTests(TestCase):
    @patch("ai_pipeline.ollama_server_service.urllib_request.urlopen")
    def test_is_running_true_on_200(self, mocked_urlopen: MagicMock) -> None:
        fake_response = MagicMock()
        fake_response.status = 200
        fake_response.__enter__.return_value = fake_response
        mocked_urlopen.return_value = fake_response

        service = OllamaServerService()
        self.assertTrue(service.is_running())

    @patch("ai_pipeline.ollama_server_service.time.sleep")
    @patch.object(OllamaServerService, "is_running", side_effect=[False, True])
    @patch.object(OllamaServerService, "start")
    def test_ensure_running_starts_when_missing(
        self,
        mocked_start: MagicMock,
        _: MagicMock,
        __: MagicMock,
    ) -> None:
        service = OllamaServerService(startup_timeout_seconds=1)
        payload = service.ensure_running(start_if_needed=True)

        mocked_start.assert_called_once()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["started"])


class OllamaHealthServiceTests(TestCase):
    @patch("ai_pipeline.ollama_health_service.urllib_request.urlopen")
    def test_status_includes_running_and_models(self, mocked_urlopen: MagicMock) -> None:
        fake_response = MagicMock()
        fake_response.status = 200
        fake_response.read.return_value = json.dumps({"models": [{"name": "llama3.2:1b"}]}).encode("utf-8")
        fake_response.__enter__.return_value = fake_response
        mocked_urlopen.return_value = fake_response

        service = OllamaHealthService()
        payload = service.status()

        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["running"])
        self.assertEqual(len(payload["models"]), 1)

    @patch("ai_pipeline.ollama_health_service.urllib_request.urlopen", side_effect=OSError("offline"))
    def test_status_reports_unreachable_service(self, _: MagicMock) -> None:
        service = OllamaHealthService()
        payload = service.status()

        self.assertEqual(payload["status"], "error")
        self.assertFalse(payload["running"])

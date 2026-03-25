"""Tests for dependency diagnostics wording."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from ai_pipeline.dependency_diagnostics import build_transformers_import_error_details


class DependencyDiagnosticsTests(TestCase):
    def test_reports_logging_shadow_when_not_in_stdlib(self) -> None:
        with (
            patch(
                "ai_pipeline.dependency_diagnostics.util.find_spec",
                side_effect=[
                    SimpleNamespace(origin=r"E:\Tests\Accessibility_AI\AccessBackEnd\app\logging.py"),
                    None,
                ],
            ),
            patch("ai_pipeline.dependency_diagnostics._is_stdlib_logging_path", return_value=False),
        ):
            details = build_transformers_import_error_details(Exception("boom"))

        self.assertIn("shadowing Python's stdlib `logging` package", details)

    def test_includes_version_mismatch_hint_for_get_full_repo_name(self) -> None:
        with (
            patch(
                "ai_pipeline.dependency_diagnostics.util.find_spec",
                side_effect=[
                    None,
                    SimpleNamespace(origin=r"C:\Python312\Lib\site-packages\huggingface_hub\__init__.py"),
                ],
            ),
            patch(
                "ai_pipeline.dependency_diagnostics._safe_package_version",
                side_effect=["4.57.0", "0.19.0"],
            ),
        ):
            details = build_transformers_import_error_details(
                ImportError("cannot import name 'get_full_repo_name' from 'huggingface_hub'")
            )

        self.assertIn("transformers=4.57.0", details)
        self.assertIn("huggingface_hub=0.19.0", details)
        self.assertIn("version mismatch", details)

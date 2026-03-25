"""Tests for ai_pipeline interface help context."""

from unittest import TestCase

from ai_pipeline.interfaces import help_context


class InterfacesTests(TestCase):
    def test_help_context_mentions_major_entry_points(self) -> None:
        context = help_context()

        self.assertIn("AIPipeline", context)
        self.assertIn("AIPipelineInteractionService", context)
        self.assertIn("HardwareAcceleration", context)
        self.assertIn("Treat model_id as opaque", context)

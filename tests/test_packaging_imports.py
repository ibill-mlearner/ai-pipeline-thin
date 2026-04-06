from __future__ import annotations

import ai_pipeline
import ai_pipeline_thin


def test_distribution_compat_imports_expose_pipeline_symbol() -> None:
    assert hasattr(ai_pipeline, "AIPipeline")
    assert hasattr(ai_pipeline_thin, "AIPipeline")

"""Helpers for clearer dependency import diagnostics."""

from __future__ import annotations

from importlib import util
from pathlib import Path


def build_transformers_import_error_details(original_error: Exception) -> str:
    """Create actionable troubleshooting details for transformers import failures."""
    diagnostics: list[str] = []

    logging_spec = util.find_spec("logging")
    if logging_spec and logging_spec.origin:
        logging_origin = Path(logging_spec.origin)
        if logging_origin.name != "__init__.py":
            diagnostics.append(
                "Detected a local module shadowing Python's stdlib `logging` package "
                f"at: {logging_origin}. Rename that file/module (for example `app_logging.py`)."
            )

    huggingface_hub_spec = util.find_spec("huggingface_hub")
    if huggingface_hub_spec and huggingface_hub_spec.origin:
        diagnostics.append(
            "Detected `huggingface_hub` at "
            f"{huggingface_hub_spec.origin}. Ensure it is compatible with your installed "
            "`transformers` version."
        )

    if not diagnostics:
        diagnostics.append(
            "Verify your Python environment has compatible versions of `transformers` and "
            "`huggingface_hub`, and that no local files shadow standard library modules."
        )

    diagnostics_text = " ".join(diagnostics)
    return (
        "Failed to import `transformers`. "
        f"{diagnostics_text} Original error: {original_error!r}"
    )

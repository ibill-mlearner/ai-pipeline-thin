"""Helpers for clearer dependency import diagnostics."""

from __future__ import annotations

from importlib import metadata, util
from pathlib import Path
import sysconfig
import sys


def _safe_package_version(package_name: str) -> str:
    """Return installed package version or a fallback label when unavailable."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _is_within(path: Path, root: Path) -> bool:
    """Check if `path` is under `root` while remaining Python 3.8+ compatible."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _is_stdlib_logging_path(logging_origin: Path) -> bool:
    """Return whether a discovered logging module is the stdlib implementation."""
    stdlib_path = sysconfig.get_paths().get("stdlib")
    if not stdlib_path:
        return False
    return _is_within(logging_origin, Path(stdlib_path))


def ensure_stdlib_logging_available() -> bool:
    """Force-load stdlib logging module when shadowed by a local package."""
    current_logging = sys.modules.get("logging")
    if current_logging is not None and hasattr(current_logging, "StreamHandler"):
        return False

    stdlib_path = sysconfig.get_paths().get("stdlib")
    if not stdlib_path:
        return False

    logging_init = Path(stdlib_path) / "logging" / "__init__.py"
    if not logging_init.exists():
        return False

    spec = util.spec_from_file_location(
        "logging",
        logging_init,
        submodule_search_locations=[str(logging_init.parent)],
    )
    if spec is None or spec.loader is None:
        return False

    module = util.module_from_spec(spec)
    sys.modules["logging"] = module
    spec.loader.exec_module(module)
    return True


def build_transformers_import_error_details(original_error: Exception) -> str:
    """Create actionable troubleshooting details for transformers import failures."""
    diagnostics: list[str] = []

    logging_spec = util.find_spec("logging")
    if logging_spec and logging_spec.origin:
        logging_origin = Path(logging_spec.origin)
        if not _is_stdlib_logging_path(logging_origin):
            diagnostics.append(
                "Detected a local module shadowing Python's stdlib `logging` package "
                f"at: {logging_origin}. Rename that file/module (for example `app_logging.py`)."
            )

    huggingface_hub_spec = util.find_spec("huggingface_hub")
    if huggingface_hub_spec and huggingface_hub_spec.origin:
        transformers_version = _safe_package_version("transformers")
        huggingface_hub_version = _safe_package_version("huggingface_hub")
        diagnostics.append(
            "Detected `huggingface_hub` at "
            f"{huggingface_hub_spec.origin}. Ensure it is compatible with your installed "
            "`transformers` version. Installed versions: "
            f"transformers={transformers_version}, "
            f"huggingface_hub={huggingface_hub_version}."
        )

    original_error_text = str(original_error)
    if "get_full_repo_name" in original_error_text and "huggingface_hub" in original_error_text:
        diagnostics.append(
            "The symbol `get_full_repo_name` missing from `huggingface_hub` usually indicates a "
            "transformers/huggingface_hub version mismatch. Reinstall both together in the same "
            "environment, e.g. `python -m pip install --upgrade \"transformers\" \"huggingface_hub\"`."
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

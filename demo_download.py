"""Demo: download and cache a model/tokenizer without generation."""

from __future__ import annotations

import json
from time import perf_counter

from ai_pipeline import AIPipelineInterface

api = AIPipelineInterface()


def run_download_demo(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
    """Download model artifacts and print a simple status payload."""
    service = api.AIPipelineModelDownloadService()

    started_at = perf_counter()
    payload = service.download(model_id=model_id, provider="huggingface")
    elapsed_seconds = perf_counter() - started_at

    print("=== Download payload ===")
    print(json.dumps(payload, indent=2))
    print(f"Elapsed: {elapsed_seconds:.2f}s")


if __name__ == "__main__":
    run_download_demo()

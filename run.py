import json
import os
from time import perf_counter

import ai_pipeline
from ai_pipeline.interfaces import HardwareAccelerationChecker, ModelBuilder, TokenizerBuilder


def run_single(prompt: str, device_map: str) -> tuple[str, float]:
    """Run a single generation and return response text plus elapsed seconds."""
    pipeline = ai_pipeline.AIPipeline(
        model_name_value="HuggingFaceTB/SmolLM2-360M-Instruct",
        system_content="You are a concise assistant.",
        prompt_value=prompt,
        download_locally=True,
    )

    model_builder: ModelBuilder = pipeline.model_loader
    tokenizer_builder: TokenizerBuilder = pipeline.tokenizer_loader

    pipeline.model_loader.device_map = device_map
    pipeline.model_loader.torch_dtype = "auto"

    started_at = perf_counter()

    model = model_builder.build()
    tokenizer = tokenizer_builder.build()

    text = pipeline.build_text(tokenizer=tokenizer)
    model_inputs = pipeline.build_model_inputs(
        tokenizer=tokenizer,
        text=text,
        model=model,
    )

    raw_ids = pipeline.build_raw_generated_ids(
        model=model,
        model_inputs=model_inputs,
        max_new_tokens=100,
    )

    generated_ids = pipeline.build_generated_ids(
        model_inputs=model_inputs,
        raw_generated_ids=raw_ids,
    )

    response = pipeline.build_response(
        tokenizer=tokenizer,
        generated_ids=generated_ids,
    )

    elapsed_seconds = perf_counter() - started_at
    return response, elapsed_seconds


def run_cpu_then_gpu_demo() -> None:
    """Run two back-to-back interactions to compare CPU vs GPU latency."""
    prompt = "Explain what an AI pipeline is in 2 sentences."

    print("=== ai_pipeline help context ===")
    print(ai_pipeline.help_context())

    print("\n=== Interaction 1/2: CPU baseline (expected slower) ===")
    cpu_response, cpu_elapsed = run_single(prompt=prompt, device_map="cpu")
    print(cpu_response)
    print(f"CPU elapsed: {cpu_elapsed:.2f}s")

    accelerator: HardwareAccelerationChecker = ai_pipeline.HardwareAcceleration()
    gpu_name = accelerator.find_gpu()
    auto_install = os.getenv("AI_PIPELINE_AUTO_INSTALL_GPU_DEPS", "0") == "1"

    print("\n=== Interaction 2/2: GPU attempt (expected faster) ===")
    if accelerator.is_valid_gpu(gpu_name) and accelerator.try_use_gpu():
        print(f"Using GPU: {gpu_name}")
        gpu_response, gpu_elapsed = run_single(prompt=prompt, device_map="auto")
        print(gpu_response)
        print(f"GPU elapsed: {gpu_elapsed:.2f}s")

        speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed else float("inf")
        print(f"Speedup (CPU/GPU): {speedup:.2f}x")
    else:
        payload = accelerator.troubleshoot_gpu(attempt_install=auto_install)
        print("GPU run skipped with troubleshooting payload:")
        print(accelerator.build_rest_response_json(payload))

    available_models = ai_pipeline.AvailableModels().build()
    print("\nAvailable model aliases:")
    print(json.dumps(available_models, indent=2))


if __name__ == "__main__":
    run_cpu_then_gpu_demo()

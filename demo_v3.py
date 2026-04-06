import json
import os
from time import perf_counter

from ai_pipeline import AIPipelineInterface
from ai_pipeline.interfaces import HardwareAccelerationChecker, ModelBuilder, TokenizerBuilder

# We instantiate one gateway object so the rest of this script can access
# every exported class through a single entry point. This keeps imports small
# and makes it obvious where all runtime objects come from in the demo.
api = AIPipelineInterface()


def run_single(prompt: str, device_map: str) -> tuple[str, float]:
    """Run a single generation and return response text plus elapsed seconds."""
    # This pipeline object carries the model name, system prompt, and user prompt.
    # We create it once per interaction so the full run is isolated and easy to time.
    # Using download_locally=True keeps the demo behavior predictable across runs.
    pipeline = api.AIPipeline(
        model_name_value="HuggingFaceTB/SmolLM2-360M-Instruct",
        system_content="You are a concise assistant.",
        prompt_value=prompt,
        download_locally=True,
    )

    # These typed builder handles make each stage explicit for demo readability.
    # Instead of calling one hidden method, we intentionally step through the full
    # pipeline so people can see where model and tokenizer work actually happens.
    model_builder: ModelBuilder = pipeline.model_loader
    tokenizer_builder: TokenizerBuilder = pipeline.tokenizer_loader

    # device_map controls where inference is placed (CPU baseline vs GPU attempt).
    # torch_dtype="auto" lets the backend choose a safe dtype for the selected device.
    # We set both values before building to keep loader behavior consistent.
    pipeline.model_loader.device_map = device_map
    pipeline.model_loader.torch_dtype = "auto"

    # We start timing just before build/execute to capture practical runtime cost.
    # This includes model setup, tokenization, generation, and decode stages.
    # The elapsed value is later used to compare CPU and GPU speed.
    started_at = perf_counter()

    # Building model/tokenizer is separated so each dependency-heavy step is obvious.
    # In real projects these may be cached, but here we keep it explicit for learning.
    # This makes the demo easier to map to the package architecture.
    model = model_builder.build()
    tokenizer = tokenizer_builder.build()

    # Text rendering applies chat-template logic and final prompt formatting.
    # Model inputs then convert text into tensors the model can consume.
    # Keeping this as two steps clarifies where prompt shaping ends and tensor prep begins.
    text = pipeline.build_text(tokenizer=tokenizer)
    model_inputs = pipeline.build_model_inputs(
        tokenizer=tokenizer,
        text=text,
        model=model,
    )

    # This step performs generation and returns raw token IDs from the model.
    # We cap max_new_tokens for demo safety so output size and latency stay reasonable.
    # That prevents runaway generation in quick local runs.
    raw_ids = pipeline.build_raw_generated_ids(
        model=model,
        model_inputs=model_inputs,
        max_new_tokens=100,
    )

    # Generated IDs are normalized relative to prompt inputs before decoding.
    # This keeps output extraction consistent when generation APIs return full sequences.
    # The normalized IDs are then ready for text decoding.
    generated_ids = pipeline.build_generated_ids(
        model_inputs=model_inputs,
        raw_generated_ids=raw_ids,
    )

    # Final decode converts generated IDs into the human-readable response string.
    # We return both the response and elapsed seconds so callers can compare modes.
    # This keeps run_single reusable for both CPU and GPU passes.
    response = pipeline.build_response(
        tokenizer=tokenizer,
        generated_ids=generated_ids,
    )

    elapsed_seconds = perf_counter() - started_at
    return response, elapsed_seconds


def run_cpu_then_gpu_demo() -> None:
    """Run two back-to-back interactions to compare CPU vs GPU latency."""
    # A fixed prompt makes CPU/GPU comparison easier to reason about.
    # Reusing the same input avoids skew from prompt complexity differences.
    # This keeps the demo focused on execution path, not prompt variance.
    prompt = "Explain what an AI pipeline is in 2 sentences."

    # help_context prints package-level guidance so the demo is self-describing.
    # This is useful when someone runs the file without reading repository docs first.
    # It provides quick orientation on major entry points.
    print("=== ai_pipeline help context ===")
    print(api.help_context())

    # These background jobs intentionally simulate parallel chat starts.
    # They are launched and ignored to demonstrate async submission behavior only.
    # The main CPU/GPU interaction path continues independently of these jobs.
    background_jobs = api.InteractionJobManager(max_workers=2)
    background_jobs.submit(lambda: {"chat": "conversation-a-started"})
    background_jobs.submit(lambda: {"chat": "conversation-b-started"})

    # CPU first gives us a baseline latency number for the same prompt.
    # We print both response and elapsed time so the baseline is easy to inspect.
    # Later GPU timing can be compared directly against this value.
    print("\n=== Interaction 1/2: CPU baseline (expected slower) ===")
    cpu_response, cpu_elapsed = run_single(prompt=prompt, device_map="cpu")
    print(cpu_response)
    print(f"CPU elapsed: {cpu_elapsed:.2f}s")

    # HardwareAcceleration encapsulates GPU detection and quick viability checks.
    # We read optional env config so users can choose whether dependency auto-install
    # attempts should run during troubleshooting.
    accelerator: HardwareAccelerationChecker = api.HardwareAcceleration()
    gpu_name = accelerator.find_gpu()
    auto_install = os.getenv("AI_PIPELINE_AUTO_INSTALL_GPU_DEPS", "0") == "1"

    # This branch tries GPU execution only when the checks say the environment is ready.
    # If successful, we compute a simple CPU/GPU speedup ratio for quick comparison.
    # If not, we emit a structured troubleshooting payload instead of failing silently.
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

    # We finish by printing available model aliases for quick discoverability.
    # This makes it easy to switch demo model IDs without searching code internals.
    # The JSON format keeps output readable for copy/paste use.
    available_models = api.AvailableModels().build()
    print("\nAvailable model aliases:")
    print(json.dumps(available_models, indent=2))


if __name__ == "__main__":
    # Optional local test command (copy/paste only if you want this in your script).
    # It is intentionally commented so demo execution stays focused on runtime behavior.
    # Uncomment only when you want a quick pre-run validation from this file context.
    # import subprocess
    # subprocess.run(["pytest", "-q"], check=True)
    run_cpu_then_gpu_demo()

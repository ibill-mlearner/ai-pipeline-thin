"""Simple test runner for the ai_pipeline module with multiple config demos."""

import ai_pipeline


def run_pipeline(config: dict) -> None:
    pipeline = ai_pipeline.AIPipeline(
        model_name_value=config["model_name"],
        system_content=config["system_content"],
        prompt_value=config["prompt"],
        download_locally=config["download_locally"],
    )

    # Keep existing pipeline classes/settings; this encourages GPU usage when available.
    pipeline.model_loader.device_map = "auto"
    pipeline.model_loader.torch_dtype = "auto"

    model = pipeline.build_model()
    tokenizer = pipeline.build_tokenizer()

    text = pipeline.build_text(tokenizer=tokenizer)
    model_inputs = pipeline.build_model_inputs(tokenizer=tokenizer, text=text, model=model)

    raw_generated_ids = pipeline.build_raw_generated_ids(
        model=model,
        model_inputs=model_inputs,
        max_new_tokens=config["max_new_tokens"],
    )
    generated_ids = pipeline.build_generated_ids(
        model_inputs=model_inputs,
        raw_generated_ids=raw_generated_ids,
    )

    response = pipeline.build_response(tokenizer=tokenizer, generated_ids=generated_ids)

    print(f"\n=== Config: {config['name']} ===")
    print(f"Model: {config['model_name']}")
    print(f"Model device: {model.device}")
    print(response)


def main() -> None:
    demo_configs = [
        {
            "name": "creative-assistant",
            "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "system_content": "You are a concise creative assistant.",
            "prompt": "Write a two-line sci-fi story about teamwork.",
            "download_locally": True,
            "max_new_tokens": 96,
        },
        {
            "name": "technical-explainer",
            "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "system_content": "You explain technical topics for beginners.",
            "prompt": "Explain what a tokenizer does in 3 bullet points.",
            "download_locally": False,
            "max_new_tokens": 128,
        },
        {
            "name": "structured-output",
            "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "system_content": "Return valid JSON only.",
            "prompt": "Give me JSON with keys: title, summary, and tags for AI pipelines.",
            "download_locally": True,
            "max_new_tokens": 80,
        },
    ]

    for config in demo_configs:
        run_pipeline(config)


if __name__ == "__main__":
    main()

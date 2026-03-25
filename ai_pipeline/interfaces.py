"""Interface contracts and help context for loose coupling across `ai_pipeline`.

The protocols in this module are intentionally lightweight and describe
behavioral boundaries instead of concrete implementations. Applications can
rely on these protocols to keep API adapters thin and avoid tight coupling.
"""

from __future__ import annotations

from typing import Any, Protocol


class NameProvider(Protocol):
    """Provides a model identifier string."""

    def get(self) -> str: ...


class PromptProvider(Protocol):
    """Provides a prompt text string."""

    def get(self) -> str: ...


class MessagesBuilder(Protocol):
    """Builds a role/content message list for chat-template tokenizers."""

    def build(self) -> list[dict[str, str]]: ...


class ModelBuilder(Protocol):
    """Builds and returns a model object for generation."""

    def build(self) -> Any: ...


class TokenizerBuilder(Protocol):
    """Builds and returns a tokenizer compatible with the selected model."""

    def build(self) -> Any: ...


class TextBuilder(Protocol):
    """Builds rendered input text from structured messages."""

    def build(self) -> str: ...


class ModelInputsBuilder(Protocol):
    """Builds model-ready tensor inputs from text/tokenizer/model metadata."""

    def build(self) -> Any: ...


class IdsBuilder(Protocol):
    """Builds generated token IDs or derived ID collections."""

    def build(self) -> Any: ...


class ResponseBuilder(Protocol):
    """Builds final decoded response text."""

    def build(self) -> str: ...


class InteractionExecutor(Protocol):
    """Executes one AI interaction from request payload to response payload."""

    def execute(self, request: Any, **kwargs: Any) -> dict[str, Any]: ...


class HardwareAccelerationChecker(Protocol):
    """Checks and troubleshoots GPU readiness for accelerated inference."""

    def find_gpu(self) -> str | None: ...

    def is_valid_gpu(self, gpu_name: str | None) -> bool: ...

    def try_use_gpu(self) -> bool: ...
    def troubleshoot_gpu(self, attempt_install: bool = False) -> dict[str, Any]: ...

    def build_rest_response_json(self, payload: dict[str, Any]) -> str: ...


def help_context() -> str:
    """Return high-level usage guidance for core `ai_pipeline` entry points."""
    return (
        "ai_pipeline high-level guide\n"
        "\n"
        "Primary entry points:\n"
        "- AIPipeline: low-level staged builders (model, tokenizer, inputs, generation, response).\n"
        "- AIPipelineInteractionService: thin adapter boundary for request execution and error wrapping.\n"
        "- HardwareAcceleration: GPU detection/validation/troubleshooting helper for CUDA readiness.\n"
        "\n"
        "Loose-coupling interfaces:\n"
        "- NameProvider / PromptProvider: simple value providers.\n"
        "- MessagesBuilder: creates role/content message structures.\n"
        "- ModelBuilder / TokenizerBuilder: isolate dependency-heavy object creation.\n"
        "- TextBuilder / ModelInputsBuilder / IdsBuilder / ResponseBuilder: staged generation pipeline.\n"
        "- InteractionExecutor: request-to-response orchestration contract.\n"
        "- HardwareAccelerationChecker: acceleration capability contract.\n"
        "\n"
        "Adapter guidance:\n"
        "- Treat model_id as opaque outside pipeline internals.\n"
        "- Keep adapter-level error handling to one top-level wrap/rethrow.\n"
        "- Let pipeline modules own model loading and model-id interpretation."
    )

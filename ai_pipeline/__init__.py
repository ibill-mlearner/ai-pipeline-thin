"""Composable class-based AI inference pipeline.

This module intentionally uses lazy exports so importing
`ai_pipeline_thin.ai_pipeline` does not require optional heavy dependencies
(such as `transformers`) until those symbols are actually used.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .generate_ids import GenerateIds
    from .generated_ids import GeneratedIds
    from .interfaces import (
        IdsBuilder,
        MessagesBuilder,
        ModelBuilder,
        ModelInputsBuilder,
        NameProvider,
        PromptProvider,
        ResponseBuilder,
        TextBuilder,
        TokenizerBuilder,
    )
    from .messages import Messages
    from .model_inputs import ModelInputs
    from .model_loader import ModelLoader
    from .model_name import ModelName
    from .pipeline import AIPipeline
    from .prompt import Prompt
    from .response import Response
    from .text import Text
    from .tokenizer_loader import TokenizerLoader

__all__ = [
    "AIPipeline",
    "GenerateIds",
    "GeneratedIds",
    "IdsBuilder",
    "Messages",
    "MessagesBuilder",
    "ModelBuilder",
    "ModelInputs",
    "ModelInputsBuilder",
    "ModelLoader",
    "ModelName",
    "NameProvider",
    "Prompt",
    "PromptProvider",
    "Response",
    "ResponseBuilder",
    "Text",
    "TextBuilder",
    "TokenizerBuilder",
    "TokenizerLoader",
]

_EXPORT_MAP = {
    "AIPipeline": (".pipeline", "AIPipeline"),
    "GenerateIds": (".generate_ids", "GenerateIds"),
    "GeneratedIds": (".generated_ids", "GeneratedIds"),
    "IdsBuilder": (".interfaces", "IdsBuilder"),
    "Messages": (".messages", "Messages"),
    "MessagesBuilder": (".interfaces", "MessagesBuilder"),
    "ModelBuilder": (".interfaces", "ModelBuilder"),
    "ModelInputs": (".model_inputs", "ModelInputs"),
    "ModelInputsBuilder": (".interfaces", "ModelInputsBuilder"),
    "ModelLoader": (".model_loader", "ModelLoader"),
    "ModelName": (".model_name", "ModelName"),
    "NameProvider": (".interfaces", "NameProvider"),
    "Prompt": (".prompt", "Prompt"),
    "PromptProvider": (".interfaces", "PromptProvider"),
    "Response": (".response", "Response"),
    "ResponseBuilder": (".interfaces", "ResponseBuilder"),
    "Text": (".text", "Text"),
    "TextBuilder": (".interfaces", "TextBuilder"),
    "TokenizerBuilder": (".interfaces", "TokenizerBuilder"),
    "TokenizerLoader": (".tokenizer_loader", "TokenizerLoader"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

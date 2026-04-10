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
        HardwareAccelerationChecker,
        IdsBuilder,
        InteractionExecutor,
        MessagesBuilder,
        ModelBuilder,
        ModelInputsBuilder,
        NameProvider,
        PromptProvider,
        ResponseBuilder,
        TextBuilder,
        TokenizerBuilder,
        help_context,
    )
    from .messages import Messages
    from .model_inputs import ModelInputs
    from .interaction_service import AIPipelineInteractionService
    from .model_download_service import AIPipelineModelDownloadService
    from .ollama_interaction_service import OllamaInteractionService
    from .ollama_health_service import OllamaHealthService
    from .ollama_server_service import OllamaServerService
    from .interface_gateway import AIPipelineInterface
    from .interaction_job_manager import InteractionJobManager
    from .interaction_job_record import InteractionJobRecord
    from .interaction_job_manager_registry import InteractionJobManagerRegistry
    from .interaction_persistence_runner import InteractionPersistenceRunner
    from .request import AIPipelineRequest
    from .upstream_error import AIPipelineUpstreamError
    from .hardware_acceleration import HardwareAcceleration
    from .model_loader import ModelLoader
    from .model_name import ModelName
    from .pipeline import AIPipeline
    from .available_models import AvailableModels
    from .prompt import Prompt
    from .response import Response
    from .text import Text
    from .tokenizer_loader import TokenizerLoader

__all__ = [
    "AIPipeline",
    "AvailableModels",
    "GenerateIds",
    "HardwareAcceleration",
    "GeneratedIds",
    "HardwareAccelerationChecker",
    "AIPipelineInteractionService",
    "AIPipelineModelDownloadService",
    "OllamaInteractionService",
    "OllamaHealthService",
    "OllamaServerService",
    "AIPipelineInterface",
    "InteractionJobManager",
    "InteractionJobRecord",
    "InteractionJobManagerRegistry",
    "InteractionPersistenceRunner",
    "AIPipelineRequest",
    "AIPipelineUpstreamError",
    "IdsBuilder",
    "InteractionExecutor",
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
    "help_context",
    "TokenizerLoader",
]

_EXPORT_MAP = {
    "AIPipeline": (".pipeline", "AIPipeline"),
    "AvailableModels": (".available_models", "AvailableModels"),
    "GenerateIds": (".generate_ids", "GenerateIds"),
    "HardwareAcceleration": (".hardware_acceleration", "HardwareAcceleration"),
    "GeneratedIds": (".generated_ids", "GeneratedIds"),
    "HardwareAccelerationChecker": (".interfaces", "HardwareAccelerationChecker"),
    "AIPipelineInteractionService": (".interaction_service", "AIPipelineInteractionService"),
    "AIPipelineModelDownloadService": (".model_download_service", "AIPipelineModelDownloadService"),
    "OllamaInteractionService": (".ollama_interaction_service", "OllamaInteractionService"),
    "OllamaHealthService": (".ollama_health_service", "OllamaHealthService"),
    "OllamaServerService": (".ollama_server_service", "OllamaServerService"),
    "AIPipelineInterface": (".interface_gateway", "AIPipelineInterface"),
    "InteractionJobManager": (".interaction_job_manager", "InteractionJobManager"),
    "InteractionJobRecord": (".interaction_job_record", "InteractionJobRecord"),
    "InteractionJobManagerRegistry": (".interaction_job_manager_registry", "InteractionJobManagerRegistry"),
    "InteractionPersistenceRunner": (".interaction_persistence_runner", "InteractionPersistenceRunner"),
    "AIPipelineRequest": (".request", "AIPipelineRequest"),
    "AIPipelineUpstreamError": (".upstream_error", "AIPipelineUpstreamError"),
    "IdsBuilder": (".interfaces", "IdsBuilder"),
    "InteractionExecutor": (".interfaces", "InteractionExecutor"),
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
    "help_context": (".interfaces", "help_context"),
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

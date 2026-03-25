"""Interface contracts for loose coupling across ai_pipeline components."""

from __future__ import annotations

from typing import Any, Protocol


class NameProvider(Protocol):
    def get(self) -> str: ...


class PromptProvider(Protocol):
    def get(self) -> str: ...


class MessagesBuilder(Protocol):
    def build(self) -> list[dict[str, str]]: ...


class ModelBuilder(Protocol):
    def build(self) -> Any: ...


class TokenizerBuilder(Protocol):
    def build(self) -> Any: ...


class TextBuilder(Protocol):
    def build(self) -> str: ...


class ModelInputsBuilder(Protocol):
    def build(self) -> Any: ...


class IdsBuilder(Protocol):
    def build(self) -> Any: ...


class ResponseBuilder(Protocol):
    def build(self) -> str: ...

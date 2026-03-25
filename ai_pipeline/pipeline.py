"""Facade object that exposes all pipeline components from a single instance."""

from dataclasses import dataclass

from .generate_ids import GenerateIds
from .generated_ids import GeneratedIds
from .interfaces import MessagesBuilder, ModelBuilder, NameProvider, PromptProvider, TokenizerBuilder
from .messages import Messages
from .model_inputs import ModelInputs
from .model_loader import ModelLoader
from .model_name import ModelName
from .prompt import Prompt
from .response import Response
from .text import Text
from .tokenizer_loader import TokenizerLoader


@dataclass
class AIPipeline:
    """Single access object for all configurable AI pipeline stages."""

    model_name_value: str = "Qwen/Qwen2.5-3B-Instruct"
    system_content: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    prompt_value: str = "Give me a short introduction to large language model."
    download_locally: bool = True

    def __post_init__(self) -> None:
        self.model_name: NameProvider = ModelName(value=self.model_name_value)
        self.model_loader: ModelBuilder = ModelLoader(
            model_name=self.model_name.get(),
            download_locally=self.download_locally,
        )
        self.tokenizer_loader: TokenizerBuilder = TokenizerLoader(
            model_name=self.model_name.get(),
            download_locally=self.download_locally,
        )
        self.prompt: PromptProvider = Prompt(value=self.prompt_value)
        self.messages: MessagesBuilder = Messages(
            system_content=self.system_content,
            prompt=self.prompt.get(),
        )

    def build_model(self):
        return self.model_loader.build()

    def build_tokenizer(self):
        return self.tokenizer_loader.build()

    def build_text(self, tokenizer):
        return Text(tokenizer=tokenizer, messages=self.messages.build()).build()

    def build_model_inputs(self, tokenizer, text, model):
        return ModelInputs(tokenizer=tokenizer, text=text, model=model).build()

    def build_raw_generated_ids(self, model, model_inputs, max_new_tokens: int = 512):
        return GenerateIds(model=model, model_inputs=model_inputs, max_new_tokens=max_new_tokens).build()

    def build_generated_ids(self, model_inputs, raw_generated_ids):
        return GeneratedIds(model_inputs=model_inputs, generated_ids=raw_generated_ids).build()

    def build_response(self, tokenizer, generated_ids):
        return Response(tokenizer=tokenizer, generated_ids=generated_ids).build()

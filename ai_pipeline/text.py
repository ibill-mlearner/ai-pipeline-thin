"""Chat-template text abstraction for the AI pipeline."""

from dataclasses import dataclass


@dataclass
class Text:
    """Renders tokenizer chat-template text from message dictionaries."""

    tokenizer: object
    messages: list[dict[str, str]]
    tokenize: bool = False
    add_generation_prompt: bool = True

    def build(self) -> str:
        """Return the chat-formatted text string."""
        return self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=self.tokenize,
            add_generation_prompt=self.add_generation_prompt,
        )

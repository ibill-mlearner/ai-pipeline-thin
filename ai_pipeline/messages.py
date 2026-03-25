"""Chat messages abstraction for the AI pipeline."""

from dataclasses import dataclass, field


@dataclass
class Messages:
    """Builds the chat messages list for a chat-template tokenizer."""

    system_content: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    prompt: str = ""
    items: list[dict[str, str]] = field(default_factory=list)

    def build(self) -> list[dict[str, str]]:
        """Construct and return the final list of role/content dictionaries."""
        base_items = [
            {"role": "system", "content": self.system_content},
            {"role": "user", "content": self.prompt},
        ]
        self.items = base_items if not self.items else self.items
        return self.items

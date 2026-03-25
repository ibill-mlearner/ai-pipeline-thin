"""Response decoding abstraction for the AI pipeline."""

from dataclasses import dataclass


@dataclass
class Response:
    """Decodes generated IDs into a response string."""

    tokenizer: object
    generated_ids: object
    skip_special_tokens: bool = True

    def build(self) -> str:
        """Return decoded model response text."""
        return self.tokenizer.batch_decode(
            self.generated_ids,
            skip_special_tokens=self.skip_special_tokens,
        )[0]

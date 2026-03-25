"""Generation abstraction for the AI pipeline."""

from dataclasses import dataclass


@dataclass
class GenerateIds:
    """Runs model generation and returns token ids."""

    model: object
    model_inputs: object
    max_new_tokens: int = 512

    def build(self):
        """Generate and return raw output token IDs."""
        return self.model.generate(
            **self.model_inputs,
            max_new_tokens=self.max_new_tokens,
        )

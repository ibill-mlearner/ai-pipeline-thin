"""Model inputs abstraction for the AI pipeline."""

from dataclasses import dataclass


@dataclass
class ModelInputs:
    """Tokenizes text and moves tensors to the model device."""

    tokenizer: object
    text: str
    model: object
    return_tensors: str = "pt"

    def build(self):
        """Return the tokenized model inputs."""
        return self.tokenizer([self.text], return_tensors=self.return_tensors).to(self.model.device)

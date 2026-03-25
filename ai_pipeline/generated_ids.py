"""Post-processed generation IDs abstraction for the AI pipeline."""

from dataclasses import dataclass


@dataclass
class GeneratedIds:
    """Removes prompt tokens from generated token IDs."""

    model_inputs: object
    generated_ids: object

    def build(self):
        """Return generated IDs without the original input prefix."""
        return [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(self.model_inputs.input_ids, self.generated_ids)
        ]

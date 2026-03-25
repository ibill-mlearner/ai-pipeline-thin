"""Local model inventory helper for the AI pipeline."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AvailableModels:
    """Collects locally cached model identifiers grouped by provider."""

    models_dir: Path | None = None

    def _models_dir(self) -> Path:
        if self.models_dir is not None:
            return self.models_dir
        return Path(__file__).resolve().parent / "models"

    def _iter_cached_model_ids(self) -> set[str]:
        models_dir = self._models_dir()
        if not models_dir.exists():
            return set()

        model_ids: set[str] = set()

        # Hugging Face cache entries look like: models--provider--model-name
        for entry in models_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue

            encoded_name = entry.name.removeprefix("models--")
            name_parts = encoded_name.split("--")
            if len(name_parts) < 2:
                continue

            provider = name_parts[0]
            model_name = "--".join(name_parts[1:])
            model_ids.add(f"{provider}/{model_name}")

        return model_ids

    def build(self) -> dict[str, list[str]]:
        """Return locally available models grouped by provider."""
        grouped: defaultdict[str, list[str]] = defaultdict(list)

        for model_id in sorted(self._iter_cached_model_ids()):
            provider, model_name = model_id.split("/", maxsplit=1)
            grouped[provider].append(model_name)

        return {provider: sorted(models) for provider, models in sorted(grouped.items())}

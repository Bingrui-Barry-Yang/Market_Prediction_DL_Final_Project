from dataclasses import dataclass
from typing import Protocol


class LLMAdapter(Protocol):
    model_name: str

    def complete(self, prompt: str) -> str:
        """Return a raw model response for a prompt."""


@dataclass(frozen=True)
class NotConfiguredAdapter:
    model_name: str

    def complete(self, prompt: str) -> str:
        raise RuntimeError(
            f"{self.model_name} adapter is not configured yet. "
            "Wire this model through a provider-specific adapter before live runs."
        )


def build_default_adapters(model_names: list[str]) -> list[LLMAdapter]:
    return [NotConfiguredAdapter(model_name=name) for name in model_names]

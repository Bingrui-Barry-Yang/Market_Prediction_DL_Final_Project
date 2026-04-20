from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationScore:
    source: str
    article_count: int
    directional_accuracy: float

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptCandidate:
    version: str
    template: str
    score: float


def select_best_candidate(candidates: list[PromptCandidate]) -> PromptCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.score)

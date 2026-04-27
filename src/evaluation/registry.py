"""Registry of task models and prompt sources for the test-set evaluation.

Hardcodes the 4 valid GEPA runs (Stage 1 outputs). Smoke runs are not listed
here, so they're naturally excluded from the test plan.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskModel:
    slug: str
    litellm_id: str


@dataclass(frozen=True)
class SourceModel:
    slug: str
    litellm_id: str
    run_dir: Path
    result_path: Path


@dataclass(frozen=True)
class PromptCandidate:
    source_slug: str
    idx: int
    text: str
    sha256: str
    is_seed: bool
    is_best: bool


REPO_ROOT = Path(__file__).resolve().parents[2]
GEPA_RUNS_DIR = REPO_ROOT / "outputs" / "gepa_runs" / "bitcoin_sentiment"


TASK_MODELS: list[TaskModel] = [
    TaskModel("claude_sonnet46", "anthropic/claude-sonnet-4-6"),
    TaskModel("gptoss120b", "ollama_chat/gpt-oss:120b"),
    TaskModel("qwen36", "ollama_chat/qwen3.6:latest"),
    TaskModel("gemma4e2b", "ollama_chat/gemma4:e2b"),
]


SOURCE_MODELS: list[SourceModel] = [
    SourceModel(
        slug="claude_sonnet46",
        litellm_id="anthropic/claude-sonnet-4-6",
        run_dir=GEPA_RUNS_DIR / "claude_sonnet46",
        result_path=GEPA_RUNS_DIR / "gepa_result_claude_sonnet46.json",
    ),
    SourceModel(
        slug="gptoss120b",
        litellm_id="ollama_chat/gpt-oss:120b",
        run_dir=GEPA_RUNS_DIR / "run_gptoss120b_b150",
        result_path=GEPA_RUNS_DIR / "result_gptoss120b_b150.json",
    ),
    SourceModel(
        slug="qwen36",
        litellm_id="ollama_chat/qwen3.6:latest",
        run_dir=GEPA_RUNS_DIR / "run_qwen36_b150",
        result_path=GEPA_RUNS_DIR / "result_qwen36_b150.json",
    ),
    SourceModel(
        slug="gemma4e2b",
        litellm_id="ollama_chat/gemma4:e2b",
        run_dir=GEPA_RUNS_DIR / "run_gemma4e2b_b150",
        result_path=GEPA_RUNS_DIR / "result_gemma4e2b_b150.json",
    ),
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_candidates(source: SourceModel) -> list[PromptCandidate]:
    """Read candidates.json + result_*.json and tag seed/best per candidate."""
    cand_path = source.run_dir / "candidates.json"
    with cand_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{cand_path} must be a JSON list of candidates")

    with source.result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    best_text = result.get("best_candidate", {}).get("system_prompt", "")

    out: list[PromptCandidate] = []
    for idx, entry in enumerate(raw):
        text = entry["system_prompt"]
        out.append(
            PromptCandidate(
                source_slug=source.slug,
                idx=idx,
                text=text,
                sha256=_sha256(text),
                is_seed=(idx == 0),
                is_best=(text == best_text),
            )
        )
    return out


def iter_prompts() -> list[PromptCandidate]:
    """All 38 prompts across the 4 source models, in registry order."""
    out: list[PromptCandidate] = []
    for src in SOURCE_MODELS:
        out.extend(load_candidates(src))
    return out


def get_task_model(slug: str) -> TaskModel:
    for t in TASK_MODELS:
        if t.slug == slug:
            return t
    raise KeyError(f"unknown task model slug: {slug}")


def get_source_model(slug: str) -> SourceModel:
    for s in SOURCE_MODELS:
        if s.slug == slug:
            return s
    raise KeyError(f"unknown source model slug: {slug}")

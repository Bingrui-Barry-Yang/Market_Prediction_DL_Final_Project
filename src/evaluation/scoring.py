"""Test-set scoring helpers.

Mirrors the parsing and partial-credit semantics of the GEPA training evaluator
(scripts/run_gepa.py:SentimentScoreEvaluator) so test-set numbers and training
numbers stay directly comparable.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedScore:
    pred_score: int | None
    parse_status: str
    parse_error: str | None


def parse_score_response(raw: str) -> ParsedScore:
    """Strip optional markdown fences and parse `{"score": <int 1-15>}`.

    Matches scripts/run_gepa.py:SentimentScoreEvaluator parsing exactly.
    """
    if raw is None:
        return ParsedScore(None, "failed", "empty response")

    body = raw.strip()
    if body.startswith("```"):
        parts = body.split("```")
        if len(parts) >= 2:
            body = parts[1]
            if body.startswith("json"):
                body = body[4:]
            body = body.strip()

    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        return ParsedScore(None, "failed", f"json decode: {e}")

    if not isinstance(obj, dict) or "score" not in obj:
        return ParsedScore(None, "failed", "missing 'score' key")

    try:
        pred = int(obj["score"])
    except (TypeError, ValueError) as e:
        return ParsedScore(None, "failed", f"score not an int: {e}")

    if pred not in range(1, 16):
        return ParsedScore(None, "failed", f"score {pred} out of range 1-15")

    return ParsedScore(pred, "ok", None)


def gepa_partial_credit(pred: int | None, gold: int) -> float:
    """Same scale as SentimentScoreEvaluator: 1.0 / 0.75 / 0.5 / 0.25 / 0.0."""
    if pred is None:
        return 0.0
    diff = abs(pred - gold)
    if diff == 0:
        return 1.0
    return {1: 0.75, 2: 0.5, 3: 0.25}.get(diff, 0.0)


def direction_band(score: int) -> str:
    """3-band split matching the gold scale."""
    if 1 <= score <= 5:
        return "bearish"
    if 6 <= score <= 10:
        return "neutral"
    if 11 <= score <= 15:
        return "bullish"
    raise ValueError(f"score {score} not in 1-15")


def build_article_input(row: dict[str, Any]) -> str:
    """Same shape used during GEPA training (scripts/run_gepa.py:build_article_input)."""
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    if title and text:
        return f"Title: {title}\n\nArticle text:\n{text}"
    if title:
        return f"Title: {title}"
    return text


_RETRYABLE_NAMES = (
    "RateLimitError",
    "Timeout",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
)


def litellm_completion_with_retries(
    model_name: str,
    messages: list[dict[str, str]],
    *,
    num_retries: int = 8,
    base_backoff: float = 2.0,
    max_backoff: float = 30.0,
    timeout: int = 60,
    temperature: float = 0.0,
) -> tuple[Any, int]:
    """Call litellm.completion with hand-rolled retries on transient errors.

    Returns (response, retries_used) so the caller can log retry counts.
    """
    import litellm

    retryable = tuple(
        getattr(litellm, name) for name in _RETRYABLE_NAMES if hasattr(litellm, name)
    )

    last_err: BaseException | None = None
    for attempt in range(num_retries + 1):
        try:
            response = litellm.completion(
                model=model_name,
                messages=list(messages),
                timeout=timeout,
                temperature=temperature,
            )
            return response, attempt
        except retryable as e:
            last_err = e
            if attempt >= num_retries:
                break
            backoff = min(base_backoff * (2 ** attempt), max_backoff)
            backoff += random.uniform(0, 2.0)
            print(
                f"[WARN] {type(e).__name__} on attempt "
                f"{attempt + 1}/{num_retries + 1}; sleeping {backoff:.1f}s. "
                f"Details: {str(e)[:160]}",
                flush=True,
            )
            time.sleep(backoff)

    assert last_err is not None
    raise last_err


def extract_token_counts(response: Any) -> tuple[int | None, int | None]:
    """Best-effort token usage extraction (LiteLLM returns OpenAI-style usage)."""
    try:
        usage = getattr(response, "usage", None) or response.get("usage")
    except Exception:
        return None, None
    if usage is None:
        return None, None
    try:
        if isinstance(usage, dict):
            tin = usage.get("prompt_tokens")
            tout = usage.get("completion_tokens")
        else:
            tin = getattr(usage, "prompt_tokens", None)
            tout = getattr(usage, "completion_tokens", None)
        return (
            int(tin) if tin is not None else None,
            int(tout) if tout is not None else None,
        )
    except Exception:
        return None, None

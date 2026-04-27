"""Aggregate metrics for one (task_model, source_model, prompt_idx) cell.

Inputs are the per-article rows produced by scripts/run_test_eval.py — one
dict per article — and the output is a single summary dict suitable for
serializing into outputs/test_eval/metrics/.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    """Macro-averaged F1 over the supplied labels.

    Stdlib-only so we don't pull sklearn into the hot loop. Equivalent to
    sklearn.metrics.f1_score(..., average="macro", zero_division=0).
    """
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)
        denom = (2 * tp + fp + fn)
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return sum(f1s) / len(f1s) if f1s else 0.0


def aggregate_cell_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary metrics over the per-article rows of one cell.

    Expected row keys: gold_score, pred_score, abs_error, signed_error,
    gepa_score, exact_match, direction_correct, gold_band, pred_band,
    parse_status, latency_ms, n_retries.
    """
    n = len(rows)
    parsed = [r for r in rows if r["parse_status"] == "ok"]
    n_parsed = len(parsed)

    if n == 0:
        return {
            "n_articles": 0,
            "n_parsed": 0,
            "parse_rate": 0.0,
        }

    gepa_scores = [r["gepa_score"] for r in rows]
    parse_rate = n_parsed / n

    # Metrics over parsed rows only — None preds skew anything that uses pred_score.
    abs_errors = [r["abs_error"] for r in parsed]
    signed_errors = [r["signed_error"] for r in parsed]
    exact = sum(1 for r in parsed if r["exact_match"]) / n if n else 0.0
    direction_acc = sum(1 for r in parsed if r["direction_correct"]) / n if n else 0.0

    bands = ["bearish", "neutral", "bullish"]
    direction_macro_f1 = _macro_f1(
        [r["gold_band"] for r in parsed],
        [r["pred_band"] for r in parsed],
        bands,
    ) if n_parsed else 0.0

    mae = sum(abs_errors) / n_parsed if n_parsed else None
    rmse = math.sqrt(sum(e * e for e in abs_errors) / n_parsed) if n_parsed else None
    mean_signed_error = sum(signed_errors) / n_parsed if n_parsed else None
    mean_gepa = sum(gepa_scores) / n

    latencies = [r.get("latency_ms") for r in rows if r.get("latency_ms") is not None]
    retries = [r.get("n_retries", 0) for r in rows]

    pred_dist = Counter(r["pred_score"] for r in parsed)
    error_dist = Counter(r["abs_error"] for r in parsed)
    band_confusion: dict[str, dict[str, int]] = {b: {bb: 0 for bb in bands} for b in bands}
    for r in parsed:
        band_confusion[r["gold_band"]][r["pred_band"]] += 1

    return {
        "n_articles": n,
        "n_parsed": n_parsed,
        "parse_rate": parse_rate,
        "mean_gepa_score": mean_gepa,
        "exact_match_accuracy": exact,
        "direction_accuracy": direction_acc,
        "direction_macro_f1": direction_macro_f1,
        "mean_absolute_error": mae,
        "root_mean_squared_error": rmse,
        "mean_signed_error": mean_signed_error,
        "score_distribution": dict(sorted(pred_dist.items())),
        "abs_error_distribution": dict(sorted(error_dist.items())),
        "band_confusion": band_confusion,
        "median_latency_ms": _median(latencies),
        "p95_latency_ms": _percentile(latencies, 95),
        "total_retries": sum(retries),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(s[lo])
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

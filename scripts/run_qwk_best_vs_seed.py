"""Compute Quadratic Weighted Kappa for best vs seed prompts.

For each (task_model_slug, source_model_slug) pair (4 x 4 = 16 pairs), compare
the QWK of the seed prompt and the GEPA-best prompt against gold labels.
QWK is computed on direction (-1/0/+1, 3-class ordinal) and confidence (1-5,
5-class ordinal). MAE and exact-match rate on the full 1-15 score are also
recorded for context.

Inputs:  data/qwk/<task_model_slug>/<task_model_slug>.csv
Outputs: outputs/qwk/best_vs_seed/{summary.json, summary.csv, README.md}

Usage:
    python scripts/run_qwk_best_vs_seed.py
    python scripts/run_qwk_best_vs_seed.py --input data/qwk \\
                                           --output-dir outputs/qwk/best_vs_seed
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "qwk"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "qwk" / "best_vs_seed"

CSV_COLUMNS = [
    "task_model_slug", "source_model_slug",
    "seed_prompt_idx", "best_prompt_idx",
    "n_articles_total", "n_seed_ok", "n_best_ok", "n_paired_ok",
    "seed_qwk_direction", "seed_qwk_confidence",
    "seed_mae", "seed_exact_match_rate",
    "best_qwk_direction", "best_qwk_confidence",
    "best_mae", "best_exact_match_rate",
    "delta_qwk_direction", "delta_qwk_confidence",
    "delta_mae", "delta_exact_match_rate",
    "seed_warning", "best_warning",
]


def _to_int(s: str) -> int | None:
    return int(s) if s != "" else None


def _to_bool(s: str) -> bool:
    return s.strip().lower() == "true"


def load_all_qwk_inputs(input_dir: Path) -> list[dict[str, Any]]:
    """Read every <slug>/<slug>.csv into a flat list of dicts with typed fields."""
    rows: list[dict[str, Any]] = []
    for task_dir in sorted(input_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        csv_path = task_dir / f"{task_dir.name}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    "article_id":        r["article_id"],
                    "task_model_slug":   r["task_model_slug"],
                    "source_model_slug": r["source_model_slug"],
                    "prompt_idx":        int(r["prompt_idx"]),
                    "is_seed":           _to_bool(r["is_seed"]),
                    "is_best":           _to_bool(r["is_best"]),
                    "parse_status":      r["parse_status"],
                    "gold_score":        _to_int(r["gold_score"]),
                    "pred_score":        _to_int(r["pred_score"]),
                    "gold_direction":    _to_int(r["gold_direction"]),
                    "gold_confidence":   _to_int(r["gold_confidence"]),
                    "pred_direction":    _to_int(r["pred_direction"]),
                    "pred_confidence":   _to_int(r["pred_confidence"]),
                })
    if not rows:
        sys.exit(f"[ERROR] no per-task CSVs found in {input_dir}")
    return rows


def safe_qwk(gold: list[int], pred: list[int]) -> tuple[float | None, str | None]:
    """Compute quadratic weighted kappa, returning (value, warning)."""
    if len(gold) == 0:
        return None, "empty sample"
    g = np.asarray(gold)
    p = np.asarray(pred)
    if len(np.unique(g)) < 2 and len(np.unique(p)) < 2:
        return None, "single-class gold and pred (kappa undefined)"
    try:
        k = cohen_kappa_score(g, p, weights="quadratic")
    except Exception as e:
        return None, f"sklearn error: {e}"
    if k is None or (isinstance(k, float) and np.isnan(k)):
        return None, "kappa is NaN"
    return float(k), None


def subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """QWK on direction + confidence, plus MAE and exact-match on 1-15 score."""
    if not rows:
        return {
            "qwk_direction": None, "qwk_confidence": None,
            "mae": None, "exact_match_rate": None,
            "warning": "no parsed rows",
        }
    gold_dir = [r["gold_direction"] for r in rows]
    pred_dir = [r["pred_direction"] for r in rows]
    gold_conf = [r["gold_confidence"] for r in rows]
    pred_conf = [r["pred_confidence"] for r in rows]
    gold_score = np.asarray([r["gold_score"] for r in rows])
    pred_score = np.asarray([r["pred_score"] for r in rows])

    qwk_dir, w_dir = safe_qwk(gold_dir, pred_dir)
    qwk_conf, w_conf = safe_qwk(gold_conf, pred_conf)
    warnings = [w for w in (w_dir, w_conf) if w]

    return {
        "qwk_direction": qwk_dir,
        "qwk_confidence": qwk_conf,
        "mae": float(np.abs(gold_score - pred_score).mean()),
        "exact_match_rate": float((gold_score == pred_score).mean()),
        "warning": "; ".join(warnings) if warnings else None,
    }


def delta(best: dict[str, Any], seed: dict[str, Any]) -> dict[str, Any]:
    def sub(a: Any, b: Any) -> float | None:
        if a is None or b is None:
            return None
        return float(a - b)
    return {
        "qwk_direction": sub(best["qwk_direction"], seed["qwk_direction"]),
        "qwk_confidence": sub(best["qwk_confidence"], seed["qwk_confidence"]),
        "mae": sub(best["mae"], seed["mae"]),
        "exact_match_rate": sub(best["exact_match_rate"], seed["exact_match_rate"]),
    }


def compute_pair(
    rows: list[dict[str, Any]],
    task: str,
    source: str,
) -> dict[str, Any]:
    pair = [r for r in rows if r["task_model_slug"] == task and r["source_model_slug"] == source]
    pair_seed = [r for r in pair if r["is_seed"]]
    pair_best = [r for r in pair if r["is_best"]]
    seed_ok = [r for r in pair_seed if r["parse_status"] == "ok"]
    best_ok = [r for r in pair_best if r["parse_status"] == "ok"]

    seed_articles = {r["article_id"] for r in seed_ok}
    best_articles = {r["article_id"] for r in best_ok}
    n_paired = len(seed_articles & best_articles)

    seed_idx = pair_seed[0]["prompt_idx"] if pair_seed else None
    best_idx = pair_best[0]["prompt_idx"] if pair_best else None

    seed_metrics = subset_metrics(seed_ok)
    best_metrics = subset_metrics(best_ok)

    return {
        "task_model_slug": task,
        "source_model_slug": source,
        "seed_prompt_idx": seed_idx,
        "best_prompt_idx": best_idx,
        "n_articles_total": len({r["article_id"] for r in pair}),
        "n_seed_ok": len(seed_ok),
        "n_best_ok": len(best_ok),
        "n_paired_ok": n_paired,
        "seed": seed_metrics,
        "best": best_metrics,
        "delta": delta(best_metrics, seed_metrics),
    }


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "task_model_slug": record["task_model_slug"],
        "source_model_slug": record["source_model_slug"],
        "seed_prompt_idx": record["seed_prompt_idx"],
        "best_prompt_idx": record["best_prompt_idx"],
        "n_articles_total": record["n_articles_total"],
        "n_seed_ok": record["n_seed_ok"],
        "n_best_ok": record["n_best_ok"],
        "n_paired_ok": record["n_paired_ok"],
    }
    for prefix, block in (("seed", record["seed"]), ("best", record["best"])):
        flat[f"{prefix}_qwk_direction"] = block["qwk_direction"]
        flat[f"{prefix}_qwk_confidence"] = block["qwk_confidence"]
        flat[f"{prefix}_mae"] = block["mae"]
        flat[f"{prefix}_exact_match_rate"] = block["exact_match_rate"]
        flat[f"{prefix}_warning"] = block["warning"]
    d = record["delta"]
    flat["delta_qwk_direction"] = d["qwk_direction"]
    flat["delta_qwk_confidence"] = d["qwk_confidence"]
    flat["delta_mae"] = d["mae"]
    flat["delta_exact_match_rate"] = d["exact_match_rate"]
    return flat


def write_readme(out_dir: Path, n_pairs: int) -> None:
    body = f"""# QWK best vs seed prompt comparison

Generated by `scripts/run_qwk_best_vs_seed.py`.

For each `(task_model_slug, source_model_slug)` pair, compare the QWK of the
seed prompt and the GEPA-best prompt against gold labels. Total: {n_pairs} pairs.

## Files

- `summary.json` — full structured results, one entry per pair, with nested
  `seed`, `best`, and `delta` blocks.
- `summary.csv` — flat one-row-per-pair table for plotting / analysis.

## Metrics

- `qwk_direction` — Cohen's kappa (quadratic weights) on the 3-class direction
  (`-1` bearish / `0` neutral / `+1` bullish).
- `qwk_confidence` — Cohen's kappa (quadratic weights) on the 5-class
  confidence (1-5 within the predicted band).
- `mae` — mean absolute error of the full 1-15 score (lower = better).
- `exact_match_rate` — fraction of articles where `pred_score == gold_score`.
- `delta_*` — `best - seed` for each metric. Positive = GEPA improved.

Rows with `parse_status == "failed"` are excluded. Sample sizes
(`n_seed_ok`, `n_best_ok`, `n_paired_ok`) are recorded so post-hoc filtering
or paired analysis is possible.

`warning` is non-null when kappa is undefined (single-class sample, etc).
"""
    (out_dir / "README.md").write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="root containing <slug>/<slug>.csv files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                        help="where to write summary.json + summary.csv")
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.output_dir)
    if not src.exists():
        sys.exit(f"[ERROR] input dir not found: {src}")
    out.mkdir(parents=True, exist_ok=True)

    rows = load_all_qwk_inputs(src)
    print(f"[INFO] loaded {len(rows)} rows from {src}")

    by_task: dict[str, set[str]] = defaultdict(set)
    by_source: set[str] = set()
    for r in rows:
        by_task[r["task_model_slug"]].add(r["source_model_slug"])
        by_source.add(r["source_model_slug"])
    task_models = sorted(by_task.keys())
    source_models = sorted(by_source)
    print(f"[INFO] {len(task_models)} task models x {len(source_models)} source models = "
          f"{len(task_models) * len(source_models)} pairs")

    records: list[dict[str, Any]] = []
    for task in task_models:
        for source in source_models:
            rec = compute_pair(rows, task, source)
            records.append(rec)
            d = rec["delta"]["qwk_direction"]
            d_str = f"{d:+.3f}" if d is not None else "  n/a"
            seed_dir = rec["seed"]["qwk_direction"]
            best_dir = rec["best"]["qwk_direction"]
            seed_str = f"{seed_dir:.3f}" if seed_dir is not None else " n/a "
            best_str = f"{best_dir:.3f}" if best_dir is not None else " n/a "
            print(f"[INFO] {task:<16} <- {source:<16}  "
                  f"seed_dir={seed_str}  best_dir={best_str}  delta={d_str}  "
                  f"(n_seed={rec['n_seed_ok']:>2}, n_best={rec['n_best_ok']:>2})")

    json_path = out / "summary.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {json_path}")

    csv_path = out / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in records:
            w.writerow(flatten(r))
    print(f"[INFO] wrote {csv_path}")

    write_readme(out, len(records))
    print(f"[INFO] wrote {out / 'README.md'}")


if __name__ == "__main__":
    main()

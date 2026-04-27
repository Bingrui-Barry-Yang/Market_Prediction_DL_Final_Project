"""Stage 2: Test-set evaluation across the full (task_model x prompt) cross-product.

Runs every prompt candidate from every GEPA source-model on every task model
against the held-out test set. Single-shot, temperature=0.

For 4 task models * (11+7+11+9)=38 prompts * 35 articles = 5,320 LLM calls,
this script writes per-cell predictions JSONL incrementally so it is fully
resumable. After all cells finish, it produces:
  - long-format per_article_scores.jsonl
  - wide score / abs_error / parse matrices (CSV)
  - per-cell metrics JSON files
  - summary.json

The script does NOT compute paired-t-tests; the matrices are shaped so that
`scipy.stats.ttest_rel(matrix[col_a], matrix[col_b])` works downstream.

Usage (test machine):

    export ANTHROPIC_API_KEY=...
    # Ollama default base http://localhost:11434
    python scripts/run_test_eval.py

    # Filtered runs:
    python scripts/run_test_eval.py --task-model gemma4e2b --limit 3
    python scripts/run_test_eval.py --resume
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.jsonl import read_jsonl, write_jsonl  # noqa: E402
from src.data.schemas import ArticleRecord  # noqa: E402
from src.evaluation.metrics import aggregate_cell_metrics  # noqa: E402
from src.evaluation.registry import (  # noqa: E402
    SOURCE_MODELS,
    TASK_MODELS,
    PromptCandidate,
    SourceModel,
    TaskModel,
    get_source_model,
    get_task_model,
    load_candidates,
)
from src.evaluation.scoring import (  # noqa: E402
    build_article_input,
    direction_band,
    extract_token_counts,
    gepa_partial_credit,
    litellm_completion_with_retries,
    parse_score_response,
)

DEFAULT_DATA = REPO_ROOT / "data" / "test" / "articles.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "test_eval"
DEFAULT_EXPECTED_N = 35


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_articles(path: Path, expected_n: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if expected_n is not None and len(rows) != expected_n:
        raise SystemExit(
            f"[ERROR] expected {expected_n} test articles in {path}, "
            f"found {len(rows)}. Pass --expected-n N to override."
        )
    seen: set[str] = set()
    for i, row in enumerate(rows, start=1):
        try:
            ArticleRecord.model_validate(row)
        except Exception as e:
            raise SystemExit(f"[ERROR] {path}:{i} fails ArticleRecord: {e}") from e
        aid = row["article_id"]
        if aid in seen:
            raise SystemExit(f"[ERROR] duplicate article_id '{aid}' at {path}:{i}")
        seen.add(aid)
    return rows


def _cell_predictions_path(out: Path, task_slug: str, source_slug: str, idx: int) -> Path:
    return out / "predictions" / task_slug / f"{source_slug}__prompt_{idx:02d}.jsonl"


def _cell_metrics_path(out: Path, task_slug: str, source_slug: str, idx: int) -> Path:
    return out / "metrics" / task_slug / f"{source_slug}__prompt_{idx:02d}.json"


def _column_id(task_slug: str, source_slug: str, idx: int) -> str:
    return f"{task_slug}__{source_slug}__p{idx:02d}"


def _is_cell_complete(path: Path, expected_n: int) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("rb") as f:
            return sum(1 for line in f if line.strip()) >= expected_n
    except OSError:
        return False


def _build_row(
    *,
    article: dict[str, Any],
    task: TaskModel,
    source: SourceModel,
    cand: PromptCandidate,
    raw: str,
    parse_result,
    latency_ms: float,
    n_retries: int,
    tokens_in: int | None,
    tokens_out: int | None,
) -> dict[str, Any]:
    gold = int(article["gold_score"])
    pred = parse_result.pred_score
    gepa_score = gepa_partial_credit(pred, gold)
    abs_error = abs(pred - gold) if pred is not None else None
    signed_error = (pred - gold) if pred is not None else None
    gold_b = direction_band(gold)
    pred_b = direction_band(pred) if pred is not None else None
    return {
        "article_id": article["article_id"],
        "task_model": task.litellm_id,
        "task_model_slug": task.slug,
        "source_model_slug": source.slug,
        "prompt_idx": cand.idx,
        "is_seed": cand.is_seed,
        "is_best": cand.is_best,
        "prompt_sha256": cand.sha256,
        "gold_score": gold,
        "pred_score": pred,
        "abs_error": abs_error,
        "signed_error": signed_error,
        "gepa_score": gepa_score,
        "exact_match": pred is not None and pred == gold,
        "direction_correct": pred_b is not None and pred_b == gold_b,
        "gold_band": gold_b,
        "pred_band": pred_b,
        "parse_status": parse_result.parse_status,
        "parse_error": parse_result.parse_error,
        "raw_response": raw,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "n_retries": n_retries,
        "timestamp": _now_iso(),
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_prompts_index(out: Path, prompts_by_source: dict[str, list[PromptCandidate]]) -> None:
    payload = []
    for src in SOURCE_MODELS:
        for cand in prompts_by_source[src.slug]:
            payload.append(
                {
                    "source_model_slug": src.slug,
                    "source_model_litellm_id": src.litellm_id,
                    "prompt_idx": cand.idx,
                    "is_seed": cand.is_seed,
                    "is_best": cand.is_best,
                    "sha256": cand.sha256,
                    "system_prompt": cand.text,
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "prompts_index.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _build_matrices(
    out: Path,
    article_ids: list[str],
    columns: list[tuple[str, str, int]],
    rows_by_cell: dict[tuple[str, str, int], list[dict[str, Any]]],
) -> None:
    """Pivot per-article rows into 3 wide CSVs (rows=articles, cols=cells)."""
    score_path = out / "score_matrix.csv"
    error_path = out / "error_matrix.csv"
    parse_path = out / "parse_matrix.csv"

    column_ids = [_column_id(t, s, i) for (t, s, i) in columns]
    by_cell_index: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {
        cell: {r["article_id"]: r for r in rows} for cell, rows in rows_by_cell.items()
    }

    def _write(path: Path, value_key: str, missing: Any = "") -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["article_id", *column_ids])
            for aid in article_ids:
                row_out = [aid]
                for cell in columns:
                    rec = by_cell_index.get(cell, {}).get(aid)
                    if rec is None:
                        row_out.append(missing)
                    else:
                        v = rec.get(value_key)
                        row_out.append("" if v is None else v)
                w.writerow(row_out)

    _write(score_path, "gepa_score")
    _write(error_path, "abs_error")
    _write(parse_path, "parse_status_int")  # populated below

    # parse_matrix needs an int conversion; rebuild it cleanly.
    with parse_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["article_id", *column_ids])
        for aid in article_ids:
            row_out = [aid]
            for cell in columns:
                rec = by_cell_index.get(cell, {}).get(aid)
                if rec is None:
                    row_out.append("")
                else:
                    row_out.append(1 if rec.get("parse_status") == "ok" else 0)
            w.writerow(row_out)


def _write_readme(out: Path, expected_n: int) -> None:
    text = f"""# outputs/test_eval — Stage 2 test-set evaluation

Generated by `scripts/run_test_eval.py`. Each cell is one
(task_model x source_model x prompt_idx) combination evaluated single-shot
on every article in `data/test/articles.jsonl` (n={expected_n}).

Layout:

- `predictions/{{task_slug}}/{{source_slug}}__prompt_{{idx:02d}}.jsonl` —
  one JSON record per article, written incrementally during the run.
- `metrics/{{task_slug}}/{{source_slug}}__prompt_{{idx:02d}}.json` — aggregated
  metrics for the cell (mean GEPA score, accuracy, MAE, RMSE, direction
  accuracy, macro F1, parse rate, latency p50/p95).
- `prompts_index.json` — snapshot of every prompt evaluated (text + sha256
  + is_seed / is_best flags).
- `per_article_scores.jsonl` — long-format concatenation of all per-cell
  predictions, suitable for loading into pandas.
- `score_matrix.csv` — rows=article_id, cols=`{{task}}__{{source}}__p{{idx:02d}}`,
  cells=`gepa_score` (0.0/0.25/0.5/0.75/1.0).
- `error_matrix.csv` — same shape, cells=`abs_error` (None encoded as empty).
- `parse_matrix.csv` — same shape, cells=`1` if parse ok else `0`.
- `summary.json` — one entry per cell with the headline aggregates.
- `run_log.jsonl` — one line per LLM call (timing, retries, parse status).

Slug map:

| task_slug / source_slug | LiteLLM model id |
|---|---|
| claude_sonnet46 | anthropic/claude-sonnet-4-6 |
| gptoss120b | ollama_chat/gpt-oss:120b |
| qwen36 | ollama_chat/qwen3.6:latest |
| gemma4e2b | ollama_chat/gemma4:e2b |

Regenerate with:

    python scripts/run_test_eval.py --resume
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def _aggregate_all(out: Path) -> None:
    """Read all per-cell predictions and produce summary + matrices + concat."""
    pred_root = out / "predictions"
    if not pred_root.exists():
        print("[INFO] no predictions to aggregate")
        return

    rows_by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for task in TASK_MODELS:
        task_dir = pred_root / task.slug
        if not task_dir.exists():
            continue
        for src in SOURCE_MODELS:
            for cand in load_candidates(src):
                p = _cell_predictions_path(out, task.slug, src.slug, cand.idx)
                if not p.exists():
                    continue
                rows = read_jsonl(p)
                rows_by_cell[(task.slug, src.slug, cand.idx)] = rows
                all_rows.extend(rows)

    write_jsonl(out / "per_article_scores.jsonl", all_rows)

    # Per-cell metrics
    summary: list[dict[str, Any]] = []
    for (task_slug, src_slug, idx), rows in rows_by_cell.items():
        m = aggregate_cell_metrics(rows)
        m_full = {
            "task_model_slug": task_slug,
            "source_model_slug": src_slug,
            "prompt_idx": idx,
            **m,
        }
        mp = _cell_metrics_path(out, task_slug, src_slug, idx)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("w", encoding="utf-8") as f:
            json.dump(m_full, f, indent=2, ensure_ascii=False)
        summary.append(m_full)

    summary.sort(key=lambda s: (s["task_model_slug"], s["source_model_slug"], s["prompt_idx"]))
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Wide matrices — sort articles by id for stable column-pairing
    article_ids: list[str] = sorted({r["article_id"] for r in all_rows})
    columns: list[tuple[str, str, int]] = sorted(rows_by_cell.keys())
    _build_matrices(out, article_ids, columns, rows_by_cell)


def _evaluate_cell(
    *,
    task: TaskModel,
    source: SourceModel,
    cand: PromptCandidate,
    articles: list[dict[str, Any]],
    out: Path,
    args: argparse.Namespace,
    run_log_path: Path,
) -> int:
    """Run one cell to completion. Returns number of new calls made."""
    pred_path = _cell_predictions_path(out, task.slug, source.slug, cand.idx)
    expected_n = len(articles)

    if args.resume and _is_cell_complete(pred_path, expected_n):
        print(
            f"[SKIP] {task.slug} <- {source.slug} p{cand.idx:02d} "
            f"already has >= {expected_n} rows"
        )
        return 0

    # If the file exists but is incomplete, resume by skipping article_ids already done.
    done_ids: set[str] = set()
    if pred_path.exists() and args.resume:
        for r in read_jsonl(pred_path):
            done_ids.add(r["article_id"])

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    n_made = 0

    for article in articles:
        if article["article_id"] in done_ids:
            continue
        messages = [
            {"role": "system", "content": cand.text},
            {"role": "user", "content": build_article_input(article)},
        ]

        t0 = time.perf_counter()
        n_retries = 0
        raw = ""
        tokens_in = tokens_out = None
        try:
            response, n_retries = litellm_completion_with_retries(
                model_name=task.litellm_id,
                messages=messages,
                num_retries=args.max_retries,
                base_backoff=args.base_backoff,
                max_backoff=args.max_backoff,
                timeout=args.timeout,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            tokens_in, tokens_out = extract_token_counts(response)
            err_kind = None
            err_detail = None
        except Exception as e:
            err_kind = type(e).__name__
            err_detail = str(e)[:300]
            raw = ""

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if raw:
            parsed = parse_score_response(raw)
        else:
            parsed = type(
                "P",
                (),
                {
                    "pred_score": None,
                    "parse_status": "failed",
                    "parse_error": err_detail or "no response",
                },
            )()

        row = _build_row(
            article=article,
            task=task,
            source=source,
            cand=cand,
            raw=raw,
            parse_result=parsed,
            latency_ms=latency_ms,
            n_retries=n_retries,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        if err_kind:
            row["call_error_kind"] = err_kind
            row["call_error_detail"] = err_detail

        _append_jsonl(pred_path, row)
        _append_jsonl(
            run_log_path,
            {
                "timestamp": _now_iso(),
                "task_model_slug": task.slug,
                "source_model_slug": source.slug,
                "prompt_idx": cand.idx,
                "article_id": article["article_id"],
                "latency_ms": latency_ms,
                "n_retries": n_retries,
                "parse_status": row["parse_status"],
                "call_error_kind": row.get("call_error_kind"),
            },
        )
        n_made += 1
        print(
            f"[CALL] {task.slug:<16} <- {source.slug:<16} p{cand.idx:02d} "
            f"art={article['article_id']:<24} pred={row['pred_score']} gold={row['gold_score']} "
            f"gepa={row['gepa_score']:.2f} {row['parse_status']} {latency_ms:.0f}ms",
            flush=True,
        )
    return n_made


def _planned_cells(
    args: argparse.Namespace,
) -> list[tuple[TaskModel, SourceModel, PromptCandidate]]:
    task_models = [get_task_model(args.task_model)] if args.task_model else TASK_MODELS
    source_models = [get_source_model(args.source_model)] if args.source_model else SOURCE_MODELS

    cells: list[tuple[TaskModel, SourceModel, PromptCandidate]] = []
    for src in source_models:
        for cand in load_candidates(src):
            if args.prompt_idx is not None and cand.idx != args.prompt_idx:
                continue
            for task in task_models:
                cells.append((task, src, cand))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 test-set evaluation runner.",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--task-model", default=None, choices=[t.slug for t in TASK_MODELS],
    )
    parser.add_argument(
        "--source-model", default=None, choices=[s.slug for s in SOURCE_MODELS],
    )
    parser.add_argument("--prompt-idx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--expected-n",
        type=int,
        default=DEFAULT_EXPECTED_N,
        help="hard-error if test JSONL row count != N (0 to disable)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--base-backoff", type=float, default=2.0)
    parser.add_argument("--max-backoff", type=float, default=30.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="skip the post-run aggregation phase (matrices/summary)",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    run_log_path = out / "run_log.jsonl"

    expected_n = None if args.expected_n == 0 else args.expected_n
    articles = _validate_articles(Path(args.data), expected_n)
    if args.limit is not None:
        articles = articles[: args.limit]

    cells = _planned_cells(args)
    n_calls = len(cells) * len(articles)
    print(
        f"[PLAN] task_models={len(set(c[0].slug for c in cells))} "
        f"prompts={len(set((c[1].slug, c[2].idx) for c in cells))} "
        f"articles={len(articles)} cells={len(cells)} total_calls={n_calls}",
        flush=True,
    )

    # Snapshot prompts_index before any LLM call.
    prompts_by_source = {src.slug: load_candidates(src) for src in SOURCE_MODELS}
    _write_prompts_index(out, prompts_by_source)
    _write_readme(out, expected_n if expected_n is not None else len(articles))

    if args.dry_run:
        for task, src, cand in cells:
            print(f"[PLANNED] {task.slug:<16} <- {src.slug:<16} p{cand.idx:02d} "
                  f"(seed={cand.is_seed} best={cand.is_best})")
        print("[DRY-RUN] no LLM calls made.")
        return

    total_new = 0
    for task, src, cand in cells:
        total_new += _evaluate_cell(
            task=task, source=src, cand=cand, articles=articles,
            out=out, args=args, run_log_path=run_log_path,
        )
    print(f"[DONE] new calls: {total_new}", flush=True)

    if not args.no_aggregate:
        _aggregate_all(out)
        print(f"[AGGREGATED] outputs at {out}", flush=True)


if __name__ == "__main__":
    main()

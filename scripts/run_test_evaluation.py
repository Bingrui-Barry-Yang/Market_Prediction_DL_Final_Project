"""
Evaluate all 4 GEPA-optimized prompts against Claude Sonnet 4.6 on the test dataset.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    cd Market_Prediction_DL_Final_Project
    python scripts/run_test_evaluation.py
"""

import json
import math
import re
import time
from pathlib import Path

import litellm

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_JSONL = Path("data/test/testing_dataset.jsonl")
OUTPUT_DIR = Path("outputs/test_evaluation")

CANDIDATES = {
    "claude_sonnet46": Path("outputs/gepa_runs/bitcoin_sentiment/claude_sonnet46/candidates.json"),
    "gptoss120b":      Path("outputs/gepa_runs/bitcoin_sentiment/run_gptoss120b_b150/candidates.json"),
    "qwen36":          Path("outputs/gepa_runs/bitcoin_sentiment/run_qwen36_b150/candidates.json"),
    "gemma4e2b":       Path("outputs/gepa_runs/bitcoin_sentiment/run_gemma4e2b_b150/candidates.json"),
}

EVAL_MODEL = "anthropic/claude-sonnet-4-6"
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_articles(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_best_prompt(candidates_path: Path) -> str:
    with candidates_path.open(encoding="utf-8") as f:
        candidates = json.load(f)
    return candidates[-1]["system_prompt"]


def build_user_message(article: dict) -> str:
    title = article.get("title", "").strip()
    text  = article.get("text", "").strip()
    if title and text:
        return f"Title: {title}\n\nArticle text:\n{text}"
    return title or text


def parse_score(response_text: str) -> int | None:
    text = response_text.strip()
    # Try {"score": N}
    m = re.search(r'"score"\s*:\s*(\d+)', text)
    if m:
        return int(m.group(1))
    # Fallback: bare integer
    m = re.search(r'\b(\d+)\b', text)
    if m:
        return int(m.group(1))
    return None


def call_with_retries(system: str, user: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=EVAL_MODEL,
                max_tokens=64,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return response.choices[0].message.content
        except (litellm.RateLimitError, litellm.APIConnectionError,
                litellm.InternalServerError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"    Retry {attempt+1}/{MAX_RETRIES} after {wait:.0f}s ({e})")
            time.sleep(wait)
    raise RuntimeError("Unreachable")


# ── Metrics ───────────────────────────────────────────────────────────────────

def mae(preds: list[float], golds: list[int]) -> float:
    return sum(abs(p - g) for p, g in zip(preds, golds)) / len(preds)


def rmse(preds: list[float], golds: list[int]) -> float:
    return math.sqrt(sum((p - g) ** 2 for p, g in zip(preds, golds)) / len(preds))


def pearson(preds: list[float], golds: list[int]) -> float:
    n = len(preds)
    mp = sum(preds) / n
    mg = sum(golds) / n
    num = sum((p - mp) * (g - mg) for p, g in zip(preds, golds))
    dp  = math.sqrt(sum((p - mp) ** 2 for p in preds))
    dg  = math.sqrt(sum((g - mg) ** 2 for g in golds))
    if dp == 0 or dg == 0:
        return 0.0
    return num / (dp * dg)


def accuracy_within(preds: list[float], golds: list[int], tol: int) -> float:
    return sum(1 for p, g in zip(preds, golds) if abs(p - g) <= tol) / len(preds)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    articles = load_articles(TEST_JSONL)
    golds    = [a["gold_score"] for a in articles]

    print(f"Test articles : {len(articles)}")
    print(f"Eval model    : {EVAL_MODEL}")
    print(f"GEPA prompts  : {list(CANDIDATES.keys())}")
    print()

    all_results = {}

    for model_name, candidates_path in CANDIDATES.items():
        print(f"── {model_name} ──")
        system_prompt = load_best_prompt(candidates_path)
        preds = []
        parse_failures = 0

        for i, article in enumerate(articles):
            user_msg = build_user_message(article)
            raw = call_with_retries(system_prompt, user_msg)
            score = parse_score(raw)
            if score is None or not (1 <= score <= 15):
                print(f"  [WARN] article {i}: parse failed — raw={raw!r}")
                score = 8  # neutral fallback
                parse_failures += 1
            preds.append(score)
            print(f"  [{i+1:2d}/{len(articles)}] gold={article['gold_score']:2d}  pred={score:2d}  "
                  f"id={article['article_id']}")
            time.sleep(0.1)

        metrics = {
            "mae":        round(mae(preds, golds), 3),
            "rmse":       round(rmse(preds, golds), 3),
            "pearson_r":  round(pearson(preds, golds), 3),
            "acc_within_1": round(accuracy_within(preds, golds, 1), 3),
            "acc_within_2": round(accuracy_within(preds, golds, 2), 3),
            "parse_failures": parse_failures,
        }
        all_results[model_name] = {"preds": preds, "metrics": metrics}

        print(f"  MAE={metrics['mae']}  RMSE={metrics['rmse']}  "
              f"r={metrics['pearson_r']}  "
              f"±1={metrics['acc_within_1']:.1%}  ±2={metrics['acc_within_2']:.1%}")
        print()

    # ── Save detailed results ────────────────────────────────────────────────
    detailed = []
    for article, gold in zip(articles, golds):
        row = {"article_id": article["article_id"], "gold_score": gold}
        for model_name, result in all_results.items():
            idx = articles.index(article)
            row[f"pred_{model_name}"] = result["preds"][idx]
        detailed.append(row)

    out_path = OUTPUT_DIR / "results.json"
    with out_path.open("w") as f:
        json.dump({"model": EVAL_MODEL, "n": len(articles),
                   "metrics": {m: r["metrics"] for m, r in all_results.items()},
                   "per_article": detailed}, f, indent=2)
    print(f"Results saved to {out_path}")

    # ── Print summary report ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"GEPA PROMPT TEST EVALUATION — {EVAL_MODEL} — n={len(articles)}")
    print("=" * 70)
    header = f"{'Prompt source':<20} {'MAE':>6} {'RMSE':>6} {'r':>6} {'±1 acc':>8} {'±2 acc':>8}"
    print(header)
    print("-" * 70)
    for model_name, result in all_results.items():
        m = result["metrics"]
        print(f"{model_name:<20} {m['mae']:>6.3f} {m['rmse']:>6.3f} "
              f"{m['pearson_r']:>6.3f} {m['acc_within_1']:>7.1%} {m['acc_within_2']:>7.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()

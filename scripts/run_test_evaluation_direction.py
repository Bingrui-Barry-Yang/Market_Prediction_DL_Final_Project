"""
Evaluate all 4 GEPA-optimized prompts against Claude Sonnet 4.6 using the
direction override (bear=1, neutral=2, bull=3) instead of the 1-15 scale.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    cd Market_Prediction_DL_Final_Project
    python scripts/run_test_evaluation_direction.py
"""

import json
import re
import time
from pathlib import Path

import litellm

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_JSONL   = Path("data/test/testing_dataset.jsonl")
OVERRIDE_RTF = Path("data/override.rtf")
OUTPUT_DIR   = Path("outputs/test_evaluation")

CANDIDATES = {
    "claude_sonnet46": Path("outputs/gepa_runs/bitcoin_sentiment/claude_sonnet46/candidates.json"),
    "gptoss120b":      Path("outputs/gepa_runs/bitcoin_sentiment/run_gptoss120b_b150/candidates.json"),
    "qwen36":          Path("outputs/gepa_runs/bitcoin_sentiment/run_qwen36_b150/candidates.json"),
    "gemma4e2b":       Path("outputs/gepa_runs/bitcoin_sentiment/run_gemma4e2b_b150/candidates.json"),
}

EVAL_MODEL    = "anthropic/claude-sonnet-4-6"
MAX_RETRIES   = 8
RETRY_BACKOFF = 3.0
CALL_SLEEP    = 7.0   # seconds between calls to stay under 30k tokens/min limit


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_rtf_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Unescape literal braces BEFORE stripping RTF groups
    raw = raw.replace("\\{", "LBRACE").replace("\\}", "RBRACE")
    # RTF special chars
    raw = raw.replace("\\'a0", " ").replace("\\'97", "—")
    # Strip RTF control words and groups
    raw = re.sub(r'\{[^{}]*\}', '', raw)          # remove {...} groups
    raw = re.sub(r'\\[a-zA-Z]+\d*[ ]?', '', raw)  # remove control words
    raw = re.sub(r'\\[\*~]', '', raw)              # remove special controls
    raw = re.sub(r'[{}\\]', '', raw)               # remove stray braces/backslashes
    # Restore literal braces
    raw = raw.replace("LBRACE", "{").replace("RBRACE", "}")
    # Collapse whitespace while preserving newlines
    lines = [l.strip() for l in raw.splitlines()]
    text = '\n'.join(l for l in lines if l)
    return text.strip()


def load_articles(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_best_prompt(candidates_path: Path, override: str) -> str:
    with candidates_path.open(encoding="utf-8") as f:
        candidates = json.load(f)
    return candidates[-1]["system_prompt"] + "\n\n" + override


def build_user_message(article: dict) -> str:
    title = article.get("title", "").strip()
    text  = article.get("text", "").strip()
    if title and text:
        return f"Title: {title}\n\nArticle text:\n{text}"
    return title or text


def gold_to_direction(score: int) -> int:
    if score <= 5:   return 1  # bear
    if score <= 10:  return 2  # neutral
    return 3                   # bull


def parse_direction(response_text: str) -> int | None:
    text = response_text.strip()
    # {"direction": "3"} or {"direction": 3}
    m = re.search(r'"direction"\s*:\s*["\']?(\d)["\']?', text)
    if m:
        val = int(m.group(1))
        if val in (1, 2, 3):
            return val
    # Bare digit fallback
    m = re.search(r'\b([123])\b', text)
    if m:
        return int(m.group(1))
    return None


def call_with_retries(system: str, user: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=EVAL_MODEL,
                max_tokens=32,
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


def direction_label(d: int) -> str:
    return {1: "bear", 2: "neutral", 3: "bull"}.get(d, "?")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    override = parse_rtf_text(OVERRIDE_RTF)
    print("Override instruction:")
    print(override)
    print()

    articles   = load_articles(TEST_JSONL)
    gold_dirs  = [gold_to_direction(a["gold_score"]) for a in articles]

    print(f"Test articles : {len(articles)}")
    print(f"Eval model    : {EVAL_MODEL}")
    print(f"Gold dist     : bear={gold_dirs.count(1)}  neutral={gold_dirs.count(2)}  bull={gold_dirs.count(3)}")
    print()

    all_results = {}

    for model_name, candidates_path in CANDIDATES.items():
        print(f"── {model_name} ──")
        system_prompt = load_best_prompt(candidates_path, override)
        preds          = []
        parse_failures = 0

        for i, (article, gold_dir) in enumerate(zip(articles, gold_dirs)):
            user_msg = build_user_message(article)
            raw      = call_with_retries(system_prompt, user_msg)
            pred     = parse_direction(raw)
            if pred is None:
                print(f"  [WARN] article {i+1}: parse failed — raw={raw!r}")
                pred = 2  # neutral fallback
                parse_failures += 1
            preds.append(pred)
            match = "✓" if pred == gold_dir else "✗"
            print(f"  [{i+1:2d}/{len(articles)}] gold={direction_label(gold_dir):<7} "
                  f"pred={direction_label(pred):<7} {match}  id={article['article_id']}")
            time.sleep(CALL_SLEEP)

        n       = len(preds)
        correct = sum(p == g for p, g in zip(preds, gold_dirs))
        by_dir  = {}
        for label in (1, 2, 3):
            true_idx = [i for i, g in enumerate(gold_dirs) if g == label]
            hits     = sum(1 for i in true_idx if preds[i] == label)
            by_dir[direction_label(label)] = f"{hits}/{len(true_idx)}"

        metrics = {
            "accuracy":       round(correct / n, 3),
            "bear_acc":       by_dir["bear"],
            "neutral_acc":    by_dir["neutral"],
            "bull_acc":       by_dir["bull"],
            "parse_failures": parse_failures,
        }
        all_results[model_name] = {"preds": preds, "metrics": metrics}

        print(f"  Accuracy={metrics['accuracy']:.1%}  "
              f"bear={metrics['bear_acc']}  neutral={metrics['neutral_acc']}  bull={metrics['bull_acc']}")
        print()

    # ── Save results ─────────────────────────────────────────────────────────
    detailed = []
    for i, article in enumerate(articles):
        row = {
            "article_id":    article["article_id"],
            "gold_score":    article["gold_score"],
            "gold_direction": direction_label(gold_dirs[i]),
        }
        for model_name, result in all_results.items():
            row[f"pred_{model_name}"] = direction_label(result["preds"][i])
        detailed.append(row)

    out_path = OUTPUT_DIR / "results_direction.json"
    with out_path.open("w") as f:
        json.dump({"model": EVAL_MODEL, "n": len(articles),
                   "metrics": {m: r["metrics"] for m, r in all_results.items()},
                   "per_article": detailed}, f, indent=2)
    print(f"Results saved to {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"DIRECTION EVAL — {EVAL_MODEL} — n={len(articles)}")
    print(f"Gold dist: bear={gold_dirs.count(1)}  neutral={gold_dirs.count(2)}  bull={gold_dirs.count(3)}")
    print("=" * 65)
    print(f"{'Prompt source':<20} {'Accuracy':>9} {'Bear':>8} {'Neutral':>8} {'Bull':>8}")
    print("-" * 65)
    for model_name, result in all_results.items():
        m = result["metrics"]
        print(f"{model_name:<20} {m['accuracy']:>8.1%} {m['bear_acc']:>8} "
              f"{m['neutral_acc']:>8} {m['bull_acc']:>8}")
    print("=" * 65)


if __name__ == "__main__":
    main()

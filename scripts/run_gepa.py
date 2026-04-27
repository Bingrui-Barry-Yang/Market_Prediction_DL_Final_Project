"""
Bitcoin Sentiment - GEPA Optimization
Reads the human gold standard JSONL file and runs GEPA to evolve the system prompt.
Saves the best prompt to gepa_result.json.

Input JSONL format (one JSON object per line):
    {
        "article_id":     "article-001",
        "text":           "",
        "title":          "Analysts expect Bitcoin to rise...",
        "url":            "https://...",
        "source":         "Example News",
        "date":           "2024-03",
        "gold_score":     15,
        "gold_reasoning": "The article links demand growth to a higher Bitcoin price."
    }

Score scale (1-15):
    Bearish range  (1-5):   1 = very weakly bearish  ... 5 = strongly bearish
    Neutral range  (6-10):  6 = very weakly neutral   ... 10 = strongly neutral
    Bullish range (11-15): 11 = very weakly bullish   ... 15 = strongly bullish

Usage:
    export GEMINI_API_KEY=...
    python scripts/run_gepa.py data/train/articles.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from threading import Lock

import gepa
from gepa.adapters.default_adapter.default_adapter import (
    DefaultDataInst,
    EvaluationResult,
)

# --- Configuration ---
GEPA_RESULT_PATH = "outputs/gepa_runs/gepa_result.json"
GEPA_RUN_DIR = "outputs/gepa_runs/bitcoin_sentiment"
DATA_PATH = "data/train/articles.jsonl"
TASK_LM = "gemini/gemini-2.5-flash-lite"
REFLECTION_LM = "gemini/gemini-2.5-flash-lite"
MAX_METRIC_CALLS = 150

SEED_PROMPT = """You are a Bitcoin sentiment analyst. Score the forward-looking \
sentiment of a Bitcoin news article on a scale of 1 to 15.

Bearish range  (1-5):   1 = very weakly bearish  ... 5 = strongly bearish
Neutral range  (6-10):  6 = very weakly neutral   ... 10 = strongly neutral
Bullish range (11-15): 11 = very weakly bullish   ... 15 = strongly bullish

Only consider forward-looking content: predictions, forecasts, price targets, \
analyst outlooks, and future expectations.
Ignore all retrospective content: past price performance, historical data, \
and anything describing what already happened.

Respond with only: {"score": <integer 1-15>}"""

SCORE_LABELS = {
    **{s: f"Bearish  (confidence {s})" for s in range(1, 6)},
    **{s: f"Neutral  (confidence {s - 5})" for s in range(6, 11)},
    **{s: f"Bullish  (confidence {s - 10})" for s in range(11, 16)},
}


# --- Custom Evaluator ---

class SentimentScoreEvaluator:
    """
    Evaluates the LLM's 1-15 score against the human gold standard.
    Exact match = 1.0, off by 1 = 0.75, off by 2 = 0.5, off by 3 = 0.25, larger = 0.0.
    """

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        expected = int(data["answer"])

        try:
            raw = response.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            predicted = int(json.loads(raw)["score"])
            if predicted not in range(1, 16):
                raise ValueError(f"score {predicted} out of range 1-15")
        except Exception as e:
            return EvaluationResult(
                score=0.0,
                feedback=(
                    f"Failed to parse response. Error: {e}. "
                    f"Got: '{response[:200]}'. "
                    f"Expected JSON: {{\"score\": {expected}}} "
                    f"({SCORE_LABELS[expected]})."
                ),
            )

        if predicted == expected:
            return EvaluationResult(
                score=1.0,
                feedback=(
                    f"Correct. Score {predicted} ({SCORE_LABELS[predicted]}) "
                    "matches gold standard."
                ),
            )

        diff = abs(predicted - expected)
        partial = {1: 0.75, 2: 0.5, 3: 0.25}.get(diff, 0.0)

        direction = "too bullish" if predicted > expected else "too bearish"
        feedback = (
            f"Incorrect. Predicted {predicted} ({SCORE_LABELS[predicted]}), "
            f"expected {expected} ({SCORE_LABELS[expected]}). "
            f"Off by {diff} — {direction}. "
            f"Human reasoning: {data['additional_context'].get('gold_reasoning', '')}. "
            f"Source: {data['additional_context'].get('source', '')} "
            f"({data['additional_context'].get('date', '')})."
        )

        return EvaluationResult(score=partial, feedback=feedback)


def build_logging_reflection_lm(model_name: str, run_dir: str):
    import litellm

    transcript_path = Path(run_dir) / "reflection_transcripts.jsonl"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    counter = {"n": 0}

    def _call(prompt):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        response = litellm.completion(model=model_name, messages=messages)
        content = response.choices[0].message.content

        with lock:
            counter["n"] += 1
            record = {
                "call_index": counter["n"],
                "timestamp": time.time(),
                "model": model_name,
                "messages": messages,
                "response": content,
            }
            with transcript_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return content

    return _call


def build_article_input(row: dict[str, object]) -> str:
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    if title and text:
        return f"Title: {title}\n\nArticle text:\n{text}"
    if title:
        return f"Title: {title}"
    return text


# --- Data Loading ---

def load_jsonl(data_path: str) -> tuple[list[DefaultDataInst], list[DefaultDataInst]]:
    """
    Load the human gold standard JSONL file.
    Every 5th row (0-indexed) goes to val, the rest to train (~80/20).
    """
    trainset: list[DefaultDataInst] = []
    valset: list[DefaultDataInst] = []

    with open(data_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rows_with_text = sum(1 for row in rows if str(row.get("text", "")).strip())

    for i, row in enumerate(rows):
        gold_score = row.get("gold_score")
        gold_reasoning = row.get("gold_reasoning", "").strip()
        article_id = row.get("article_id", "").strip()

        if gold_score is None or not gold_reasoning:
            row_label = article_id or f"row {i + 1}"
            print(f"[WARN] Skipping {row_label}: missing gold_score or gold_reasoning.")
            continue

        instance: DefaultDataInst = {
            "input": build_article_input(row),
            "answer": str(int(gold_score)),
            "additional_context": {
                "gold_reasoning": gold_reasoning,
            },
        }

        if i % 5 == 0:
            valset.append(instance)
        else:
            trainset.append(instance)

    print(f"[INFO] Loaded {len(trainset)} train, {len(valset)} val examples.")
    print(f"[INFO] Records with article text: {rows_with_text}/{len(rows)}")
    print("[INFO] Score distribution:")
    all_scores = [int(inst["answer"]) for inst in trainset + valset]
    for s in sorted(set(all_scores)):
        print(f"         {s:>2} ({SCORE_LABELS[s]}): {all_scores.count(s)}")
    print()

    return trainset, valset


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Run GEPA to optimize the Bitcoin sentiment system prompt."
    )
    parser.add_argument(
        "data",
        nargs="?",
        default=DATA_PATH,
        help=f"Input JSONL path (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        default=GEPA_RESULT_PATH,
        help=f"Output path for result JSON (default: {GEPA_RESULT_PATH})",
    )
    parser.add_argument(
        "--run-dir",
        default=GEPA_RUN_DIR,
        help=f"GEPA run directory (default: {GEPA_RUN_DIR})",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=MAX_METRIC_CALLS,
        help=f"Max rollout budget (default: {MAX_METRIC_CALLS})",
    )
    parser.add_argument(
        "--task-lm",
        default=TASK_LM,
        help=f"Task model passed to GEPA/LiteLLM (default: {TASK_LM})",
    )
    parser.add_argument(
        "--reflection-lm",
        default=REFLECTION_LM,
        help=f"Reflection model passed to GEPA/LiteLLM (default: {REFLECTION_LM})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate data without calling GEPA or any model API.",
    )
    args = parser.parse_args()

    trainset, valset = load_jsonl(args.data)

    if len(trainset) < 5:
        raise ValueError(f"Need at least 5 training examples, got {len(trainset)}.")
    if len(valset) < 2:
        raise ValueError(f"Need at least 2 validation examples, got {len(valset)}.")
    if args.dry_run:
        print("[INFO] Dry run complete. GEPA optimization was not started.")
        return

    print(f"[INFO] Starting GEPA optimization (budget={args.budget})...")
    print(f"[INFO] Task LM:       {args.task_lm}")
    print(f"[INFO] Reflection LM: {args.reflection_lm}\n")

    reflection_lm_callable = build_logging_reflection_lm(
        args.reflection_lm, args.run_dir
    )

    result = gepa.optimize(
        seed_candidate={"system_prompt": SEED_PROMPT},
        trainset=trainset,
        valset=valset,
        task_lm=args.task_lm,
        evaluator=SentimentScoreEvaluator(),
        reflection_lm=reflection_lm_callable,
        max_metric_calls=args.budget,
        run_dir=args.run_dir,
        track_best_outputs=True,
    )

    best_prompt = result.best_candidate["system_prompt"]

    output = {
        "best_candidate": {"system_prompt": best_prompt},
        "task_lm":        args.task_lm,
        "reflection_lm":  args.reflection_lm,
        "budget":         args.budget,
        "train_size":     len(trainset),
        "val_size":       len(valset),
        "run_dir":        args.run_dir,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Done. Best prompt saved to '{args.output}'.")
    print("\n=== Best GEPA Prompt ===")
    print(best_prompt)


if __name__ == "__main__":
    main()

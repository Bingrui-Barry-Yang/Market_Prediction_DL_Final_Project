"""
Bitcoin Sentiment - GEPA Optimization (rate-limit resilient variant)

Same as scripts/run_gepa.py but tolerates provider rate limits (429s).

Approach (Option A — serialize + handcrafted retries):
  - Task LM is wrapped as a *callable* passed to gepa's DefaultAdapter. When
    the adapter's model is a callable, each request is invoked sequentially
    (`[self.model(messages) for messages in batch]`), so we never use
    litellm.batch_completion — which was the source of the original crash
    (batch_completion returns RateLimitError objects in-place, and the
    adapter then dereferences `.choices[0]` on them).
  - Both the task-LM and reflection-LM paths call a shared helper,
    `_completion_with_retries`, which retries litellm.completion on
    RateLimitError / Timeout / APIConnectionError / InternalServerError
    with exponential backoff + jitter, capped at --max-backoff seconds
    (default 60s — long enough to outlast Gemini's 60s rate window).
  - Errors that are NOT in the retryable set raise immediately.

Use this variant when running against providers with tight RPM / TPM
limits (e.g. Gemini 2.5 Flash / Flash-Lite free tier).

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
    python scripts/run_gepa_rate_limited.py data/train/articles.jsonl
"""

import argparse
import json
import random
import time
from pathlib import Path
from threading import Lock

from gepa.adapters.default_adapter.default_adapter import (
    DefaultAdapter,
    DefaultDataInst,
    EvaluationResult,
)

import gepa

# --- Configuration ---
GEPA_RESULT_PATH = "outputs/gepa_runs/gepa_result.json"
GEPA_RUN_DIR = "outputs/gepa_runs/bitcoin_sentiment"
DATA_PATH = "data/train/articles.jsonl"
TASK_LM = "gemini/gemini-2.5-flash-lite"
REFLECTION_LM = "gemini/gemini-2.5-flash-lite"
MAX_METRIC_CALLS = 150

# Rate-limit / retry defaults (tuned for Gemini free-tier style limits:
# 10 RPM on gemini-2.5-flash, 15 RPM on flash-lite → 60s rate window).
LITELLM_NUM_RETRIES = 8
LITELLM_TIMEOUT_SECONDS = 60
RETRY_BASE_BACKOFF_SECONDS = 5.0
RETRY_MAX_BACKOFF_SECONDS = 60.0

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


# --- Retry helper shared by task-LM and reflection-LM paths ---

def _completion_with_retries(
    model_name: str,
    messages,
    num_retries: int,
    base_backoff: float,
    max_backoff: float,
    timeout: int,
):
    """
    Call litellm.completion with hand-rolled retries on transient errors.
    Exponential backoff with jitter, capped at max_backoff, then raises.
    """
    import litellm

    retryable = (
        litellm.RateLimitError,
        litellm.Timeout,
        litellm.APIConnectionError,
        litellm.InternalServerError,
        litellm.ServiceUnavailableError,
    )

    last_err: BaseException | None = None
    for attempt in range(num_retries + 1):
        try:
            return litellm.completion(
                model=model_name,
                messages=list(messages),
                timeout=timeout,
            )
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


def build_task_lm(
    model_name: str,
    num_retries: int,
    base_backoff: float,
    max_backoff: float,
    timeout: int,
):
    """
    Return a ChatCompletionCallable for gepa's DefaultAdapter.

    When DefaultAdapter.model is a callable, it is invoked sequentially per
    request in the batch — so we avoid litellm.batch_completion entirely.
    """

    def _call(messages) -> str:
        response = _completion_with_retries(
            model_name=model_name,
            messages=messages,
            num_retries=num_retries,
            base_backoff=base_backoff,
            max_backoff=max_backoff,
            timeout=timeout,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    return _call


def build_logging_reflection_lm(
    model_name: str,
    run_dir: str,
    num_retries: int,
    base_backoff: float,
    max_backoff: float,
    timeout: int,
):
    transcript_path = Path(run_dir) / "reflection_transcripts.jsonl"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    counter = {"n": 0}

    def _call(prompt):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        response = _completion_with_retries(
            model_name=model_name,
            messages=messages,
            num_retries=num_retries,
            base_backoff=base_backoff,
            max_backoff=max_backoff,
            timeout=timeout,
        )
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
        description=(
            "Run GEPA to optimize the Bitcoin sentiment system prompt, "
            "with handcrafted retry/backoff on provider rate limits."
        )
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
        help=f"Task model passed to litellm (default: {TASK_LM})",
    )
    parser.add_argument(
        "--reflection-lm",
        default=REFLECTION_LM,
        help=f"Reflection model passed to litellm (default: {REFLECTION_LM})",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=LITELLM_NUM_RETRIES,
        help=(
            "Retries on retryable errors (RateLimit / Timeout / "
            f"APIConnection / InternalServer) — default: {LITELLM_NUM_RETRIES}."
        ),
    )
    parser.add_argument(
        "--base-backoff",
        type=float,
        default=RETRY_BASE_BACKOFF_SECONDS,
        help=(
            "Exponential backoff base in seconds — sleep grows as "
            f"base * 2**attempt (default: {RETRY_BASE_BACKOFF_SECONDS})."
        ),
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=RETRY_MAX_BACKOFF_SECONDS,
        help=(
            "Cap on any single backoff sleep in seconds "
            f"(default: {RETRY_MAX_BACKOFF_SECONDS} — outlasts a 60s rate window)."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=LITELLM_TIMEOUT_SECONDS,
        help=(
            "Per-request litellm timeout in seconds "
            f"(default: {LITELLM_TIMEOUT_SECONDS})."
        ),
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
    print(f"[INFO] Task LM:        {args.task_lm}")
    print(f"[INFO] Reflection LM:  {args.reflection_lm}")
    print(
        f"[INFO] Retry cfg:      num_retries={args.num_retries}, "
        f"base_backoff={args.base_backoff}s, max_backoff={args.max_backoff}s, "
        f"timeout={args.request_timeout}s"
    )
    print("[INFO] Execution:      sequential (callable task-LM, no batch_completion)\n")

    reflection_lm_callable = build_logging_reflection_lm(
        model_name=args.reflection_lm,
        run_dir=args.run_dir,
        num_retries=args.num_retries,
        base_backoff=args.base_backoff,
        max_backoff=args.max_backoff,
        timeout=args.request_timeout,
    )

    task_lm_callable = build_task_lm(
        model_name=args.task_lm,
        num_retries=args.num_retries,
        base_backoff=args.base_backoff,
        max_backoff=args.max_backoff,
        timeout=args.request_timeout,
    )

    adapter = DefaultAdapter(
        model=task_lm_callable,
        evaluator=SentimentScoreEvaluator(),
    )

    result = gepa.optimize(
        seed_candidate={"system_prompt": SEED_PROMPT},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
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
        "rate_limit_config": {
            "num_retries":     args.num_retries,
            "base_backoff":    args.base_backoff,
            "max_backoff":     args.max_backoff,
            "request_timeout": args.request_timeout,
            "execution":       "sequential",
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Done. Best prompt saved to '{args.output}'.")
    print("\n=== Best GEPA Prompt ===")
    print(best_prompt)


if __name__ == "__main__":
    main()

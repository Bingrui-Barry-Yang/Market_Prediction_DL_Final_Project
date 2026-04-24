"""
Token usage and cost smoketest for running GEPA + test evaluation on the Claude API.

Uses:
  - client.messages.count_tokens() for all input token measurements (no model charges)
  - 5 real API calls (3 task LM + 2 reflection LM) on claude-haiku-4-5 (cheapest) to
    measure actual output tokens
  - Extrapolates to full GEPA budget (102 task calls, 19 reflection calls) and
    test evaluation (~40 articles)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    cd Market_Prediction_DL_Final_Project
    python scripts/smoketest_token_cost.py
"""

import json
import statistics
import time
from pathlib import Path

import anthropic

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_JSONL = "data/train/articles.jsonl"
TEST_JSONL  = "data/test/articles.jsonl"
CANDIDATES_JSON = "outputs/gepa_runs/bitcoin_sentiment/run_gptoss120b_b150/candidates.json"
REFLECTION_JSONL = "outputs/gepa_runs/bitcoin_sentiment/run_gptoss120b_b150/reflection_transcripts.jsonl"

# ── GEPA run constants (from the completed gptoss120b run) ───────────────────
TOTAL_TASK_LM_CALLS   = 102   # total task-model calls in a full 150-budget run
TOTAL_REFLECTION_CALLS = 19   # one reflection per GEPA iteration that ran
TEST_ARTICLES          = 40   # AGENTS.md: 35-40 unseen test articles

# Model to use for actual output measurement (cheapest Claude model)
PROBE_MODEL = "claude-haiku-4-5"

# ── Pricing ($ per 1 M tokens) ───────────────────────────────────────────────
PRICING = {
    "claude-opus-4-7":   {"in": 5.00,  "out": 25.00},
    "claude-sonnet-4-6": {"in": 3.00,  "out": 15.00},
    "claude-haiku-4-5":  {"in": 1.00,  "out":  5.00},
}

SEED_PROMPT = """You are a Bitcoin sentiment analyst. Score the forward-looking \
sentiment of a Bitcoin news article on a scale of 1 to 15.

Bearish range  (1-5):   1 = very weakly bearish  ... 5 = strongly bearish
Neutral range  (6-10):  6 = very weakly neutral   ... 10 = strongly neutral
Bullish range (11-15): 11 = very weakly bullish   ... 15 = strongly bullish

Only consider forward-looking content: predictions, forecasts, price targets, \
analyst outlooks, and future expectations.
Ignore all retrospective content: past price performance, historical data, \
and anything describing what already happened.

Respond with only: {\"score\": <integer 1-15>}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_article_input(row: dict) -> str:
    title = str(row.get("title", "")).strip()
    text  = str(row.get("text",  "")).strip()
    if title and text:
        return f"Title: {title}\n\nArticle text:\n{text}"
    return title or text


def load_articles(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile_indices(lst: list, pcts: list[float]) -> list[int]:
    """Return indices corresponding to the given percentiles of lst."""
    sorted_i = sorted(range(len(lst)), key=lambda i: lst[i])
    return [sorted_i[int(p * (len(lst) - 1))] for p in pcts]


def count_input_tokens(client, model: str, system: str, user: str) -> int:
    r = client.messages.count_tokens(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return r.input_tokens


def count_reflection_input_tokens(client, model: str, prompt_text: str) -> int:
    r = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return r.input_tokens


def call_task_lm(client, model: str, system: str, user: str) -> tuple[int, int]:
    """Returns (input_tokens, output_tokens)."""
    r = client.messages.create(
        model=model,
        max_tokens=64,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return r.usage.input_tokens, r.usage.output_tokens


def call_reflection_lm(client, model: str, prompt_text: str) -> tuple[int, int]:
    """Returns (input_tokens, output_tokens)."""
    r = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return r.usage.input_tokens, r.usage.output_tokens


def cost_usd(tokens: int, price_per_million: float) -> float:
    return tokens / 1_000_000 * price_per_million


def fmt(n: float) -> str:
    return f"${n:.4f}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    client = anthropic.Anthropic()

    # ── Load data ────────────────────────────────────────────────────────────
    train_rows = load_articles(TRAIN_JSONL)
    test_rows  = load_articles(TEST_JSONL)

    train_inputs = [build_article_input(r) for r in train_rows]
    test_inputs  = [build_article_input(r) for r in (test_rows or train_rows)]

    with open(CANDIDATES_JSON, encoding="utf-8") as f:
        candidates = json.load(f)

    prompts = [c["system_prompt"] for c in candidates]
    evolved_prompts = prompts[1:]  # exclude seed

    reflection_records = []
    with open(REFLECTION_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                reflection_records.append(json.loads(line))
    reflection_prompts = [r["messages"][0]["content"] for r in reflection_records]

    print("=" * 62)
    print("GEPA Token & Cost Smoketest")
    print("=" * 62)
    print(f"Train articles : {len(train_rows)}")
    print(f"Test articles  : {len(test_rows) if test_rows else '(using train as proxy)'}")
    print(f"Prompt candidates (seed+evolved): {len(prompts)}")
    print(f"Reflection transcripts: {len(reflection_records)}")
    print(f"Probe model for output measurement: {PROBE_MODEL}")
    print()

    # ── 1. Task LM — input tokens ─────────────────────────────────────────
    print("─" * 62)
    print("1. Counting input tokens for Task LM calls …")

    # Sample 5 articles spanning the size distribution
    article_chars = [len(a) for a in train_inputs]
    sample_pcts   = [0.0, 0.25, 0.5, 0.75, 1.0]
    sample_idx    = percentile_indices(article_chars, sample_pcts)

    # Measure with seed prompt and with the median evolved prompt
    median_evolved = evolved_prompts[len(evolved_prompts) // 2]

    seed_in_tokens  = []
    evol_in_tokens  = []

    for idx in sample_idx:
        article = train_inputs[idx]
        s = count_input_tokens(client, PROBE_MODEL, SEED_PROMPT,    article)
        e = count_input_tokens(client, PROBE_MODEL, median_evolved, article)
        seed_in_tokens.append(s)
        evol_in_tokens.append(e)
        print(f"   article[{idx:2d}] chars={article_chars[idx]:5d}  "
              f"seed_in={s:5d} tok  evol_in={e:5d} tok")

    avg_seed_in = statistics.mean(seed_in_tokens)
    avg_evol_in = statistics.mean(evol_in_tokens)
    print(f"\n   Avg task LM input — seed prompt : {avg_seed_in:.0f} tok")
    print(f"   Avg task LM input — evolved prompt: {avg_evol_in:.0f} tok")

    # Approximate mix: first ~6 calls use seed, rest use evolved prompts
    seed_calls  = 6
    evol_calls  = TOTAL_TASK_LM_CALLS - seed_calls
    avg_task_in = (seed_calls * avg_seed_in + evol_calls * avg_evol_in) / TOTAL_TASK_LM_CALLS
    print(f"   Blended avg task LM input ({seed_calls} seed + {evol_calls} evolved): "
          f"{avg_task_in:.0f} tok")

    # ── 2. Task LM — output tokens (real calls) ──────────────────────────
    print()
    print("─" * 62)
    print("2. Measuring Task LM output tokens (3 real API calls) …")

    task_out_samples = []
    for idx in sample_idx[:3]:
        article = train_inputs[idx]
        in_tok, out_tok = call_task_lm(client, PROBE_MODEL, SEED_PROMPT, article)
        task_out_samples.append(out_tok)
        print(f"   article[{idx:2d}]: in={in_tok} tok  out={out_tok} tok")
        time.sleep(0.5)

    avg_task_out = statistics.mean(task_out_samples)
    print(f"\n   Avg task LM output: {avg_task_out:.0f} tok")

    # ── 3. Reflection LM — input tokens ──────────────────────────────────
    print()
    print("─" * 62)
    print("3. Counting input tokens for Reflection LM calls …")

    refl_chars = [len(p) for p in reflection_prompts]
    refl_idx   = percentile_indices(refl_chars, [0.0, 0.25, 0.5, 0.75, 1.0])

    refl_in_tokens = []
    for idx in refl_idx:
        prompt = reflection_prompts[idx]
        tok = count_reflection_input_tokens(client, PROBE_MODEL, prompt)
        refl_in_tokens.append(tok)
        print(f"   reflection[{idx:2d}] chars={refl_chars[idx]:6d}  in={tok:6d} tok")

    avg_refl_in = statistics.mean(refl_in_tokens)
    print(f"\n   Avg reflection LM input: {avg_refl_in:.0f} tok")

    # ── 4. Reflection LM — output tokens (2 real calls) ──────────────────
    print()
    print("─" * 62)
    print("4. Measuring Reflection LM output tokens (2 real API calls) …")

    # Use small and median prompts to keep cost down
    refl_out_samples = []
    for idx in refl_idx[:2]:
        prompt = reflection_prompts[idx]
        in_tok, out_tok = call_reflection_lm(client, PROBE_MODEL, prompt)
        refl_out_samples.append(out_tok)
        print(f"   reflection[{idx:2d}]: in={in_tok} tok  out={out_tok} tok")
        time.sleep(0.5)

    avg_refl_out = statistics.mean(refl_out_samples)
    print(f"\n   Avg reflection LM output: {avg_refl_out:.0f} tok")

    # ── 5. Test evaluation — input tokens ────────────────────────────────
    print()
    print("─" * 62)
    print("5. Counting input tokens for Test Evaluation …")

    best_evolved = evolved_prompts[-1]  # last/best evolved prompt
    test_sample_idx = percentile_indices(
        [len(t) for t in test_inputs], [0.0, 0.5, 1.0]
    )
    test_in_tokens = []
    for idx in test_sample_idx:
        article = test_inputs[idx]
        tok = count_input_tokens(client, PROBE_MODEL, best_evolved, article)
        test_in_tokens.append(tok)
        print(f"   test_article[{idx:2d}] chars={len(article):6d}  in={tok:5d} tok")

    avg_test_in = statistics.mean(test_in_tokens)
    print(f"\n   Avg test eval input: {avg_test_in:.0f} tok")

    # ── 6. Total token projections ─────────────────────────────────────────
    print()
    print("=" * 62)
    print("PROJECTED TOKEN USAGE — Full GEPA + Test Run")
    print("=" * 62)

    total_task_in  = int(avg_task_in  * TOTAL_TASK_LM_CALLS)
    total_task_out = int(avg_task_out * TOTAL_TASK_LM_CALLS)
    total_refl_in  = int(avg_refl_in  * TOTAL_REFLECTION_CALLS)
    total_refl_out = int(avg_refl_out * TOTAL_REFLECTION_CALLS)
    total_test_in  = int(avg_test_in  * TEST_ARTICLES)
    total_test_out = int(avg_task_out * TEST_ARTICLES)  # same short JSON response

    grand_in  = total_task_in + total_refl_in + total_test_in
    grand_out = total_task_out + total_refl_out + total_test_out

    print(f"\n{'Component':<35} {'Input':>10} {'Output':>10}  (tokens)")
    print("-" * 60)
    print(f"{'GEPA Task LM':<35} {total_task_in:>10,} {total_task_out:>10,}")
    print(f"{'GEPA Reflection LM':<35} {total_refl_in:>10,} {total_refl_out:>10,}")
    print(f"{'Test Evaluation':<35} {total_test_in:>10,} {total_test_out:>10,}")
    print("-" * 60)
    print(f"{'TOTAL':<35} {grand_in:>10,} {grand_out:>10,}")

    # ── 7. Cost by model ───────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("ESTIMATED COST BY MODEL")
    print("=" * 62)
    print(f"\n{'Model':<22} {'Input cost':>12} {'Output cost':>12} {'Total':>12}")
    print("-" * 60)

    for model, p in PRICING.items():
        in_cost  = cost_usd(grand_in,  p["in"])
        out_cost = cost_usd(grand_out, p["out"])
        total    = in_cost + out_cost
        print(f"{model:<22} {fmt(in_cost):>12} {fmt(out_cost):>12} {fmt(total):>12}")

    # ── 8. Per-phase breakdown for chosen model ───────────────────────────
    print()
    print("─" * 62)
    print("Per-phase breakdown (for each model)")
    print("─" * 62)

    phases = [
        ("GEPA Task LM",       total_task_in,  total_task_out),
        ("GEPA Reflection LM", total_refl_in,  total_refl_out),
        ("Test Evaluation",    total_test_in,  total_test_out),
    ]
    header = f"{'Phase':<22}"
    for model in PRICING:
        header += f" {model:>14}"
    print(header)
    print("-" * (22 + 15 * len(PRICING)))

    for label, in_tok, out_tok in phases:
        row = f"{label:<22}"
        for p in PRICING.values():
            c = cost_usd(in_tok, p["in"]) + cost_usd(out_tok, p["out"])
            row += f" {fmt(c):>14}"
        print(row)

    row = f"{'TOTAL':<22}"
    for p in PRICING.values():
        c = cost_usd(grand_in, p["in"]) + cost_usd(grand_out, p["out"])
        row += f" {fmt(c):>14}"
    print("-" * (22 + 15 * len(PRICING)))
    print(row)

    # ── 9. Key averages summary ────────────────────────────────────────────
    print()
    print("─" * 62)
    print("Measured averages (input from count_tokens; output from real calls)")
    print("─" * 62)
    print(f"  Task LM  avg input  (seed prompt)   : {avg_seed_in:.0f} tok")
    print(f"  Task LM  avg input  (evolved prompt) : {avg_evol_in:.0f} tok")
    print(f"  Task LM  avg output                  : {avg_task_out:.0f} tok")
    print(f"  Refl LM  avg input                   : {avg_refl_in:.0f} tok")
    print(f"  Refl LM  avg output                  : {avg_refl_out:.0f} tok")
    print(f"  Test eval avg input (best prompt)    : {avg_test_in:.0f} tok")
    print()
    print("Assumptions:")
    print(f"  • {TOTAL_TASK_LM_CALLS} task LM calls ({seed_calls} with seed, "
          f"{evol_calls} with evolved prompts)")
    print(f"  • {TOTAL_REFLECTION_CALLS} reflection LM calls")
    print(f"  • {TEST_ARTICLES} test articles")
    print()


if __name__ == "__main__":
    main()

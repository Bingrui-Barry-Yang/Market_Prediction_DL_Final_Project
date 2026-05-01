"""
Author Trustworthiness Evaluator
Evaluates how trustworthy a Bitcoin news author is by comparing their
GEPA sentiment predictions against actual Bitcoin price movement.

For each article:
- Fetches hourly Bitcoin price for 30 days post-publication via Coinbase Exchange
- Calculates green area (correct) and red area (incorrect) using trapezoidal rule
- Computes per-article ratio = (green - red) / (green + red)

Author scores:
- trust_simple   = average ratio across all articles
- trust_weighted = confidence-weighted average ratio

Score interpretation:
- Above +0.5  -> trustworthy
- -0.5 to 0.5 -> unreliable / noisy
- Below -0.5  -> consistently wrong (contrarian signal)

Usage:
    python evaluate_author.py --articles articles.jsonl --output results.csv

If an article row has no "gold_score", a 1-15 score is generated on the fly by
calling the chosen GEPA-optimised system prompt via litellm. Requires
ANTHROPIC_API_KEY (or the relevant key for --score-model) in env.

Price source: Coinbase Exchange public endpoint (no API key).
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

# --- Configuration ---
COINBASE_BASE = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
EVAL_DAYS     = 30
EVAL_HOURS    = EVAL_DAYS * 24   # 720 hours
NEUTRAL_BAND  = 0.05             # +-5%
API_DELAY     = 0.3              # seconds between Coinbase calls

# LLM scoring (used only when an article lacks "gold_score")
DEFAULT_CANDIDATES = {
    "claude_sonnet46": "outputs/gepa_runs/bitcoin_sentiment/claude_sonnet46/candidates.json",
    "qwen36":          "outputs/gepa_runs/bitcoin_sentiment/run_qwen36_b150/candidates.json",
    "gemma4e2b":       "outputs/gepa_runs/bitcoin_sentiment/run_gemma4e2b_b150/candidates.json",
    "gptoss120b":      "outputs/gepa_runs/bitcoin_sentiment/run_gptoss120b_b150/candidates.json",
}
DEFAULT_LLM_MODEL = "anthropic/claude-sonnet-4-6"
LLM_MAX_RETRIES   = 5
LLM_RETRY_BACKOFF = 2.0

# Score ranges
BEARISH_RANGE = range(1, 6)
NEUTRAL_RANGE = range(6, 11)
BULLISH_RANGE = range(11, 16)

# Offsets for confidence extraction
OFFSET = {"bearish": 0, "neutral": 5, "bullish": 10}


# --- Score helpers ---

def get_direction(score: int) -> str:
    if score in BEARISH_RANGE:
        return "bearish"
    elif score in NEUTRAL_RANGE:
        return "neutral"
    else:
        return "bullish"


def get_confidence(score: int) -> int:
    """Extract confidence 1-5 from 1-15 score."""
    return score - OFFSET[get_direction(score)]


# --- Price fetching ---

def fetch_hourly_prices(date_str: str) -> list:
    """
    Fetch hourly Bitcoin closing prices for 30 days starting from date_str.
    Uses Coinbase Exchange BTC-USD 1h candles -- no API key required.

    Args:
        date_str: Publication date "YYYY-MM-DD" or "YYYY-MM"

    Returns:
        List of (unix_timestamp_ms, close_price) tuples, sorted ascending.
    """
    if len(date_str) == 7:
        date_str = date_str + "-01"
    pub_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = pub_date + timedelta(hours=EVAL_HOURS)

    # Coinbase candle endpoint caps at 300 candles per request.
    # Use 200-hour windows to stay safely under the limit.
    chunk = timedelta(hours=200)
    all_candles: dict[int, float] = {}
    cur = pub_date

    while cur < end_dt:
        chunk_end = min(cur + chunk, end_dt)
        response = httpx.get(
            COINBASE_BASE,
            params={
                "granularity": 3600,
                "start":       cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end":         chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            timeout=30,
        )
        response.raise_for_status()
        # Coinbase returns [time_seconds, low, high, open, close, volume]
        # in DESCENDING time order.
        for candle in response.json():
            ts_ms = int(candle[0]) * 1000
            close_price = float(candle[4])
            all_candles[ts_ms] = close_price

        cur = chunk_end
        time.sleep(API_DELAY)

    return sorted(all_candles.items())


# --- Area calculation ---

def trapezoidal_segment(p0: float, p1: float, ref: float) -> float:
    """Area of one hourly segment relative to a reference line."""
    return 0.5 * (abs(p0 - ref) + abs(p1 - ref)) * 1  # x1 hour timestep


def calculate_areas(prices: list, baseline: float, direction: str) -> tuple:
    """
    Calculate green (correct) and red (incorrect) areas.

    Bullish:  green if price above baseline, red if below
    Bearish:  green if price below baseline, red if above
    Neutral:  green if price inside +-5% band, red if outside
              area measured to/from nearest band boundary

    Returns:
        (green_area, red_area)
    """
    green = 0.0
    red   = 0.0
    p_up  = baseline * (1 + NEUTRAL_BAND)
    p_lo  = baseline * (1 - NEUTRAL_BAND)

    for i in range(len(prices) - 1):
        p0  = prices[i]
        p1  = prices[i + 1]
        avg = (p0 + p1) / 2

        if direction == "bullish":
            area = trapezoidal_segment(p0, p1, baseline)
            if avg > baseline:
                green += area
            elif avg < baseline:
                red += area

        elif direction == "bearish":
            area = trapezoidal_segment(p0, p1, baseline)
            if avg < baseline:
                green += area
            elif avg > baseline:
                red += area

        elif direction == "neutral":
            if p_lo <= avg <= p_up:
                ref   = p_up if avg >= baseline else p_lo
                area  = trapezoidal_segment(p0, p1, ref)
                green += area
            else:
                ref  = p_up if avg > p_up else p_lo
                area = trapezoidal_segment(p0, p1, ref)
                red  += area

    return green, red


def calculate_ratio(green: float, red: float) -> float:
    """
    ratio = (green - red) / (green + red)
    Returns 0.0 if price never moved to avoid division by zero.
    """
    total = green + red
    if total == 0:
        return 0.0
    return (green - red) / total


# --- Author scoring ---

def trust_simple(ratios: list) -> float:
    """Simple average ratio across all articles."""
    return sum(ratios) / len(ratios)


def trust_weighted(ratios: list, confidences: list) -> float:
    """
    Confidence-weighted average ratio.
    trust_weighted = sum(ratio_i * c_i) / sum(c_i)
    """
    return sum(r * c for r, c in zip(ratios, confidences)) / sum(confidences)


def interpret(score: float) -> str:
    if score > 0.5:
        return "Trustworthy"
    elif score < -0.5:
        return "Consistently wrong (contrarian signal)"
    else:
        return "Unreliable / noisy"


# --- LLM scoring (only used when "gold_score" absent) ---

def load_system_prompt(candidates_path: Path) -> str:
    with candidates_path.open(encoding="utf-8") as f:
        candidates = json.load(f)
    return candidates[-1]["system_prompt"]


def build_scoring_user_message(article: dict) -> str:
    title = article.get("title", "").strip()
    text  = article.get("text", "").strip()
    if title and text:
        return f"Title: {title}\n\nArticle text:\n{text}"
    return title or text


def parse_score(response_text: str) -> int | None:
    """Pull a 1-15 integer out of a model response.

    Tries (in order): JSON-style ``"score": N``, the markdown phrase
    ``score is N`` / ``final score: N``, then the *last* bare 1-15 integer in
    the text -- so reasoning passages mentioning prices or scale ranges (e.g.
    ``$66,000`` or ``range 1-15``) don't poison the result.
    """
    text = response_text.strip()
    if not text:
        return None
    m = re.search(r'"score"\s*:\s*(\d+)', text)
    if m and 1 <= int(m.group(1)) <= 15:
        return int(m.group(1))
    m = re.search(r'(?:final\s+score|score\s+is)\s*[:=]?\s*(\d+)', text, re.IGNORECASE)
    if m and 1 <= int(m.group(1)) <= 15:
        return int(m.group(1))
    candidates = [int(x) for x in re.findall(r'(?<!\d)(\d{1,2})(?!\d)', text)]
    in_range = [c for c in candidates if 1 <= c <= 15]
    if in_range:
        return in_range[-1]
    return None


def score_article_via_llm(article: dict, system_prompt: str, model: str) -> int:
    """Call litellm with the GEPA-optimised system prompt; return 1-15 score.

    Reasoning models (qwen3.6, gpt-oss, etc.) put their chain-of-thought in a
    separate `reasoning_content` field and may emit the final ``{"score": N}``
    only after a long preamble.  We pass a generous token budget and search
    both fields for an integer.
    """
    import litellm  # imported lazily so the script still runs without it
    user_msg = build_scoring_user_message(article)
    last_err = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = litellm.completion(
                model=model,
                max_tokens=8192,  # reasoning models can need thousands of tokens
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                timeout=600,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning_content", None) or ""
            # Prefer the formal answer; fall back to scanning reasoning text.
            for blob in (content, reasoning):
                score = parse_score(blob)
                if score is not None and 1 <= score <= 15:
                    return score
            print(
                f"    [WARN] could not parse score "
                f"(finish={choice.finish_reason}, content_len={len(content)}, "
                f"reasoning_len={len(reasoning)}) -- using 8"
            )
            return 8
        except Exception as e:  # noqa: BLE001 -- litellm wraps many error types
            last_err = e
            if attempt == LLM_MAX_RETRIES - 1:
                break
            wait = LLM_RETRY_BACKOFF * (2 ** attempt)
            print(f"    Retry {attempt+1}/{LLM_MAX_RETRIES} after {wait:.0f}s ({e})")
            time.sleep(wait)
    raise RuntimeError(f"LLM scoring failed after {LLM_MAX_RETRIES} retries: {last_err}")


# --- Main ---

def evaluate_article(article: dict, system_prompt: str | None, llm_model: str) -> dict:
    """Evaluate a single article and return its result."""
    article_id = article.get("article_id", "unknown")
    if "gold_score" in article and article["gold_score"] is not None:
        score        = int(article["gold_score"])
        score_source = "gold"
    else:
        if system_prompt is None:
            raise RuntimeError(
                f"Article {article_id} has no 'gold_score' and no scoring prompt was loaded."
            )
        score        = score_article_via_llm(article, system_prompt, llm_model)
        score_source = "llm"
    date_str   = article["date"]
    direction  = get_direction(score)
    confidence = get_confidence(score)

    print(f"  [{article_id}] score={score} ({direction}, c={confidence}) date={date_str}")

    price_data = fetch_hourly_prices(date_str)
    if len(price_data) < 2:
        print(f"  [WARN] Not enough price data for {article_id}, skipping.")
        return None

    baseline = price_data[0][1]
    prices   = [p for _, p in price_data]

    green, red = calculate_areas(prices, baseline, direction)
    ratio      = calculate_ratio(green, red)

    print(f"         baseline=${baseline:,.0f}  green={green:,.0f}  red={red:,.0f}  ratio={ratio:.3f}")

    return {
        "article_id":   article_id,
        "score":        score,
        "score_source": score_source,
        "direction":    direction,
        "confidence":   confidence,
        "baseline":     round(baseline, 2),
        "green_area":   round(green, 2),
        "red_area":     round(red, 2),
        "ratio":        round(ratio, 4),
        "prices":       prices,
        "timestamps":   [ts for ts, _ in price_data],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bitcoin author trustworthiness.")
    parser.add_argument("--articles", required=True, help="Path to articles JSONL file")
    parser.add_argument("--output",   default="outputs/author_evaluation.csv")
    parser.add_argument(
        "--score-prompt",
        default="claude_sonnet46",
        choices=list(DEFAULT_CANDIDATES.keys()),
        help="GEPA candidate whose system prompt is used to score articles "
             "missing 'gold_score'.",
    )
    parser.add_argument(
        "--score-model",
        default=DEFAULT_LLM_MODEL,
        help="litellm model id for on-the-fly scoring (default: %(default)s).",
    )
    args = parser.parse_args()

    articles = []
    with open(args.articles, "r") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    print(f"Evaluating {len(articles)} articles...\n")

    needs_scoring = any("gold_score" not in a or a["gold_score"] is None for a in articles)
    system_prompt: str | None = None
    if needs_scoring:
        cand_path = Path(DEFAULT_CANDIDATES[args.score_prompt])
        if not cand_path.exists():
            raise FileNotFoundError(
                f"GEPA candidates file not found: {cand_path} "
                f"(needed because some articles have no 'gold_score')."
            )
        system_prompt = load_system_prompt(cand_path)
        print(
            f"Scoring articles without 'gold_score' via {args.score_model} "
            f"using GEPA prompt '{args.score_prompt}'."
        )
        if not (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ):
            print("  [WARN] No ANTHROPIC_API_KEY/OPENAI_API_KEY in env -- LLM call may fail.")
        print()

    results     = []
    ratios      = []
    confidences = []

    for i, article in enumerate(articles):
        result = evaluate_article(article, system_prompt, args.score_model)
        if result:
            results.append(result)
            ratios.append(result["ratio"])
            confidences.append(result["confidence"])
        if i < len(articles) - 1:
            time.sleep(API_DELAY)

    if not results:
        print("No valid results.")
        return

    simple   = trust_simple(ratios)
    weighted = trust_weighted(ratios, confidences)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # One row per article per hour -- running cumulative ratio
    fieldnames = [
        "article_id", "date", "score", "direction", "confidence",
        "hour", "timestamp", "price", "baseline",
        "cumulative_green", "cumulative_red", "running_ratio",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            article_date = next((a["date"] for a in articles if a.get("article_id") == r["article_id"]), "")
            prices_list  = r["prices"]
            timestamps   = r["timestamps"]
            baseline     = r["baseline"]
            direction    = r["direction"]
            p_up         = baseline * (1 + NEUTRAL_BAND)
            p_lo         = baseline * (1 - NEUTRAL_BAND)

            cum_green = 0.0
            cum_red   = 0.0

            for h in range(len(prices_list) - 1):
                p0  = prices_list[h]
                p1  = prices_list[h + 1]
                avg = (p0 + p1) / 2

                if direction == "bullish":
                    area = trapezoidal_segment(p0, p1, baseline)
                    if avg > baseline:
                        cum_green += area
                    elif avg < baseline:
                        cum_red += area
                elif direction == "bearish":
                    area = trapezoidal_segment(p0, p1, baseline)
                    if avg < baseline:
                        cum_green += area
                    elif avg > baseline:
                        cum_red += area
                elif direction == "neutral":
                    if p_lo <= avg <= p_up:
                        ref = p_up if avg >= baseline else p_lo
                        cum_green += trapezoidal_segment(p0, p1, ref)
                    else:
                        ref = p_up if avg > p_up else p_lo
                        cum_red += trapezoidal_segment(p0, p1, ref)

                running_ratio = calculate_ratio(cum_green, cum_red)
                ts = datetime.fromtimestamp(timestamps[h] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

                writer.writerow({
                    "article_id":      r["article_id"],
                    "date":            article_date,
                    "score":           r["score"],
                    "direction":       direction,
                    "confidence":      r["confidence"],
                    "hour":            h + 1,
                    "timestamp":       ts,
                    "price":           round(p1, 2),
                    "baseline":        round(baseline, 2),
                    "cumulative_green": round(cum_green, 2),
                    "cumulative_red":   round(cum_red, 2),
                    "running_ratio":   round(running_ratio, 4),
                })

    print(f"\n=== Author Trustworthiness ===")
    print(f"Articles evaluated: {len(results)}")
    print(f"Trust (simple):     {simple:.4f}  -- {interpret(simple)}")
    print(f"Trust (weighted):   {weighted:.4f}  -- {interpret(weighted)}")
    print(f"\nResults saved to '{args.output}'.")


if __name__ == "__main__":
    main()

"""
Author Trustworthiness Evaluator
Evaluates how trustworthy a Bitcoin news author is by comparing their
GEPA sentiment predictions against actual Bitcoin price movement.

For each article:
- Fetches hourly Bitcoin price for 30 days post-publication via Binance API
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
    python evaluate_author.py --articles articles.jsonl --output results.json

No API key required -- Binance public endpoint.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# --- Configuration ---
BINANCE_BASE = "https://api.binance.com/api/v3/klines"
EVAL_DAYS    = 30
EVAL_HOURS   = EVAL_DAYS * 24   # 720 hours
NEUTRAL_BAND = 0.05             # +-5%
API_DELAY    = 0.5              # seconds between calls

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
    Uses Binance BTCUSDT 1h klines -- no API key required.

    Args:
        date_str: Publication date "YYYY-MM-DD" or "YYYY-MM"

    Returns:
        List of (unix_timestamp_ms, close_price) tuples
    """
    if len(date_str) == 7:
        date_str = date_str + "-01"
    pub_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    start_ms = int(pub_date.timestamp()) * 1000
    end_ms   = start_ms + (EVAL_HOURS * 3600 * 1000)

    all_prices = []
    current_ms = start_ms

    # Binance returns max 1000 candles per request -- paginate if needed
    while current_ms < end_ms:
        response = httpx.get(
            BINANCE_BASE,
            params={
                "symbol":    "BTCUSDT",
                "interval":  "1h",
                "startTime": current_ms,
                "endTime":   end_ms,
                "limit":     1000,
            },
            timeout=30,
        )
        response.raise_for_status()
        candles = response.json()

        if not candles:
            break

        for candle in candles:
            open_time   = candle[0]   # ms timestamp
            close_price = float(candle[4])  # index 4 = close price
            all_prices.append((open_time, close_price))

        # Advance to next batch
        current_ms = candles[-1][0] + 3600000  # +1 hour in ms
        if len(candles) < 1000:
            break

    return all_prices


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


# --- Main ---

def evaluate_article(article: dict) -> dict:
    """Evaluate a single article and return its result."""
    article_id = article.get("article_id", "unknown")
    score      = int(article["gold_score"])
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
        "article_id": article_id,
        "score":      score,
        "direction":  direction,
        "confidence": confidence,
        "baseline":   round(baseline, 2),
        "green_area": round(green, 2),
        "red_area":   round(red, 2),
        "ratio":      round(ratio, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bitcoin author trustworthiness.")
    parser.add_argument("--articles", required=True, help="Path to articles JSONL file")
    parser.add_argument("--output",   default="outputs/author_evaluation.json")
    args = parser.parse_args()

    articles = []
    with open(args.articles, "r") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    print(f"Evaluating {len(articles)} articles...\n")

    results     = []
    ratios      = []
    confidences = []

    for i, article in enumerate(articles):
        result = evaluate_article(article)
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

    output = {
        "articles_evaluated": len(results),
        "trust_simple":       round(simple, 4),
        "trust_weighted":     round(weighted, 4),
        "interpretation": {
            "simple":   interpret(simple),
            "weighted": interpret(weighted),
        },
        "articles": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Author Trustworthiness ===")
    print(f"Articles evaluated: {len(results)}")
    print(f"Trust (simple):     {simple:.4f}  -- {interpret(simple)}")
    print(f"Trust (weighted):   {weighted:.4f}  -- {interpret(weighted)}")
    print(f"\nResults saved to '{args.output}'.")


if __name__ == "__main__":
    main()

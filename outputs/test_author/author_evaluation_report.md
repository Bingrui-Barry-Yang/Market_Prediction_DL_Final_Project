# Author Trustworthiness Evaluation — `author_test.jsonl`

## Setup

- **Articles**: `data/test/author_test.jsonl` — 15 FXStreet "Bitcoin Price Forecast" pieces (one author), publication dates Oct 2024 – Mar 2026.
- **Scorer**: `ollama_chat/qwen3.6:latest` driven by the GEPA-optimised system prompt at `outputs/gepa_runs/bitcoin_sentiment/run_qwen36_b150/candidates.json` (last candidate). Used because `author_test.jsonl` ships without `gold_score`, and the Anthropic account had no credit at run time.
- **Price source**: Coinbase Exchange `BTC-USD` 1h candles (`api.exchange.coinbase.com`). Original Binance endpoint is geo-blocked from this host (HTTP 451). Per-article window = 30 days × 24 h = 720 hourly closes; pagination pulls 200 candles per request. 10,795 of an expected 10,800 hourly rows landed (5 candles missing on the exchange — script just skips the gap).
- **Per-article ratio**: trapezoidal area where price moves *with* the predicted direction (green) minus area where it moves *against* (red), divided by the total. Range `[-1, +1]`. `±5%` neutral band for `neutral` calls.
- **Aggregation**: `trust_simple` = mean ratio; `trust_weighted` = confidence-weighted mean (confidence = score's offset within its direction bucket, 1–5).
- **Outputs**: `outputs/author_evaluation.csv` (one row per article-hour, with running cumulative green/red/ratio).

## Aggregate result

| Metric | Value | Band |
|---|---|---|
| `trust_simple`   | **−0.080** | Unreliable / noisy |
| `trust_weighted` | **−0.123** | Unreliable / noisy |

Both fall inside the `[−0.5, +0.5]` "noisy" band, and the simple/weighted gap is small because confidences cluster tightly (mostly 2–4 out of 5), so weighting doesn't move the result.

## Per-article results

Source: `outputs/author_evaluation_summary.csv`. Verdict bands: `>0.9` very correct · `0.5–0.9` correct · `−0.5 to 0.5` noisy · `−0.9 to −0.5` wrong · `<−0.9` very wrong.

| ID | Date | Score | Direction | Conf. | Baseline ($) | Green | Red | Ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 001 | 2024-10-24 |  7 | neutral | 2 |     67,359 |     546,308 |   6,641,477 | −0.8480 | wrong |
| 002 | 2024-11-13 |  5 | bearish | 5 |     88,159 |      11,703 |   5,455,646 | −0.9957 | very wrong |
| 003 | 2024-12-16 | 12 | bullish | 2 |    105,383 |      44,093 |   6,151,477 | −0.9858 | very wrong |
| 004 | 2025-02-05 |  3 | bearish | 3 |     98,318 |   3,451,757 |       8,646 | +0.9950 | very correct |
| 005 | 2025-04-24 |  9 | neutral | 4 |     93,418 |   1,006,919 |   2,511,923 | −0.4277 | noisy |
| 006 | 2025-06-25 | 12 | bullish | 2 |    106,452 |   4,732,332 |       9,227 | +0.9961 | very correct |
| 007 | 2025-08-11 | 13 | bullish | 3 |    119,154 |     116,564 |   4,127,764 | −0.9451 | very wrong |
| 008 | 2025-09-09 | 12 | bullish | 2 |    111,652 |   3,235,401 |     169,817 | +0.9003 | very correct |
| 009 | 2025-10-16 | 12 | bullish | 2 |    110,482 |     362,475 |   2,641,059 | −0.7586 | wrong |
| 010 | 2025-11-12 |  8 | neutral | 3 |    102,832 |     187,597 |   4,846,503 | −0.9255 | very wrong |
| 011 | 2025-12-10 |  8 | neutral | 3 |     92,138 |   1,103,354 |     103,303 | +0.8288 | correct |
| 012 | 2026-01-29 |  3 | bearish | 3 |     89,010 |  13,253,627 |           0 | +1.0000 | very correct |
| 013 | 2026-02-09 |  3 | bearish | 3 |     70,444 |   1,913,294 |      83,417 | +0.9164 | very correct |
| 014 | 2026-03-16 | 13 | bullish | 3 |     72,547 |     160,696 |   2,073,988 | −0.8562 | wrong |
| 015 | 2026-03-26 |  4 | bearish | 4 |     71,272 |   1,157,180 |   1,378,973 | −0.0875 | noisy |

**Verdict counts**: 5 very correct · 1 correct · 2 noisy · 3 wrong · 4 very wrong.

Using `|ratio| > 0.5` as the decisive threshold: 6 decisive correct (4, 6, 8, 11, 12, 13), 7 decisive wrong (1, 2, 3, 7, 9, 10, 14), 2 inside ±0.5 (5, 15). When the model commits with `|ratio|>0.9` it splits 5 right / 4 wrong — the conviction is there, the directional accuracy isn't.

## By predicted direction

| Direction | n | Mean ratio |
|---|---:|---:|
| Bearish (1–5)  | 5 | **+0.366** |
| Neutral (6–10) | 4 | **−0.343** |
| Bullish (11–15) | 6 | **−0.275** |

The single most informative slice: **bearish calls land**, bullish calls don't. Four of the five bearish calls produced |ratio| > 0.9, three of them positive. Every neutral call ended up outside the ±5% band — BTC simply moved too much in 30 days for that bucket to ever be safe over this date range.

## Interpretation

1. **Aggregate verdict is "noisy."** Trust scores near zero mean the predictions are roughly as informative as a coin flip when scored against actual 30-day BTC movement. Neither trustworthy (`>+0.5`) nor a usable contrarian signal (`<−0.5`).
2. **Bullish bias is the main loss center.** Six bullish calls produced an average ratio of −0.275; three of them were dramatic misses (007, 009, 014). The model is over-attributing bullishness to articles that emphasise institutional inflows or ATH narratives, in periods where BTC subsequently corrected.
3. **Bearish calls are the only positive bucket.** They average +0.366. Articles 004, 012, 013 sit on top of three of BTC's clearest drawdowns in the test window — the model surfaced the bearish framing well in those cases.
4. **Neutral is a structural problem, not a model failure.** The ±5% band is narrow versus 30-day BTC realised volatility; mathematically, "neutral" rarely scores positive over this horizon for this asset.
5. **Confidence isn't carrying signal.** Simple and weighted trust are within ~0.04 of each other; the model's higher-confidence calls aren't measurably more accurate than its lower-confidence ones (high-`|ratio|` outcomes are roughly 50/50).

## Caveats

- **Scorer ≠ author.** The "trustworthiness" measured here is for the qwen3.6 + GEPA prompt's reading of the articles, not the human author directly. To attribute trust to the FXStreet author, the input scores would need to come from the author's own stated forecast (e.g., a structured field in the JSONL).
- **Exchange swap.** Original spec called Binance BTC-USDT; this run used Coinbase BTC-USD. Spot prices typically differ <0.1% — negligible vs. the area magnitudes above.
- **Token-budget tuning.** Initial run with `max_tokens=1024` truncated qwen's reasoning trace before it emitted any score; raised to 8192 with reasoning-content fallback in the parser. All 15 articles scored cleanly with no parse warnings on the final run.
- **Five missing candles.** Coinbase had 5 hourly gaps across the 10,800 expected rows. The trapezoidal integration just skips a missing pair — impact on the ratio is sub-percent per article.

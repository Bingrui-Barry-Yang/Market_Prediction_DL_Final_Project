# Claude API Token & Cost Estimate — GEPA + Test Evaluation

**Date:** 2026-04-24  
**Method:** Live measurements via `client.messages.count_tokens()` (input) and real API calls on `claude-haiku-4-5` (output)  
**Scope:** One full GEPA training run (budget=150) + test evaluation on ~40 articles

---

## Summary

| Model | Total Cost |
|---|---|
| claude-opus-4-7 | **$4.50** |
| claude-sonnet-4-6 | **$2.70** |
| claude-haiku-4-5 | **$0.90** |

A full GEPA run plus test evaluation costs under $5 even on Opus 4.7. Input tokens dominate; output is nearly free because the task LM only ever returns a short JSON score.

---

## What Was Measured

The smoketest (`scripts/smoketest_claude_token_cost.py`) made **5 real API calls** and **10 token-count calls** against actual article and reflection-prompt data from the completed `gptoss120b_b150` run.

| Measurement | Method | Calls |
|---|---|---|
| Task LM input tokens | `count_tokens()` on 5 articles × 2 prompts | 10 |
| Task LM output tokens | Real `messages.create()` on 3 articles | 3 |
| Reflection LM input tokens | `count_tokens()` on 5 reflection prompts | 5 |
| Reflection LM output tokens | Real `messages.create()` on 2 reflection prompts | 2 |
| Test eval input tokens | `count_tokens()` on 3 test-sized articles | 3 |

Total smoketest cost (all on `claude-haiku-4-5`): well under $0.01.

---

## Measured Averages

### Task LM (scores articles 1–15)

Articles were sampled at the 0th, 25th, 50th, 75th, and 100th percentile of character length (1,998–45,007 chars).

| Prompt type | Avg input tokens | Avg output tokens |
|---|---|---|
| Seed prompt (638 chars) | 3,230 tok | 11 tok |
| Median evolved prompt (~5,800 chars) | 4,564 tok | 11 tok |
| Blended (6 seed + 96 evolved calls) | 4,486 tok | 11 tok |

Output is consistently **~11 tokens** — the model returns only `{"score": <integer>}`.

The jump from seed to evolved prompts (+1,334 tok input) is because GEPA's evolved prompts include detailed scoring rubrics and article-type guidance, roughly 9× longer than the seed.

### Reflection LM (proposes new prompts)

Reflection prompts were sampled at the 0th, 25th, 50th, 75th, and 100th percentile of character length (12,802–59,795 chars). These prompts contain the full task description, all training examples with model responses, and scoring feedback.

| Percentile | Chars | Input tokens |
|---|---|---|
| 0th (smallest) | 12,802 | 3,392 |
| 25th | 18,499 | 4,341 |
| 50th (median) | 23,570 | 5,729 |
| 75th | 30,176 | 7,655 |
| 100th (largest) | 59,795 | 14,349 |
| **Average** | **28,969** | **7,093** |

| Avg output tokens (2 real calls) | 572 tok |
|---|---|

Output length reflects the newly generated system prompt (~500–700 tok for a detailed prompt).

### Test Evaluation

Sampled at small/median/large. Uses the best evolved prompt as the system prompt.

| Avg input tokens | 6,129 tok |
|---|---|
| Avg output tokens | 11 tok |

---

## Token Projections — Full Run

Based on the completed `gptoss120b_b150` run: 102 task LM calls, 19 reflection LM calls, 40 test articles.

| Component | Input tokens | Output tokens |
|---|---|---|
| GEPA Task LM | 457,564 | 1,088 |
| GEPA Reflection LM | 134,770 | 10,868 |
| Test Evaluation | 245,173 | 426 |
| **Total** | **837,507** | **12,382** |

Input tokens are ~67× greater than output tokens. Pricing is almost entirely driven by input costs.

---

## Cost Estimates

### Total cost

| Model | Input (837K tok) | Output (12K tok) | **Total** |
|---|---|---|---|
| claude-opus-4-7 ($5/$25 per MTok) | $4.19 | $0.31 | **$4.50** |
| claude-sonnet-4-6 ($3/$15 per MTok) | $2.51 | $0.19 | **$2.70** |
| claude-haiku-4-5 ($1/$5 per MTok) | $0.84 | $0.06 | **$0.90** |

### Per-phase cost

| Phase | Opus 4.7 | Sonnet 4.6 | Haiku 4.5 |
|---|---|---|---|
| GEPA Task LM | $2.32 | $1.39 | $0.46 |
| GEPA Reflection LM | $0.95 | $0.57 | $0.19 |
| Test Evaluation | $1.24 | $0.74 | $0.25 |
| **Total** | **$4.50** | **$2.70** | **$0.90** |

---

## Key Observations

**Output tokens are negligible.** At $0.31 on Opus 4.7, output is only 7% of total cost. The task LM produces a 2–5 word JSON response every call; the reflection LM produces ~500 tokens per call. You could run 10× more GEPA iterations before output cost becomes meaningful.

**Evolved prompts cost more than the seed.** The seed prompt is 638 chars; evolved prompts average ~5,800 chars. This adds ~1,334 input tokens per task LM call — meaningful when multiplied across 96 evolved-prompt calls (+128K tokens total for training).

**Reflection LM is not the dominant cost.** Despite the large prompts (avg 7,093 tok input), there are only 19 reflection calls. They account for ~16% of total input tokens and ~22% of total cost on Opus 4.7.

**Test evaluation is relatively expensive.** 40 articles × 6,129 tok avg = 245K input tokens — almost as much as the GEPA task LM phase (458K). If you use a longer evolved prompt for test eval, this cost scales proportionally.

**The one 45K-char article is an outlier.** That article alone generates ~10,934 input tokens with the seed prompt and ~12,268 with evolved prompts. If a similar article appears in the test set, one call costs ~12× the average.

---

## Assumptions & Limitations

- **Run structure** extrapolated from the completed `gptoss120b_b150` run: 20 iterations, 6 accepted proposals, 102 total task calls, 19 reflection calls.
- **Test articles** assumed similar in length to training articles (no `data/test/articles.jsonl` available at measurement time; training set used as proxy).
- **Reflection output** measured on the two smallest prompts only (to minimize smoketest cost). Larger reflection prompts may produce longer outputs — the 572-tok average may undercount by 10–20%.
- **No prompt caching.** If the system prompt is stable across calls, Anthropic prompt caching could reduce input costs by ~50–90% on the fixed prefix. This report does not account for that.

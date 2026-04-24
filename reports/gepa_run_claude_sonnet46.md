# GEPA Run Report — Claude Sonnet 4.6

**Date:** 2026-04-24  
**Script:** `scripts/run_gepa_rate_limited.py`  
**Task LM:** `anthropic/claude-sonnet-4-6`  
**Reflection LM:** `anthropic/claude-sonnet-4-6`  
**Result file:** `outputs/gepa_runs/gepa_result_claude_sonnet46.json`

---

## Summary

GEPA successfully evolved the Bitcoin sentiment scoring prompt over 15 iterations, improving validation accuracy from **0.792 → 0.833** (+5.3 pp). The best prompt is 4,148 characters — 6.5× longer than the 638-character seed — and adds concrete scoring rules and domain-specific calibration derived from training errors.

| Metric | Value |
|---|---|
| Iterations completed | 15 |
| Budget (max metric calls) | 150 |
| Candidates accepted into pool | 10 (+1 seed = 11 total) |
| Reflection LM calls logged | 14 |
| Seed val score | 0.792 |
| **Best val score** | **0.833** |
| Best candidate index | 9 |
| Best prompt length | 4,148 chars |

---

## Iteration-by-Iteration Val Scores

| Iteration | Candidate accepted | Val score |
|---|---|---|
| seed | 0 | 0.792 |
| 1 | 1 | 0.792 |
| 2 | 2 | 0.792 |
| 3 | 3 | 0.708 |
| 4 | 4 | 0.750 |
| 5 | — (skipped) | — |
| 6 | 5 | 0.792 |
| 7 | 6 | 0.792 |
| 8 | — (skipped) | — |
| 9 | 7 | 0.792 |
| 10 | — (skipped) | — |
| 11 | — (skipped) | — |
| 12 | 8 | 0.750 |
| 13 | — (skipped) | — |
| **14** | **9** | **0.833** ← best |
| 15 | 10 | 0.792 |

The best candidate (index 9, 4,148 chars) emerged at iteration 14 — relatively late in the run. Candidates 3 and 4 (iterations 3–4) were weaker than the seed, suggesting early mutations overfit. The run stabilised around 0.792 through iterations 1–13 before finding the breakthrough in iteration 14.

---

## What GEPA Changed

The best evolved prompt preserves the seed's structure (1–15 scale, bullish/bearish/neutral bands, forward-looking vs. retrospective distinction) and adds:

**Hedged language penalty**  
Articles using "could," "may," "might," or "is likely to" are capped at score 12 even if the direction is bullish.

**Risk-factor adjustment rule**  
A primarily bullish article with meaningful risk factors (geopolitical, regulatory, macro) is reduced by 1. A primarily bearish article with recovery scenarios is increased by 1.

**Technical pattern guidance**  
Double-top and head-and-shoulders patterns are labelled "moderately bearish, not strongly bearish." Mixed technical signals (some buy, some sell indicators) default to neutral or weakly directional.

**Calibrated price-target ranges**  
$100K–$140K targets score ~13–14. Targets above $140K with strong consensus may warrant 14–15. Neither automatically triggers a 15.

**"Big move either way" framing**  
Consolidation articles with genuine two-directional uncertainty score neutral-to-mildly bullish (12–13 only if institutional accumulation is emphasized alongside uncertainty).

**Domain-specific anchors**  
Three named calibration examples provide concrete ground truth: "Uptober" framing (net bearish if combined with macro uncertainty), September seasonal drops (score 2–3), and $140K multi-analyst consensus (score 13–14).

---

## Reliability Notes

The run hit **persistent connection timeouts** (60s each) during the iteration-13 reflection call, exhausting 8 of 9 retry attempts. The call ultimately succeeded on the 9th attempt — the run did not crash. Rate-limit errors (429) appeared throughout but were handled with 2–30s exponential backoff.

**Recommendation for future runs:** Use `--request-timeout 300` when running large reflection prompts against Claude Sonnet. By iteration 12–13 the reflection context (all training examples + all prior feedback) grows to ~10K–14K tokens and can take longer than 60 seconds to return.

---

## Cost Estimate

Based on the smoketest in `reports/claude_token_cost_estimate.md` (Sonnet 4.6: $3/$15 per MTok input/output):

| Phase | Estimated cost |
|---|---|
| GEPA training (15 iterations, ~102 task calls, 14 reflection calls) | ~$2.00 |
| Test evaluation (40 articles, best prompt) | ~$0.74 |
| **Total** | **~$2.74** |

Actual cost may differ slightly — 15 iterations with 11 accepted candidates involves more val-set evaluations than the baseline projection assumed.

---

## Next Steps

1. **Run test evaluation** — score the held-out test set with candidate 9 and compare to baseline.
2. **Rerun with `--request-timeout 300`** — if another full run is needed, prevents the late-stage timeout risk.
3. **Compare against other GEPA runs** — the `gptoss120b_b150` run (local model) achieved similar val scores; a head-to-head on the test set will show whether Sonnet's evolved prompt transfers better to unseen articles.

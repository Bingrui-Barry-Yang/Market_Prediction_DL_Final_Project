# qwen36 task model — partial run, killed early

**Date killed:** 2026-04-29
**Status:** 805 / 1,330 rows captured (60.5 %), 662 parsed (82 % parse rate over rows that landed)

## Per-source breakdown

| Prompt source | Cells | Rows | Parsed | Parse rate |
|---|---:|---:|---:|---:|
| `claude_sonnet46` | 11/11 | 385 | 376 | 98 % |
| `gptoss120b`      | 7/7   | 245 | 111 | 45 % |
| `qwen36`          | 5/11  | 175 | 175 | 100 % |
| `gemma4e2b`       | 0/9   | 0   | 0   | n/a |

## Why this run was killed instead of left to finish

At the time of killing, the container had been running ~22 hours and was still
working on qwen-source cells. Per-call wall-clock had grown from ~50 s (warm,
on clean prompts) to ~700 s (on the GEPA-evolved gpt-oss prompts), with ~705
calls still queued. Extrapolating at the current rate would have meant
**~6 more days** of compute that would mostly land at `parse_status=failed`
and `gepa_score=0.0`. The other three task models had already completed at
100 % (`claude_sonnet46`, `gptoss120b`, `gemma4e2b`), so blocking analysis on
qwen completion wasn't worth the cost.

## What's actually wrong (three layers)

### 1. The 700-second per-call latency is the script's retry budget, not generation time

`scripts/run_test_eval.py` calls
`src/evaluation/scoring.py:litellm_completion_with_retries` with
`num_retries=8`, `base_backoff=2.0`, `max_backoff=30.0`, `timeout=60`. On a
hung call, the budget unfolds as:

- 8 attempts × 60 s timeout = 480 s
- 7 backoff sleeps = 2 + 5 + 10 + 17 + 31 + 31 + 31 ≈ 127 s
- final attempt (~60 s)

≈ **~670 s of wasted wall-clock** per failing call, with no information
recovered. Inspecting failed rows confirms this: `tokens_out: None`,
`raw_response: ""`, `latency_ms` ≈ 700,000.

### 2. Raising `--timeout` does NOT help

Higher timeout would only matter if qwen were slowly generating tokens that
truncate at 60 s. The data shows the opposite: qwen returns **zero** tokens on
the failing prompts, not a partial response. The hang is inside Ollama's
buffering of qwen3's thinking-mode output, not progress that just needs more
time. Raising `--timeout 60` to `--timeout 600` would convert "8 fast empty
attempts" into "1 slow empty attempt" — same outcome, similar wall-clock.

### 3. Why specifically the gpt-oss-source prompts fail (and not claude- or qwen-source)

qwen3.6 ships with an auto-triggered "thinking mode" that emits hidden
`<think>...</think>` reasoning before producing visible output. Ollama
serves qwen3 with thinking enabled by default. The trigger pattern is
*procedural chain-of-thought instruction language* — phrases like *"identify
all forward-looking statements... weight them... synthesize... determine the
core prediction"*.

GEPA-evolved prompts differ by source model:

- **`claude_sonnet46` source** — Claude is concise; its evolved prompts are
  instruction-style ("score this 1-15 using these criteria"), not procedural.
  qwen reads them as classification tasks → no thinking trigger → **98 %
  parse rate**.
- **`gptoss120b` source** — gpt-oss:120b is itself a heavy-reasoning model;
  GEPA evolved prompts that lean on explicit chain-of-thought instructions.
  Lexical scan: every gpt-oss prompt contains a "show_reasoning" phrase
  (e.g. *"identify… then weight… then synthesize"*). qwen sees this and
  enters thinking mode → response either hangs or is malformed → **45 %
  parse rate**.
- **`qwen36` source** — co-evolved with qwen during training (which had
  thinking enabled by default), so these prompts are tuned to qwen's
  thinking behavior → **100 % parse rate** (on the 5 cells that ran).
- **`gemma4e2b` source** — never reached in this run. Prompts are short
  (≤ 379 words) and lexically clean of "show_reasoning" trigger phrases, so
  prediction is they would have worked, but **this is unverified**.

## What data here is still useful

The 805 rows are not garbage:

- The 376 parsed rows on `claude_sonnet46` × `qwen36` (98 % parse) are a
  clean cell for cross-model paired analysis.
- The 175 parsed rows on `qwen36` × `qwen36` (100 % parse, 5 of 11 prompts)
  are a partial within-model comparison.
- The 111 parsed rows on `gptoss120b` × `qwen36` (45 % parse) document the
  failure mode itself — `parse_status=failed` rows with `gepa_score=0.0` are
  themselves a measurement of "this prompt × task pairing doesn't transfer."

## What it would take to complete qwen cleanly

Two changes, either of which fixes the gpt-oss-source failures:

1. **Disable thinking mode** — add `extra_body={"think": False}` to the
   `litellm.completion` call in `src/evaluation/scoring.py:124` and prefix
   the system prompt with `/no_think`. Estimated re-run cost: ~3-5 hours
   for the full 1,330 qwen rows on warm Ollama.
2. **Make the parser tolerant** — change `parse_score_response` to extract
   the first `{"score": <int 1-15>}` match anywhere in `raw_response`
   instead of only at the start. Salvages many of the rows we already have
   without re-running, *if* the response actually contained a JSON answer
   somewhere (which the empty-response rows on `gptoss120b` p03+ did not —
   those still need a re-run after disabling thinking).

For the current writeup, neither change is strictly required — the partial
data plus this note characterizes the qwen3.6 × verbose-prompt failure
mode, which is itself a finding.

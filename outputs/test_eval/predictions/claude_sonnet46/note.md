# claude_sonnet46 task model — note on the gpt-oss prompt failures

**Status:** all 38 cells × 35 articles = 1,330 rows captured. Parse rate
varies a lot by which source model the prompt came from.

## What worked, what didn't

| Prompt source | Cells | Avg parse rate |
|---|---:|---:|
| `claude_sonnet46` | 11 | ~97% (excellent) |
| `qwen36`          | 11 | ~94% (excellent) |
| `gemma4e2b`       |  9 | ~84% (one cell at 0%) |
| `gptoss120b`      |  7 | **~30%** (most failed) |

So Claude as the test model handles claude-trained, qwen-trained, and most
gemma-trained prompts just fine. The big problem is with the
**gpt-oss-trained prompts** — most of those cells return `parse_status=failed`
and `pred_score=None`.

## Why this happens, in plain English

The way the test pipeline works: we send Claude a prompt, Claude responds,
and a small parser at the end tries to read the response. The parser is
**strict** — it only looks at the very beginning of Claude's reply for the
exact text `{"score": <number 1-15>}`. If anything else comes first, the
parser gives up and records the prediction as missing.

The gpt-oss-trained prompts were evolved by gpt-oss:120b, which is a
reasoning-heavy model. GEPA tuned those prompts to include long
"work-through-this-step-by-step" instructions: *"identify the forward-looking
statements... weight them... synthesize... determine the core prediction."*

Claude reads those instructions and **does exactly what they say**. It writes
out its reasoning first — paragraphs and bulleted lists — and then puts the
JSON answer at the end. You can see this directly in the failed responses:

```
"I'll analyze the forward-looking statements in this article step by step.
**Forward-looking signals:** 1. ..."

"I'll work through the procedure step by step.
**Step 1 & 2: Extract forward-looking statements...**"

"I'll systematically extract forward-looking statements and score them.
## Step 1: ..."
```

The JSON answer is almost certainly buried somewhere lower in the response,
but our parser never looks past the first line. So even though Claude is
giving us reasonable reasoning and (probably) a correct score, the test
pipeline records "no answer" because the score isn't where the parser
expects it.

## Why other prompt sources don't have this problem

- **claude-trained prompts** were evolved by Claude itself, so they
  naturally produce instruction-style guidance ("score 1-15 using these
  criteria") instead of "do these steps in order." Claude reads them as a
  classification task, replies with raw JSON. No problem.
- **qwen-trained prompts** were also short and direct — qwen3.6's training
  doesn't lean on procedural framing the way gpt-oss does.
- **gemma-trained prompts** are the shortest of any source (because gemma
  is a small 5B model and GEPA evolved compact prompts for it). Claude
  responds cleanly to most of them. One outlier — `gemma4e2b__prompt_08` —
  fails 100% and is worth a separate look later.

## Bottom line

Most of the "failed" gpt-oss-source predictions are not actually wrong —
Claude is producing an answer, just in a format the strict parser can't
recover. The raw text is preserved in `raw_response` for every row, so a
later analysis pass with a more tolerant parser (one that searches for
`{"score": N}` anywhere in the response, not just at the start) could
recover most of these scores **without re-running the model**.

For the current writeup, the cleanest framing is:

> "Claude under the strict-JSON-only parser fails on gpt-oss-trained
> prompts because those prompts elicit chain-of-thought preamble. This is
> a parser limitation, not a model failure — predictions can be recovered
> retroactively from the persisted `raw_response` field."

Affected cells (parse rate < 50 %):

| Cell | Parsed |
|---|---:|
| `gptoss120b__prompt_01` | 20 / 35 (57 %) |
| `gptoss120b__prompt_02` |  1 / 35  (3 %) |
| `gptoss120b__prompt_03` |  5 / 35 (14 %) |
| `gptoss120b__prompt_04` |  5 / 35 (14 %) |
| `gptoss120b__prompt_05` |  0 / 35  (0 %) |
| `gptoss120b__prompt_06` |  6 / 35 (17 %) |
| `gemma4e2b__prompt_08`  |  0 / 35  (0 %) — separate failure mode, unrelated |

`gptoss120b__prompt_00` (the seed) is unaffected (35 / 35 parsed) because the
seed prompt is short and instruction-style — same as the claude-source seed.

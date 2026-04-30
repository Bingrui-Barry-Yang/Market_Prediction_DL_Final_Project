# AGENTS.md

## Project Goal

This project has been restructured into a Bitcoin news prompt optimization study.

The old trust-weighted Bitcoin trend prediction pipeline is obsolete. Do not reintroduce historical scraping, author trust modeling, FastAPI serving, Streamlit dashboards, Temporal Fusion Transformers, or production trading workflows unless the user explicitly asks for them.

## Research Question

Is an optimal prompt effective at extracting structured prediction information from existing Bitcoin news articles?

## Current Project Design

The project uses:

- Python `>=3.10.19`
- `uv` for dependency and environment management
- Docker and Docker Compose for reproducible execution
- JSONL for manually curated article datasets
- a clean Python monorepo layout under `src/`

The repository is a compact research workflow, not a production inference service.

## Pipeline Stages

### Stage 1: GEPA Training

Use 30 manually curated Bitcoin price prediction articles to optimize extraction prompts.

Input:

- `data/train/articles.jsonl`

Each article contains reserved full text, title, URL, source, month-level date, one integrated `gold_score`, and gold reasoning.

Output:

- prompt candidates
- per-generation run metadata
- selected best prompt per model, and optionally one overall best prompt
- artifacts under `outputs/gepa_runs/`

### Stage 2: Test Evaluation

Run every prompt candidate from every completed GEPA source against every configured task model on the held-out test set. Implemented in `scripts/run_test_eval.py` with model and prompt registry in `src/evaluation/registry.py` and scoring helpers in `src/evaluation/scoring.py`.

Input:

- `data/test/articles_test.jsonl`
- `candidates.json` and `result_*.json` files under `outputs/gepa_runs/bitcoin_sentiment/<source_run>/`

Output:

- per-prediction JSONL rows (`predictions/<task_slug>/<source_slug>__prompt_NN.jsonl`)
- per-cell aggregate metrics (`metrics/<task_slug>/<source_slug>__prompt_NN.json`)
- long-format `per_article_scores.jsonl`
- wide `score_matrix.csv`, `error_matrix.csv`, `parse_matrix.csv`
- `summary.json`, `prompts_index.json`, `run_log.jsonl`

Output location:

- `outputs/test_eval/`

### Stage 3: Optional Real-World Validation

Optionally select articles from specific news sources across time periods, use the best prompt to extract predicted viewpoints, and compare those predictions with actual Bitcoin price movement.

This stage should stay optional and isolated under `src/validation/` and `outputs/validation/`.

### Stage 4: Final Analysis

Produce final tables and written summaries comparing prompt and model performance.

Output:

- final analysis artifacts under `outputs/reports/`

## Target Models

The GEPA runner uses LiteLLM and accepts any LiteLLM-compatible model id via `--task-lm` / `--reflection-lm`. The script default is:

- `gemini/gemini-2.5-flash-lite`

Test evaluation iterates a fixed registry of task models (`src/evaluation/registry.py`):

- `anthropic/claude-sonnet-4-6`
- `ollama_chat/gpt-oss:120b`
- `ollama_chat/qwen3.6:latest`
- `ollama_chat/gemma4:e2b`

Source models (those whose GEPA candidates are evaluated) are the same four. Do not require live model credentials for unit tests.

## Data Contracts

Canonical article records are JSONL objects:

```json
{
  "article_id": "article-001",
  "text": "Full article text...",
  "title": "Bitcoin price prediction article title",
  "url": "https://example.com/article",
  "source": "Example News",
  "date": "2024-03",
  "gold_score": 15,
  "gold_reasoning": "The article argues that institutional demand will push BTC higher."
}
```

The `text` field contains full article text from the curated worksheet.

Human annotations include both direction and confidence, but the canonical JSONL maps those two fields together into one `gold_score`:

- direction `-1` with confidence `1-5` maps to scores `1-5`
- direction `0` with confidence `1-5` maps to scores `6-10`, so confidence `1` gives score `6`
- direction `1` with confidence `1-5` maps to scores `11-15`

Do not emit separate `gold_direction` or `gold_confidence` fields in the converted JSONL.

For GEPA runs, use the canonical `data/train/articles.jsonl`. For test evaluation, use `data/test/articles_test.jsonl`.

Canonical per-prediction records (one per (task_model, source_model, prompt_idx, article_id) cell, written by `scripts/run_test_eval.py`) include:

```json
{
  "article_id": "btc-gepa-test-001",
  "task_model": "anthropic/claude-sonnet-4-6",
  "task_model_slug": "claude_sonnet46",
  "source_model_slug": "gptoss120b",
  "prompt_idx": 0,
  "is_seed": true,
  "is_best": false,
  "prompt_sha256": "...",
  "gold_score": 13,
  "pred_score": 12,
  "abs_error": 1,
  "signed_error": -1,
  "gepa_score": 0.75,
  "exact_match": false,
  "direction_correct": true,
  "gold_band": "bullish",
  "pred_band": "bullish",
  "parse_status": "ok",
  "parse_error": null,
  "raw_response": "{\"score\": 12}",
  "latency_ms": 3247,
  "tokens_in": 2476,
  "tokens_out": 6,
  "n_retries": 0,
  "timestamp": "2026-04-27T20:20:30Z"
}
```

## Architecture Rules

- Keep business logic in `src/`.
- Keep executable stage runners in `scripts/`.
- Keep source datasets under `data/`.
- Keep generated artifacts under `outputs/`.
- Keep Docker setup simple: one `research` service is enough.
- Keep schemas explicit with Pydantic.
- Keep GEPA logic independent of any one model provider.
- Keep prompt versions and run metadata reproducible.
- Keep tests small and fast.

## Module Responsibilities

```text
src/common/       JSONL helpers, logging, shared utilities
src/data/         article schema (Pydantic ArticleRecord)
src/evaluation/   test-eval scoring helpers, per-cell aggregate metrics,
                  task/source model registry, prompt candidate loader
```

Future modules may be added as new pipeline stages are introduced. Keep business logic in `src/`, executable runners in `scripts/`, source datasets in `data/`, generated artifacts in `outputs/`, and post-run analysis artifacts in `reports/`.

## Testing Expectations

Add or maintain tests for:

- JSONL schema validation
- prompt rendering
- structured output parsing
- metric calculations
- GEPA candidate selection behavior
- script-level smoke tests with mocked model outputs

Do not require live model credentials for unit tests.

## Environment

Use Python `>=3.10.19`.

Use:

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY (for GEPA training with Gemini)
# and ANTHROPIC_API_KEY (for test evaluation with Claude task models)
uv sync
uv run python -m pytest
uv run python -m ruff check .
```

Use Docker for reproducible execution:

```bash
docker compose build research
docker compose run --rm research uv run python scripts/run_gepa.py \
  --dry-run \
  data/train/articles.jsonl
```

Run a Docker GEPA smoke test:

```bash
docker compose run --rm research uv run python scripts/run_gepa.py \
  --budget 1 \
  --output outputs/gepa_runs/bitcoin_sentiment/smoke_result_001.json \
  --run-dir outputs/gepa_runs/bitcoin_sentiment/smoke_run_001 \
  data/train/articles.jsonl
```

## Implementation Priorities

1. Keep the repo simple and modular.
2. Preserve dataset and output boundaries.
3. Prefer deterministic local tests over live API calls.
4. Make model/provider integrations replaceable.
5. Keep optional real-world validation separate from the core GEPA and test-evaluation workflow.

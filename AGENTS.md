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

Each article contains full text, URL, source, month-level date, gold direction, gold confidence, and gold reasoning.

Output:

- prompt candidates
- per-generation run metadata
- selected best prompt per model, and optionally one overall best prompt
- artifacts under `outputs/gepa_runs/`

### Stage 2: Test Evaluation

Evaluate the best prompt on a separate unseen dataset of 35-40 articles.

Input:

- `data/test/articles.jsonl`
- selected prompt artifacts from `outputs/gepa_runs/`

Output:

- model predictions
- parse status
- direction accuracy
- macro F1
- confidence agreement or correlation
- parse failure rate
- comparison tables under `outputs/evaluations/`

### Stage 3: Optional Real-World Validation

Optionally select articles from specific news sources across time periods, use the best prompt to extract predicted viewpoints, and compare those predictions with actual Bitcoin price movement.

This stage should stay optional and isolated under `src/validation/` and `outputs/validation/`.

### Stage 4: Final Analysis

Produce final tables and written summaries comparing prompt and model performance.

Output:

- final analysis artifacts under `outputs/reports/`

## Target Models

Initial model targets are:

- Qwen
- Kimi
- GPT-OSS 120B

Implement model access through provider-agnostic adapters in `src/models/`. Do not hardcode provider-specific behavior into GEPA, evaluation, or analysis code. Adapter configuration should come from settings or environment variables.

## Data Contracts

Canonical article records are JSONL objects:

```json
{
  "article_id": "article-001",
  "text": "Full article text...",
  "url": "https://example.com/article",
  "source": "Example News",
  "date": "2024-03",
  "gold_direction": "up",
  "gold_confidence": "high",
  "gold_reasoning": "The article argues that institutional demand will push BTC higher."
}
```

Allowed direction labels:

- `up`
- `down`
- `neutral`

Initial allowed confidence labels:

- `low`
- `medium`
- `high`

Canonical extraction records should include:

```json
{
  "article_id": "article-001",
  "model_name": "qwen",
  "prompt_version": "prompt-v001",
  "pred_direction": "up",
  "pred_confidence": "high",
  "pred_reasoning": "The article expects stronger demand to increase BTC price.",
  "raw_response": "{...}",
  "parse_status": "ok"
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

## Recommended Module Responsibilities

```text
src/common/       logging, JSONL helpers, shared utilities
src/config/       settings and environment parsing
src/data/         article and extraction schemas, dataset loading
src/prompts/      prompt templates and prompt rendering
src/models/       LLM adapter protocols and provider implementations
src/gepa/         prompt optimization loop and candidate selection
src/extraction/   structured response parsing and normalization
src/evaluation/   metrics and model comparison
src/validation/   optional real-world BTC price movement validation
src/analysis/      final report generation
```

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
uv sync
uv run pytest
```

Use Docker for reproducible execution:

```bash
docker compose run --rm research
```

## Implementation Priorities

1. Keep the repo simple and modular.
2. Preserve dataset and output boundaries.
3. Prefer deterministic local tests over live API calls.
4. Make model/provider integrations replaceable.
5. Keep optional real-world validation separate from the core GEPA and test-evaluation workflow.

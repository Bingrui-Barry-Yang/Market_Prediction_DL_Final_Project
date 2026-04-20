# Bitcoin News Prompt Optimization Study

## Research Question

Is an optimal prompt effective at extracting structured prediction information from existing Bitcoin news articles?

This project studies whether GEPA-based prompt optimization can improve how large language models extract annotated Bitcoin price prediction viewpoints from manually curated news articles. The project is intentionally compact: it is a reproducible research pipeline, not a live trading or production forecasting system.

## Project Description

The dataset starts with 30 manually curated Bitcoin price prediction articles. Each article includes:

- full article text
- title
- URL
- news source
- date with month-level granularity
- human-annotated direction and confidence, mapped together into one `gold_score`
- a short sentence describing the prediction reasoning

These 30 records are used for GEPA-based prompt optimization. The optimized prompt is then evaluated on a separate unseen test set of 35-40 articles with the same annotation format.

The current GEPA runner uses Gemini through LiteLLM. The default task and reflection model are both:

- `gemini/gemini-2.5-flash-lite`

The broader project can still compare other models later, such as Qwen, Kimi, and GPT-OSS 120B, through provider-specific adapters or GEPA model arguments.

## Pipeline

### Stage 1: GEPA Training

Use the 30 curated training articles to optimize prompts for structured information extraction.

Expected inputs:

- `data/train/articles.jsonl`
- `data/train/articles_with_text.jsonl` for GEPA runs that include scraped article body text
- prompt templates under `src/prompts/` or future prompt asset files

Expected outputs:

- prompt candidates
- prompt run metadata
- per-generation metrics
- selected best prompt per model, and optionally one overall best prompt

Output location:

- `outputs/gepa_runs/`

### Stage 2: Test Evaluation

Apply the best prompt from Stage 1 to the unseen test dataset of 35-40 articles.

Expected inputs:

- `data/test/articles.jsonl`
- best prompt artifacts from `outputs/gepa_runs/`

Expected outputs:

- model predictions
- parse status
- direction accuracy
- macro F1
- confidence agreement or correlation
- parse failure rate
- model comparison tables

Output location:

- `outputs/evaluations/`

### Stage 3: Optional Real-World Validation

Select articles from specific news sources across different time periods, extract predicted viewpoints with the best prompt, and compare those predictions with actual Bitcoin price movements.

This stage is optional because the core project question is prompt extraction quality, not production price prediction.

Expected outputs:

- extracted source-level predictions
- realized BTC movement labels
- optional confidence-weighted accuracy scores
- source comparison summaries

Output location:

- `outputs/validation/`

### Stage 4: Final Analysis

Summarize whether optimized prompts improved extraction performance and compare model behavior across Qwen, Kimi, and GPT-OSS 120B.

Expected outputs:

- final metric tables
- prompt comparison summaries
- model comparison summaries
- discussion of failure modes
- reproducible report artifacts

Output location:

- `outputs/reports/`

## Data Format

JSONL is the canonical dataset format. Each line in `data/train/articles.jsonl` and `data/test/articles.jsonl` should follow this schema:

```json
{
  "article_id": "article-001",
  "text": "",
  "title": "Bitcoin price prediction article title",
  "url": "https://example.com/article",
  "source": "Example News",
  "date": "2024-03",
  "gold_score": 15,
  "gold_reasoning": "The article argues that institutional demand will push BTC higher."
}
```

The `text` field is reserved for full article text and may be blank when only the worksheet title is available.

Human annotations include both direction and confidence. The JSONL stores these together as one `gold_score`:

- direction `-1` with confidence `1-5` maps to final scores `1-5`
- direction `0` with confidence `1-5` maps to final scores `6-10`, so confidence `1` gives score `6`
- direction `1` with confidence `1-5` maps to final scores `11-15`

## Repository Structure

```text
Market_Prediction_DL_Final_Project/
├── src/
│   ├── common/          # Logging, JSONL helpers, shared utility code
│   ├── config/          # Settings and environment parsing
│   ├── data/            # Dataset schemas, validation, loading
│   ├── prompts/         # Prompt templates and rendering helpers
│   ├── models/          # Provider-agnostic LLM adapter interfaces
│   ├── gepa/            # Prompt optimization loop skeleton
│   ├── extraction/      # Structured response parsing and normalization
│   ├── evaluation/      # Metrics and comparison logic
│   ├── validation/      # Optional real-world BTC movement validation
│   └── analysis/        # Final report generation helpers
├── scripts/
│   ├── run_gepa.py
│   ├── run_test_evaluation.py
│   ├── run_real_world_validation.py
│   ├── run_final_analysis.py
│   └── setup/bootstrap.py
├── data/
│   ├── train/
│   ├── test/
│   ├── validation/
│   └── external/
├── outputs/
│   ├── gepa_runs/
│   ├── evaluations/
│   ├── validation/
│   └── reports/
├── docker/research/Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── AGENTS.md
```

## Environment Setup

The project uses Python `>=3.10.19`, `uv`, and Docker.

Create `.env` from the example file:

```bash
cp .env.example .env
```

Open `.env` and set:

```bash
GEMINI_API_KEY=your_real_gemini_api_key_here
```

Create or refresh the local scaffold:

```bash
uv run python scripts/setup/bootstrap.py
```

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run python -m pytest
uv run python -m ruff check .
```

Build the Docker research container:

```bash
docker compose build research
```

## Common Commands

Convert the human gold-standard worksheet to JSONL:

```bash
uv run python scripts/convert_gold_standard_xlsx.py
```

Scrape article body text into a separate JSONL so the original labels file stays unchanged:

```bash
uv run python scripts/scrape_article_text.py
```

Run a local GEPA dry run with the enriched JSONL:

```bash
uv run python scripts/run_gepa.py --dry-run data/train/articles_with_text.jsonl
```

Run a Docker GEPA dry run:

```bash
docker compose run --rm research uv run python scripts/run_gepa.py \
  --dry-run \
  data/train/articles_with_text.jsonl
```

Run a Docker GEPA smoke test with budget `1`:

```bash
docker compose run --rm research uv run python scripts/run_gepa.py \
  --budget 1 \
  --output outputs/gepa_runs/bitcoin_sentiment/smoke_result_001.json \
  --run-dir outputs/gepa_runs/bitcoin_sentiment/smoke_run_001 \
  data/train/articles_with_text.jsonl
```

Run a larger GEPA optimization after the smoke test succeeds:

```bash
docker compose run --rm research uv run python scripts/run_gepa.py \
  --budget 20 \
  --output outputs/gepa_runs/bitcoin_sentiment/result_budget20.json \
  --run-dir outputs/gepa_runs/bitcoin_sentiment/run_budget20 \
  data/train/articles_with_text.jsonl
```

Run test evaluation:

```bash
uv run python scripts/run_test_evaluation.py
```

Run optional real-world validation:

```bash
uv run python scripts/run_real_world_validation.py
```

Build final analysis artifacts:

```bash
uv run python scripts/run_final_analysis.py
```

## Notes

- This repository no longer follows the old trust-weighted trading pipeline.
- The project does not require FastAPI, Streamlit, TFT modeling, or author trust modeling.
- The first implementation target is a reproducible research workflow with clean schemas, prompt versioning, model adapters, and metrics.
- Keep `.env` private. Commit `.env.example`, not `.env`.

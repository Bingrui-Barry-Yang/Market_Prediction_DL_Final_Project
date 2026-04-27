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

The broader project can still compare other models later through GEPA model arguments and saved run artifacts.

## Pipeline

### GEPA Training

Use the 30 curated training articles to optimize prompts for structured information extraction.

Inputs:

- `data/train/articles.jsonl`

Outputs:

- prompt candidates
- prompt run metadata
- per-generation metrics
- selected best prompt per model

Output location:

- `outputs/gepa_runs/`

### Report Generation

Generate analysis figures and PDF/TeX summaries from completed GEPA run directories.

Inputs:

- completed run directories under `outputs/gepa_runs/bitcoin_sentiment/`

Outputs:

- PDF reports
- figure PDFs
- LaTeX source
- summary metric JSON

Output location:

- `reports/`

## Data Format

JSONL is the canonical dataset format. Each line in `data/train/articles.jsonl` follows this schema:

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

Human annotations include both direction and confidence. The JSONL stores these together as one `gold_score`:

- direction `-1` with confidence `1-5` maps to final scores `1-5`
- direction `0` with confidence `1-5` maps to final scores `6-10`, so confidence `1` gives score `6`
- direction `1` with confidence `1-5` maps to final scores `11-15`

## Repository Structure

```text
Market_Prediction_DL_Final_Project/
├── src/
│   ├── common/          # JSONL helpers and shared utilities
│   └── data/            # Canonical dataset schemas
├── scripts/
│   ├── convert_gold_standard_xlsx.py
│   ├── run_gepa.py
│   ├── run_gepa_rate_limited.py
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
├── reports/
│   ├── generate_report.py
│   └── run_*/
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

Run a local GEPA dry run with the canonical training JSONL:

```bash
uv run python scripts/run_gepa.py --dry-run data/train/articles.jsonl
```

Run a Docker GEPA dry run:

```bash
docker compose run --rm research uv run python scripts/run_gepa.py \
  --dry-run \
  data/train/articles.jsonl
```

Run a Docker GEPA smoke test with budget `1`:

```bash
docker compose run --rm research uv run python scripts/run_gepa.py \
  --budget 1 \
  --output outputs/gepa_runs/bitcoin_sentiment/smoke_result_001.json \
  --run-dir outputs/gepa_runs/bitcoin_sentiment/smoke_run_001 \
  data/train/articles.jsonl
```

Run a larger GEPA optimization after the smoke test succeeds:

```bash
docker compose run --rm research uv run python scripts/run_gepa.py \
  --budget 20 \
  --output outputs/gepa_runs/bitcoin_sentiment/result_budget20.json \
  --run-dir outputs/gepa_runs/bitcoin_sentiment/run_budget20 \
  data/train/articles.jsonl
```

Run the rate-limit-resilient GEPA variant:

```bash
uv run python scripts/run_gepa_rate_limited.py data/train/articles.jsonl
```

Generate a report from a completed GEPA run:

```bash
uv run python reports/generate_report.py \
  --run-dir outputs/gepa_runs/bitcoin_sentiment/run_gptoss120b_b150
```

## Notes

- This repository no longer follows the old trust-weighted trading pipeline.
- The project does not require FastAPI, Streamlit, TFT modeling, or author trust modeling.
- The current implementation target is a reproducible GEPA research workflow with a compact dataset converter, training runners, and run reports.
- Keep `.env` private. Commit `.env.example`, not `.env`.

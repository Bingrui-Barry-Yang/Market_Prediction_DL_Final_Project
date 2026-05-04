# Bitcoin News Prompt Optimization Study

## Project Summary

This repository studies whether GEPA prompt optimization helps large language models extract structured Bitcoin price-prediction signals from existing news articles. Instead of forecasting Bitcoin prices directly, the pipeline asks an LLM to read a curated article and output one ordinal `score` from 1 to 15, where low scores are bearish, middle scores are neutral, and high scores are bullish. GEPA was run on 30 manually labeled training articles to evolve extraction prompts, then 38 seed/evolved prompt candidates were evaluated across four task models on a 35-article held-out test set. The key result is mixed: GEPA produced longer, more domain-specific prompts, but GEPA-selected best prompts did not consistently beat the shared seed prompt on test QWK; Qwen 3.6 achieved the strongest combined direction/confidence QWK, while Claude Sonnet 4.6 achieved the strongest direction agreement.

## Codebase Map

```text
.
├── src/
│   ├── common/                  # JSONL reading/writing helpers
│   ├── data/                    # Pydantic schemas for canonical article records
│   └── evaluation/              # Prompt registry, parsing, scoring, aggregate metrics
├── scripts/
│   ├── convert_gold_standard_xlsx.py   # Convert annotated spreadsheets to JSONL
│   ├── run_gepa.py                    # Stage 1 GEPA prompt optimization runner
│   ├── run_test_eval.py               # Stage 2 full prompt x model test evaluation
│   ├── run_gepa_reports.py            # Build GEPA run figures and PDF reports
│   ├── build_qwk_inputs.py            # Convert test predictions into QWK inputs
│   ├── run_qwk_best_vs_seed.py        # Compare GEPA-best prompts with seed prompts
│   ├── run_qwk_per_model.py           # Aggregate QWK by task model
│   └── evaluate_author.py             # Optional author-level validation analysis
├── data/
│   ├── train/articles.jsonl           # 30 canonical GEPA training articles
│   ├── test/articles_test.jsonl       # 35 held-out test articles
│   ├── qwk/                           # Derived QWK input tables
│   └── authordemo/                    # Optional author-validation sample
├── outputs/
│   ├── gepa_runs/                     # GEPA candidates, run logs, reports, figures
│   ├── test_eval/                     # Per-prediction rows, matrices, metrics
│   ├── qwk/                           # QWK summary CSV/JSON outputs
│   └── test_author/                   # Optional author-validation outputs
├── BTC_Price_Sentiment_Prediction/    # Final ACM-style paper source and figures
├── tests/unit/                        # Fast unit tests with mocked/no live model calls
├── docker/research/Dockerfile         # Reproducible research image
├── docker-compose.yml                 # One-service research container
├── Makefile                           # Convenience commands
├── pyproject.toml                     # Python package and dependency metadata
└── README.md
```

## Architecture

The project is a research workflow, not a production trading system. No neural network is fine-tuned in this repository; the "model architecture" is the prompt-optimization and evaluation loop around external LLMs.

```mermaid
flowchart LR
    A["Curated Bitcoin news articles<br/>train: 30 JSONL<br/>test: 35 JSONL"] --> B["ArticleRecord schema validation"]
    B --> C["Stage 1: GEPA prompt optimization<br/>scripts/run_gepa.py"]
    C --> D["Prompt candidates<br/>outputs/gepa_runs/bitcoin_sentiment"]
    D --> E["Stage 2: full test evaluation<br/>scripts/run_test_eval.py"]
    A --> E
    E --> F["Structured LLM output<br/>{score: 1..15}"]
    F --> G["Scoring<br/>partial credit, MAE, RMSE,<br/>direction accuracy, parse rate"]
    G --> H["QWK analysis<br/>best vs seed and per model"]
    H --> I["Final report and figures<br/>BTC_Price_Sentiment_Prediction/"]
```

```mermaid
flowchart TB
    S["LLM prompt"] --> R["Read article title + full text"]
    R --> J["Return JSON only:<br/>{\"score\": integer}"]
    J --> K{"Score band"}
    K --> B["1-5 bearish"]
    K --> N["6-10 neutral"]
    K --> U["11-15 bullish"]
    B --> M["Confidence = position inside band"]
    N --> M
    U --> M
```

## Data Contract

Canonical article datasets are JSONL files. Each line follows:

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

Human direction and confidence annotations are stored as one `gold_score`:

| Direction | Confidence | Score range |
|---|---:|---:|
| bearish | 1-5 | 1-5 |
| neutral | 1-5 | 6-10 |
| bullish | 1-5 | 11-15 |

The converted JSONL intentionally does not emit separate `gold_direction` or `gold_confidence` fields.

## Setup And Run

Requirements:

- Python `>=3.10.19`
- `uv`
- Docker and Docker Compose for reproducible container runs
- Optional API/model services for live runs:
  - `GEMINI_API_KEY` for Gemini GEPA runs
  - `ANTHROPIC_API_KEY` for Claude task-model evaluation
  - local Ollama server for the configured Ollama models

Install locally:

```bash
cp .env.example .env
# edit .env with any live model credentials needed for your run
uv sync
```

Validate the repository without live model calls:

```bash
uv run python -m pytest
uv run python -m ruff check .
uv run python scripts/run_gepa.py --dry-run data/train/articles.jsonl
uv run python scripts/run_test_eval.py --dry-run --expected-n 0
```

Build the Docker research image:

```bash
docker compose build research
```

## Reproduce Training And Evaluation

Convert source spreadsheets to canonical JSONL:

```bash
uv run python scripts/convert_gold_standard_xlsx.py
```

Run a GEPA smoke test:

```bash
docker compose run --rm research uv run python scripts/run_gepa.py \
  --budget 1 \
  --output outputs/gepa_runs/bitcoin_sentiment/smoke_result_001.json \
  --run-dir outputs/gepa_runs/bitcoin_sentiment/smoke_run_001 \
  data/train/articles.jsonl
```

Run a full GEPA optimization. The completed study used a budget of 150 for each source model; model IDs can be changed with `--task-lm` and `--reflection-lm`.

```bash
docker compose run --rm research uv run python scripts/run_gepa.py \
  --budget 150 \
  --task-lm gemini/gemini-2.5-flash-lite \
  --reflection-lm gemini/gemini-2.5-flash-lite \
  --output outputs/gepa_runs/bitcoin_sentiment/result_budget150.json \
  --run-dir outputs/gepa_runs/bitcoin_sentiment/run_budget150 \
  data/train/articles.jsonl
```

Run the Stage 2 test evaluation over the registered task models and prompt candidates:

```bash
uv run python scripts/run_test_eval.py --resume
```

For local Ollama models from Docker, make sure Ollama is reachable from the container:

```bash
docker run --rm --network host \
  --env-file .env \
  -e OLLAMA_API_BASE=http://localhost:11434 \
  -v "$(pwd)":/app -w /app \
  market_prediction_dl_final_project-research:latest \
  uv run python scripts/run_test_eval.py --resume
```

Regenerate analysis tables:

```bash
uv run python scripts/build_qwk_inputs.py
uv run python scripts/run_qwk_best_vs_seed.py
uv run python scripts/run_qwk_per_model.py
```

Regenerate a GEPA run report:

```bash
uv run python scripts/run_gepa_reports.py \
  --run-dir outputs/gepa_runs/bitcoin_sentiment/run_gptoss120b_b150
```

Run the optional author-level validation. This evaluates 15 FXStreet Bitcoin forecast articles from one author against the following 30 days of BTC-USD hourly prices. Articles without `gold_score` are first scored with a GEPA prompt and the selected scoring model.

```bash
uv run python scripts/evaluate_author.py \
  --articles data/authordemo/author_test.jsonl \
  --output outputs/test_author/author_evaluation.csv \
  --score-prompt qwen36 \
  --score-model ollama_chat/qwen3.6:latest
```

## Results Summary

Main report artifacts are in `BTC_Price_Sentiment_Prediction/`, with the compiled paper also available as `Optimizing_Prompts_for_Extracting_Bitcoin_Price_Predictions_from_News_Articles.pdf`.

Stage 1 produced 38 prompt candidates across four source-model GEPA runs:

| Source model | Candidates | GEPA-selected best prompt words | Seed expansion |
|---|---:|---:|---:|
| Claude Sonnet 4.6 | 11 | 615 | 6.68x |
| GPT-OSS 120B | 7 | 92 | 1.00x |
| Qwen 3.6 | 11 | 714 | 7.76x |
| Gemma 4 E2B | 9 | 259 | 2.82x |

Overall QWK by task model, computed from `outputs/qwk/per_model/summary.csv`:

| Task model | Parsed rows | Parse rate | Direction QWK | Confidence QWK | QWK sum | MAE | Exact match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 1081 / 1330 | 81.3% | **0.873** | 0.137 | 1.010 | **1.713** | 16.0% |
| GPT-OSS 120B | 1325 / 1330 | 99.6% | 0.792 | 0.270 | 1.061 | 1.922 | 22.3% |
| Qwen 3.6 | 662 / 805 | 82.2% | 0.794 | **0.385** | **1.178** | 1.923 | **24.5%** |
| Gemma 4 E2B | 1330 / 1330 | **100.0%** | 0.636 | 0.338 | 0.974 | 2.422 | 23.6% |

Seed-vs-GEPA-best comparison from `outputs/qwk/best_vs_seed/summary.csv`: among 14 task/source pairs with both seed and best scores available, the combined direction-plus-confidence QWK improved in 2 cases, was unchanged in 4 cases, and decreased in 8 cases. This supports the paper's main conclusion: GEPA often made prompts more detailed and sometimes improved broad direction labeling, but it did not reliably improve calibrated 1-15 scoring on the held-out test set.

Optional author-level validation results are stored in `outputs/test_author/`. This stage used the Qwen 3.6 GEPA prompt to score 15 FXStreet "Bitcoin Price Forecast" articles by Manish Chhetri, then compared each extracted direction against Coinbase BTC-USD movement over the next 30 days.

| Author-eval metric | Value | Interpretation |
|---|---:|---|
| Simple trust score | -0.080 | Unreliable / noisy |
| Confidence-weighted trust score | -0.123 | Unreliable / noisy |
| Very correct articles | 5 / 15 | Strong positive 30-day alignment |
| Correct articles | 1 / 15 | Positive 30-day alignment |
| Noisy articles | 2 / 15 | Inside the ambiguous band |
| Wrong articles | 3 / 15 | Negative 30-day alignment |
| Very wrong articles | 4 / 15 | Strong negative 30-day alignment |

By predicted direction, bearish calls were the strongest slice: 5 bearish calls averaged `+0.366`, while 4 neutral calls averaged `-0.343` and 6 bullish calls averaged `-0.275`. The aggregate author-eval conclusion is therefore noisy rather than trustworthy or consistently contrarian. See `outputs/test_author/author_evaluation_report.md` and `outputs/test_author/author_evaluation_summary.csv` for the per-article table.

Useful figures:

- `BTC_Price_Sentiment_Prediction/claude_prompt_evolution_tree.png`
- `BTC_Price_Sentiment_Prediction/keyword_heatmap_claude_sonnet46.pdf`
- `BTC_Price_Sentiment_Prediction/keyword_heatmap_gptoss120b.pdf`
- `BTC_Price_Sentiment_Prediction/keyword_heatmap_qwen36.pdf`
- `BTC_Price_Sentiment_Prediction/keyword_heatmap_gemma4e2b.pdf`

## AI Usage Log

LLM tools were used as part of the project workflow for prompt optimization, structured extraction, and documentation support:

| Use | Details |
|---|---|
| GEPA prompt optimization | GEPA used LLM task/reflection calls to generate and select prompt candidates from the 30-article training set. |
| Test-set extraction | Claude Sonnet 4.6, GPT-OSS 120B, Qwen 3.6, and Gemma 4 E2B were used as task models to emit structured `{ "score": ... }` predictions. |
| Analysis assistance | AI tools helped inspect outputs, summarize tables, and draft report/README language. Final metrics come from repository artifacts, not from manual estimation. |
| Safeguards | Unit tests and dry-run commands avoid requiring live model credentials; generated outputs are stored under `outputs/` for reproducibility and auditability. |

## Notes

- This repository intentionally does not implement the old trust-weighted trading pipeline.
- It does not include FastAPI serving, Streamlit dashboards, Temporal Fusion Transformers, or production trading workflows.
- Keep generated artifacts under `outputs/`, canonical datasets under `data/`, executable stage runners under `scripts/`, and business logic under `src/`.

# AGENTS.md

## Project Goal

Build a Bitcoin price trend prediction system based on semantic analysis of Bitcoin-related news articles. The core idea is that different news authors and media sources have varying levels of accuracy when predicting Bitcoin prices. The system should learn a rolling **trust value** for each author or source, then use **trust-weighted sentiment signals** to predict future Bitcoin price direction.

The entire project uses:

- `Python >= 3.10`
- `uv` for dependency and environment management
- `Docker` and `docker compose` for service orchestration
- `FastAPI` for inference serving
- `Streamlit` for visualization
- `MLflow` for experiment tracking

The storage flow is:

`raw JSONL -> filtered JSONL -> scored JSONL -> feature Parquet -> model artifacts`

---

## Architecture Summary

The repository should be implemented as a single Python monorepo with:

- shared business logic in `src/`
- container-facing application entrypoints in `apps/`
- reproducible operational runners in `scripts/`
- layered persisted datasets in `data/` with generated runtime byproducts consolidated under `outputs/`

The runtime is split into four containers:

1. `scraper-filter`
2. `llm-scoring`
3. `training`
4. `serving`

This separation keeps scraping, expensive LLM operations, heavy model training, and online serving isolated while still sharing one codebase.

---

## Full Commented Directory Structure

```text
Market_Prediction_DL_Final_Project/                      # Project root; shared Python monorepo for all pipeline stages and services
├── apps/                                               # Runtime application entrypoints grouped by container/service role
│   ├── scraper_filter/                                 # Entrypoints for scraping historical news and running prediction-content filtering
│   ├── llm_scoring/                                    # Entrypoints for GEPA prompt evolution and Gemini batch sentiment scoring
│   ├── training/                                       # Entrypoints for author statistics, trust model training, TFT training, and evaluation
│   ├── inference_api/                                  # FastAPI application for prediction serving and metadata endpoints
│   └── dashboard/                                      # Streamlit application for visual analytics and model monitoring
├── src/                                                # Shared reusable Python package with all core business logic
│   ├── common/                                         # Shared utilities: logging, retries, IDs, time/date helpers, file IO, exceptions
│   ├── config/                                         # Typed settings, environment parsing, constants, stage-specific configuration
│   ├── ingestion/                                      # News API clients, async scraping, normalization, deduplication logic
│   ├── filtering/                                      # Binary LLM filter logic for forward-looking Bitcoin prediction detection
│   ├── scoring/                                        # Gemini prompting, structured output parsing, scoring orchestration, GEPA helpers
│   ├── pricing/                                        # Bitcoin historical price download, validation, cleaning, resampling
│   ├── labeling/                                       # Horizon alignment and future price-movement label generation
│   ├── features/                                       # Feature engineering and parquet dataset creation for downstream models
│   ├── trust_model/                                    # Author/source statistics, rolling windows, LightGBM and LSTM trust modeling
│   ├── forecasting/                                    # Daily aggregation, TFT dataset building, training, and prediction logic
│   ├── evaluation/                                     # Metrics, backtesting, walk-forward validation, Sharpe calculation, reports
│   ├── orchestration/                                  # Stage runners, pipeline sequencing, job manifests, resumability/checkpoint logic
│   └── serving/                                        # Shared inference-time loading, prediction services, and response schemas
├── scripts/                                            # Thin operational runners that call into src/ for reproducible stage execution
│   ├── setup/                                          # Environment bootstrap, sanity checks, local initialization tasks
│   ├── stage1_ingest/                                  # Commands for 2017-2021 and 2022-2026 article scraping backfills
│   ├── stage2_filter/                                  # Commands for binary prediction-content filtering batches
│   ├── stage3_score/                                   # Commands for anchor evaluation, GEPA loops, and full Gemini scoring runs
│   ├── stage4_prices/                                  # Commands for BTC historical price fetch and article-price alignment
│   ├── stage5_author_stats/                            # Commands for author/source directional success statistics generation
│   ├── stage6_trust_train/                             # Commands for rolling trust feature generation and trust model training
│   ├── stage7_forecast_train/                          # Commands for TFT dataset generation, training, and walk-forward evaluation
│   ├── backfill/                                       # Full historical rebuild workflows and one-off reprocessing jobs
│   ├── validation/                                     # Data quality, schema, completeness, and drift validation commands
│   └── deploy/                                         # Local release, model promotion, and container startup helper commands
├── data/                                               # Layered data lake for raw, intermediate, and model-ready persisted datasets
│   ├── raw/                                            # Raw JSONL articles and raw BTC market data exactly after collection/normalization
│   ├── filtered/                                       # Stage 2 output: only articles judged to contain future BTC price predictions
│   ├── scored/                                         # Stage 3 output: LLM-enriched JSONL with sentiment, confidence, and horizon
│   ├── features/                                       # Parquet datasets for labels, author stats, trust features, and forecast inputs
│   ├── models/                                         # Exported trained models, metadata bundles, and production-ready inference assets
│   ├── external/                                       # Manually supplied assets such as anchor labels, vendor exports, or reference data
│   └── interim/                                        # Temporary checkpoints, partial batch outputs, and recoverable join artifacts
├── prompts/                                            # Versioned prompt assets used by filtering, scoring, and prompt optimization
│   ├── filter/                                         # Binary forward-looking prediction detection prompts and evaluation variants
│   ├── scoring/                                        # Gemini scoring prompts for sentiment certainty and horizon extraction
│   ├── gepa/                                           # Reflection, mutation, and candidate-generation prompts for GEPA
│   └── evaluation/                                     # Prompt QA templates and spot-check prompts for quality review
├── experiments/                                        # Research workspace for human annotations, GEPA outputs, and exploratory analysis
│   ├── anchors/                                        # Human-labeled anchor article sets used to optimize and evaluate prompts
│   ├── gepa_runs/                                      # Per-generation prompt candidates, fitness metrics, and selected winners
│   ├── notebooks/                                      # Optional exploratory notebooks for analysis that should not hold production logic
│   └── reports/                                        # Charts, experiment summaries, benchmark tables, and evaluation exports
├── tests/                                              # Automated validation across units, integrations, and end-to-end workflows
│   ├── unit/                                           # Fast tests for isolated logic, parsers, metrics, and utility functions
│   ├── integration/                                    # Multi-module tests for APIs, pipeline stages, storage boundaries, and contracts
│   ├── e2e/                                            # End-to-end smoke tests over small fixture datasets covering full pipeline paths
│   ├── fixtures/                                       # Sample article, price, and mock LLM response inputs used by tests
│   └── golden/                                         # Golden expected outputs for prompt parsing and structured response regression checks
├── docker/                                             # Docker build context and image definitions for all services
│   ├── base/                                           # Shared base image setup with Python, uv, common OS packages, and base env
│   ├── scraper_filter/                                 # Docker build assets for scraping/filtering container
│   ├── llm_scoring/                                    # Docker build assets for Gemini scoring and GEPA container
│   ├── training/                                       # Docker build assets for feature generation, training, and MLflow workloads
│   ├── serving/                                        # Docker build assets for FastAPI and Streamlit serving workloads
│   └── mlflow/                                         # Optional dedicated MLflow image assets if tracking server is separated later
├── infra/                                              # Non-code operational configuration for local orchestration and deployment hygiene
│   ├── compose/                                        # docker compose definitions for local development and reproducible service startup
│   ├── env/                                            # Environment variable templates and documented runtime configuration sets
│   ├── logging/                                        # Centralized logging configuration for all containers and local jobs
│   └── healthchecks/                                   # Service health check definitions and helper scripts
├── outputs/                                            # Consolidated generated outputs so runtime byproducts stay in one clean top-level area
│   ├── artifacts/                                      # Exported reports, plots, packaged outputs, and promoted evaluation artifacts
│   ├── logs/                                           # Runtime logs for batch jobs, services, and debugging
│   ├── mlruns/                                         # Local MLflow tracking store for experiments, metrics, and artifacts
│   └── monitoring/                                     # Prediction logs, drift snapshots, inference traces, and dashboard-ready monitoring data
└── docs/                                               # Human-readable technical documentation for architecture and operations
    ├── architecture/                                   # System diagrams, service boundaries, and data-flow documentation
    ├── pipeline/                                       # Stage-by-stage operational docs and run order documentation
    ├── modeling/                                       # Trust modeling, TFT design, metrics, and experiment rationale
    └── operations/                                     # Deployment, troubleshooting, and maintenance runbooks
```

---

## Four-Container Docker Design

### Docker bootstrap instructions

Install Docker Desktop on macOS:

```bash
brew install --cask docker
```

Start Docker Desktop:

```bash
open -a Docker
```

Verify installation and daemon health:

```bash
docker --version
docker compose version
docker info
```

Success criteria:

- Docker CLI is installed
- Docker Compose is installed
- `docker info` returns server details

Important usage note:

- `docker compose up --build` is not needed to verify installation
- use `docker compose up --build` only when you want to build and run the project services
- in this repo, that step is most useful after `.env` is populated and service logic exists beyond scaffolding

### Base image

- base on `python:3.10-slim` or newer
- install `uv`
- use `uv sync --frozen` to install dependencies
- keep one shared dependency definition and split optional groups if needed for `training`, `serving`, `llm`, and `dev`

### Container 1: `scraper-filter`

Responsibilities:

- Stage 1 historical scraping
- Stage 2 prediction-content filtering

Mounts:

- `data/`
- `outputs/logs/`
- `prompts/`

Operational concerns:

- asynchronous throughput
- API rate limits
- checkpointing and resumability
- article deduplication

### Container 2: `llm-scoring`

Responsibilities:

- GEPA prompt optimization
- Gemini batch scoring

Mounts:

- `data/`
- `prompts/`
- `experiments/`
- `outputs/artifacts/`

Operational concerns:

- prompt versioning
- structured output validation
- dead-letter handling for malformed responses
- cost and retry controls

### Container 3: `training`

Responsibilities:

- Stage 4 through Stage 7 processing
- trust model training
- TFT training
- walk-forward evaluation
- MLflow logging

Mounts:

- `data/`
- `outputs/mlruns/`
- `outputs/artifacts/`

Operational concerns:

- parquet feature generation
- leakage-safe rolling windows
- experiment reproducibility

### Container 4: `serving`

Responsibilities:

- FastAPI inference service
- Streamlit dashboard

Mounts:

- `data/models/`
- `outputs/monitoring/`
- `outputs/artifacts/`

Operational concerns:

- loading promoted model artifacts
- stable response contracts
- lightweight online inference

Recommended environment bootstrap order:

1. Install and start Docker Desktop.
2. Verify `docker --version`, `docker compose version`, and `docker info`.
3. Create `.env` from `.env.example`.
4. Run `uv sync`.
5. Fill in API keys.
6. Run `docker compose up --build` when container execution is needed.

---

## Public Interfaces and Data Contracts

These schemas should be fixed early because they define the pipeline boundaries.

### Stage 1 normalized article record

- `article_id`
- `source`
- `author`
- `published_at`
- `title`
- `body`
- `url`

### Stage 2 filter output additions

- `contains_prediction`
- `filter_confidence`
- `filter_model_version`

### Stage 3 scoring output additions

- `score` as an ordinal sentiment-certainty score, initially designed as integer `1-10`
- `direction` as `bull | bear | neutral`
- `confidence` as float `0-1`
- `prediction_horizon` as `1-7d | 7-30d | 30d+ | unspecified`
- `scoring_model_version`
- `prompt_version`

Design note:

- a raw `1-10` scale may be too granular for stable LLM behavior
- the implementation should preserve the raw score for analysis, but also support a derived bucketed feature for modeling, such as 3-level or 5-level certainty bands

### Stage 4 article label output additions

- `target_date`
- `horizon_days`
- `price_t`
- `price_t_plus_h`
- `realized_return`
- `realized_direction`

### Stage 5 author/source statistics table

One row per author or fallback source per evaluation window. Include:

- article count
- directional accuracy
- Bayesian-smoothed or Wilson-adjusted accuracy
- recency-weighted accuracy
- horizon mix
- average model confidence
- evaluation-window timestamps

Sparse-entity safeguards:

- apply Bayesian smoothing so authors with one or two articles do not get extreme trust estimates
- enforce a minimum article threshold before treating author-level metrics as reliable
- fall back to source-level metrics when author history is missing or too sparse
- fall back to global default trust when both author and source evidence are insufficient

### Stage 6 trust value output

- `entity_id`
- `entity_type` as `author | source`
- `as_of_date`
- `trust_value` in `0-1`
- `trust_model_version`

Trust value construction note:

- trust should be derived from multiple historical windows, not only a single fixed window
- recommended windows are `3` month, `6` month, and `12` month views with stronger weighting on recent performance

### Stage 7 daily forecast feature row

- `date`
- trust-weighted sentiment aggregates
- trust-weighted confidence aggregates
- article-volume and source-diversity features
- BTC historical price features
- labels for next `7d`, `14d`, and `30d` direction

### Serving API

FastAPI should expose:

- trust lookup by author or source
- latest daily aggregate feature summary
- forecast outputs for `7`, `14`, and `30` day horizons
- model metadata and version endpoints

---

## Detailed Stage-by-Stage Implementation Plan

### Stage 1: Scrape Historical News (2017-2021)

Use Google News API and NewsAPI to scrape Bitcoin-related articles.

Implementation details:

- use `asyncio` and `httpx`
- validate records with `Pydantic`
- query by time windows to reduce rate-limit pressure
- normalize each article into the canonical schema
- remove empty-body records
- filter non-English records
- deduplicate near-duplicates with `SimHash`
- persist normalized JSONL into `data/raw/`
- keep manifests or checkpoints so reruns can resume cleanly

### Stage 2: Prediction Content Filter

Run a lower-cost LLM to decide whether an article contains a forward-looking Bitcoin price judgment.

Keep:

- explicit price predictions
- implied bullish or bearish outlooks

Discard:

- pure daily market recaps
- historical summaries
- technical explainers
- policy/regulation reports without a price claim

Implementation details:

- batch requests
- validate structured responses
- track keep-rate metrics
- write passing records to `data/filtered/`
- retain sample QA outputs for false-positive/false-negative inspection

### Stage 3: LLM Sentiment Scoring + Prediction Horizon Extraction

For every filtered article, call Gemini and produce:

- `score`
- `direction`
- `confidence`
- `prediction_horizon`

Important interpretation:

- score measures the certainty and strength of the author's view
- score does not measure correctness

Scoring caution:

- a `1-10` score is useful as a raw research signal, but may be too detailed for stable downstream modeling
- the implementation should evaluate whether bucketed certainty classes outperform the raw scale in robustness and reproducibility

#### GEPA optimization loop

Before full scoring:

1. label `50-100` anchor articles manually
2. run current prompt on anchors
3. compute Spearman correlation
4. ask for prompt reflection on ambiguity around certainty vs hedging
5. generate `3-5` candidate prompts
6. re-evaluate
7. keep the best prompt
8. repeat for `8-15` generations until stable

Implementation details:

- store prompt versions and metrics in `experiments/gepa_runs/`
- log results to `MLflow`
- freeze the best prompt for production scoring
- save scored records into `data/scored/`
- preserve malformed outputs in a retry or dead-letter path

### Stage 4: Fetch Bitcoin Historical Prices and Align by Prediction Horizon

Download daily BTC data for `2017-2021` and map article predictions to future outcomes.

Horizon mapping:

- `1-7d -> T+7`
- `7-30d -> T+30`
- `30d+ -> T+90`
- `unspecified -> T+7`

Implementation details:

- normalize dates to a consistent standard
- compute realized return and realized direction
- join article predictions to future price windows
- store article-level labeled outputs in `data/features/`

### Stage 5: Calculate Per-Author / Per-Source Prediction Success Rate

Compare each article's predicted direction with actual BTC direction over its aligned horizon.

Compute:

- total prediction count
- directional accuracy
- horizon-specific accuracy
- average LLM confidence
- recency-weighted hit rate
- consistency over time

Recommended reliability logic:

- do not trust raw accuracy alone
- use Bayesian smoothing or Wilson lower bound
- require a minimum article threshold before using an author-specific estimate
- assign a default trust near `0.5` when evidence is too sparse
- aggregate at author level first
- fall back to source when author data is missing or sparse

Output:

- author/source statistics table in Parquet

### Stage 6: Train the Trust Value Model

Generate rolling author/source feature vectors and predict future trustworthiness.

Implementation details:

- do not rely on only one rolling window
- build trust features from short-term `3` month, mid-term `6` month, and long-term `12` month history
- weight recent performance more strongly than older performance
- refresh trust values quarterly
- build features only from information available up to each cutoff date
- start with `LightGBM` baseline
- then train a two-layer `LSTM` over time-window sequences
- track experiments with `MLflow`
- emit trust values in `0-1`
- assign new entities a default trust near `0.5`
- update that default incrementally as evidence accumulates

Reasoning:

- a single short window is often too unstable because many authors publish very few articles
- a single long window can become stale because market regimes shift
- combining multiple windows balances stability and responsiveness

### Stage 7: Train the 2022-2026 Price Trend Prediction Model

Repeat Stages 1-4 for `2022-2026`, then use trust-weighted signals to predict BTC direction.

Per-record inputs:

- date
- sentiment score or bucketed sentiment-certainty feature
- prediction horizon
- author trust value
- Bitcoin price on that date

Daily aggregation details:

- trust-weighted sentiment
- trust-weighted confidence
- article counts
- source diversity
- BTC market covariates

Model:

- `Temporal Fusion Transformer (TFT)`

Prediction targets:

- next `7` days
- next `14` days
- next `30` days

Evaluation:

- walk-forward validation
- retrain every `30` days
- directional accuracy
- Sharpe Ratio from a signal-based trading strategy

---

## Data Layer Design

Use the layered storage and output layout exactly as follows:

- `data/raw/`: raw article JSONL and raw BTC market pulls
- `data/filtered/`: forward-looking prediction-only article JSONL
- `data/scored/`: sentiment and horizon enriched article JSONL
- `data/features/`: Parquet datasets for labels, author stats, trust features, and TFT inputs
- `data/models/`: trained models, metadata, and inference-ready bundles
- `outputs/mlruns/`: MLflow experiment tracking data
- `outputs/artifacts/`: exported reports, plots, and promoted bundles
- `outputs/logs/`: runtime logs
- `outputs/monitoring/`: inference and drift monitoring outputs

This structure keeps the pipeline reproducible and easy to debug.

---

## Execution Order

1. Create folder structure and dependency groups.
2. Define schemas, settings, and path conventions.
3. Build Stage 1 scraping and normalization.
4. Build Stage 2 prediction filtering.
5. Build GEPA and finalize the Gemini prompt.
6. Run full Stage 3 scoring.
7. Build Stage 4 price alignment and label generation.
8. Build Stage 5 author/source statistics.
9. Train Stage 6 trust models.
10. Repeat Stages 1-4 for `2022-2026`.
11. Build Stage 7 daily aggregates and train the TFT model.
12. Add FastAPI inference service.
13. Add Streamlit dashboard.
14. Run end-to-end validation and promote final model artifacts.

---

## Testing Strategy

### Unit tests

- schema validation
- SimHash deduplication logic
- horizon mapping logic
- metric calculations
- trust-value postprocessing

### Integration tests

- Stage 1 through Stage 4 data flow with fixtures
- mocked news APIs
- mocked LLM filter and Gemini responses
- storage boundary checks between JSONL and Parquet stages

### Golden tests

- parsing of structured LLM outputs
- regression checks for prompt output shape and normalization

### End-to-end smoke test

Run a small fixture dataset through:

`raw -> filtered -> scored -> labeled -> trust -> forecast`

### Validation rules

- no duplicate `article_id`
- no empty article bodies after cleaning
- only allowed horizon labels
- no future leakage in rolling trust features
- strictly time-ordered walk-forward splits
- serving API returns stable response shapes

---

## Assumptions and Defaults

- Python version is `>=3.10`
- `uv` is the only dependency and environment manager
- `docker compose` is the local orchestration standard
- stage datasets are stored in layered folders rather than a database for v1
- JSONL is the canonical format for article-level stages
- Parquet is the canonical format for analytics and model-ready stages
- trust is modeled as a smoothed reliability estimate, not raw historical accuracy alone
- trust uses multiple historical windows, with recent evidence weighted more heavily than older evidence
- missing or sparse authors fall back to source-level aggregates, then to a global mean
- raw LLM sentiment scores may be bucketed into a coarser scale if the `1-10` signal is too noisy in practice
- `unspecified` prediction horizon defaults to `T+7`
- the repository's previous `README.md` direction is obsolete and replaced by this trust-weighted pipeline design

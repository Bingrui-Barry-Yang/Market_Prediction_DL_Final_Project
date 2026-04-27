UV ?= uv
RUN_DIR ?= outputs/gepa_runs/bitcoin_sentiment/run_gptoss120b_b150

.PHONY: bootstrap sync test lint convert gepa gepa-dry gepa-rate-limited report docker

bootstrap:
	$(UV) run python scripts/setup/bootstrap.py

sync:
	$(UV) sync

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .

convert:
	$(UV) run python scripts/convert_gold_standard_xlsx.py

gepa:
	$(UV) run python scripts/run_gepa.py

gepa-dry:
	$(UV) run python scripts/run_gepa.py --dry-run data/train/articles.jsonl

gepa-rate-limited:
	$(UV) run python scripts/run_gepa_rate_limited.py data/train/articles.jsonl

report:
	$(UV) run python reports/generate_report.py --run-dir $(RUN_DIR)

docker:
	docker compose run --rm research

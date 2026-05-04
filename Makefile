UV ?= uv
RUN_DIR ?= outputs/gepa_runs/bitcoin_sentiment/run_gptoss120b_b150

.PHONY: bootstrap sync test lint convert gepa gepa-dry report docker test-eval test-eval-dry

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

report:
	$(UV) run python scripts/run_gepa_reports.py --run-dir $(RUN_DIR)

docker:
	docker compose run --rm research

test-eval:
	$(UV) run python scripts/run_test_eval.py

test-eval-dry:
	$(UV) run python scripts/run_test_eval.py --dry-run --expected-n 0

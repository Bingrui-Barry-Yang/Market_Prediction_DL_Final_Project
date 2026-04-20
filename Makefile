UV ?= uv

.PHONY: bootstrap sync test lint gepa evaluate validate analyze docker

bootstrap:
	$(UV) run python scripts/setup/bootstrap.py

sync:
	$(UV) sync

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .

gepa:
	$(UV) run python scripts/run_gepa.py

evaluate:
	$(UV) run python scripts/run_test_evaluation.py

validate:
	$(UV) run python scripts/run_real_world_validation.py

analyze:
	$(UV) run python scripts/run_final_analysis.py

docker:
	docker compose run --rm research

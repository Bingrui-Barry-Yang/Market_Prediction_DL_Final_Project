UV ?= uv

.PHONY: bootstrap sync test api dashboard

bootstrap:
	$(UV) run python scripts/setup/bootstrap.py

sync:
	$(UV) sync

test:
	$(UV) run pytest

api:
	$(UV) run python -m apps.inference_api.main

dashboard:
	$(UV) run streamlit run apps/dashboard/app.py --server.port 8501 --server.address 0.0.0.0

FROM python:3.10.19-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md AGENTS.md /app/
COPY src /app/src
COPY scripts /app/scripts
RUN uv sync --frozen --no-dev

COPY . /app

CMD ["uv", "run", "python", "scripts/run_gepa.py", "--dry-run", "data/train/articles.jsonl"]

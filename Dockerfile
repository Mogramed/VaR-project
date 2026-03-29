FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY alembic.ini ./
COPY alembic ./alembic
COPY src ./src
COPY scripts ./scripts
COPY config ./config
COPY data/processed ./data/processed
COPY data/fixtures ./data/fixtures

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN mkdir -p /app/data/app /app/data/processed /app/data/snapshots /app/reports

EXPOSE 8000

CMD ["var-project", "api", "--host", "0.0.0.0", "--port", "8000"]

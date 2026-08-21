# Multi-stage: the build toolchain never reaches the published image.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt


FROM python:3.11-slim AS runtime

# libgomp1 is the OpenMP runtime the gradient-boosting wheels link against;
# curl backs the HEALTHCHECK below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY --chown=app:app pipeline_lib.py ./
COPY --chown=app:app app ./app

# Model binaries are fetched from the GitHub release before the build; they are
# not in git. See .github/workflows/docker-publish.yml.
COPY --chown=app:app models ./models

# Never run a network service as root.
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

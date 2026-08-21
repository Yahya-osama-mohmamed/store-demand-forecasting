# Multi-stage: the build toolchain never reaches the published image.
FROM python:3.11-slim AS builder

# uv resolves and installs from uv.lock, so the image gets the exact same
# transitive tree that CI tested - not whatever pip happened to resolve today.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Only the dependency manifests, so this layer caches until they change.
COPY pyproject.toml uv.lock ./

# --no-dev keeps jupyter, mlflow, streamlit and the plotting stack out of the
# image; --frozen fails if the lockfile is stale rather than silently drifting.
RUN uv sync --frozen --no-dev --no-install-project


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

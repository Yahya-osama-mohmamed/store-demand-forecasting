"""
FastAPI REST API — Real-time demand forecasting endpoints.

Provides:
- GET /health: Health check and model status
- GET /metrics: Basic API usage and performance metrics
- POST /predict: Single (date, store, item) forecast
- POST /predict/batch: CSV upload for bulk forecasts
"""

import datetime
import io
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.schemas import (
    BatchForecastResponse,
    ForecastInput,
    ForecastResponse,
    HealthResponse,
)
from pipeline_lib import FEATURE_NAMES_PATH, FINAL_PIPELINE_PATH, MODEL_METADATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

logger = get_logger("api")

# =============================================================================
# App Initialization & State
# =============================================================================

app = FastAPI(
    title="Store Item Demand Forecasting API",
    description="REST API for forecasting daily store-item demand.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
START_TIME = time.time()
metrics = {
    "total_predictions": 0,
    "batch_requests": 0,
    "errors": 0,
    "avg_latency_ms": 0.0,
    "predict_requests": 0,  # request count backing the latency average
}

# Lazy loading of model to avoid startup crashes if model isn't built yet.
# The loaded pipeline is keyed by the model file's mtime, so retraining
# (or artifacts appearing after API startup) is picked up automatically —
# no stale globals, no restart required.
MODEL_PIPELINE = None
FEATURE_NAMES = None
MODEL_METADATA = {}
_LOADED_MODEL_MTIME = None


def load_model():
    """Load the trained pipeline, feature names, and metadata."""
    global MODEL_PIPELINE, FEATURE_NAMES, MODEL_METADATA, _LOADED_MODEL_MTIME

    try:
        if not FINAL_PIPELINE_PATH.exists():
            logger.warning(f"Model file not found at {FINAL_PIPELINE_PATH}. Run training first.")
            MODEL_PIPELINE = None
            _LOADED_MODEL_MTIME = None
            return None, None

        mtime = FINAL_PIPELINE_PATH.stat().st_mtime
        if MODEL_PIPELINE is not None and mtime == _LOADED_MODEL_MTIME:
            return MODEL_PIPELINE, FEATURE_NAMES

        MODEL_PIPELINE = joblib.load(FINAL_PIPELINE_PATH)
        _LOADED_MODEL_MTIME = mtime

        if FEATURE_NAMES_PATH.exists():
            FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)
        else:
            logger.warning("Feature names file not found. Assuming raw input matches training.")

        MODEL_METADATA = {}
        if MODEL_METADATA_PATH.exists():
            try:
                MODEL_METADATA = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
                logger.info(
                    f"Loaded metadata: model={MODEL_METADATA.get('best_model')}, "
                    f"test SMAPE={MODEL_METADATA.get('test_smape')}"
                )
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Could not parse model metadata: {e}.")

        logger.info(f"Successfully loaded model from {FINAL_PIPELINE_PATH}")
        return MODEL_PIPELINE, FEATURE_NAMES

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL_PIPELINE = None
        _LOADED_MODEL_MTIME = None
        return None, None


# =============================================================================
# Middleware
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and update latency metrics."""
    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000

        # Update rolling average latency — averaged over REQUESTS, not
        # prediction rows (a 1000-row batch is still one latency sample)
        if request.url.path.startswith("/predict"):
            n = metrics["predict_requests"]
            metrics["avg_latency_ms"] = (metrics["avg_latency_ms"] * n + process_time) / (n + 1)
            metrics["predict_requests"] = n + 1

        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Latency: {process_time:.2f}ms"
        )
        return response

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Unhandled exception during {request.method} {request.url.path}: {str(e)}")
        raise


# =============================================================================
# Helper Functions
# =============================================================================

def determine_demand_level(predicted: float, store: int, item: int, pipeline) -> str:
    """
    Compare the forecast to this store-item's historical average
    (learned by the pipeline's FeatureEngineer during training).
    """
    try:
        fe = pipeline.named_steps["features"]
        lookup = fe.store_item_mean_
        row = lookup[(lookup["store"] == store) & (lookup["item"] == item)]
        baseline = float(row["store_item_mean"].iloc[0]) if len(row) else fe.global_mean_
    except Exception:
        return "Unknown"

    if baseline <= 0:
        return "Unknown"
    ratio = predicted / baseline
    if ratio < 0.8:
        return "Low"
    elif ratio > 1.2:
        return "High"
    return "Normal"


def process_prediction(df: pd.DataFrame, pipeline) -> list:
    """Run forecasts and return formatted results."""
    # The pipeline is fully self-contained: calendar features, learned
    # aggregates, and encoding all happen inside predict. Raw
    # (date, store, item) records go straight in.
    predictions = np.clip(pipeline.predict(df), 0, None)  # demand can't be negative

    results = []
    for (_, row), pred in zip(df.iterrows(), predictions, strict=False):
        results.append(ForecastResponse(
            predicted_sales=round(float(pred), 2),
            demand_level=determine_demand_level(
                float(pred), int(row["store"]), int(row["item"]), pipeline,
            ),
        ))

    return results


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", include_in_schema=False)
def root():
    """Redirect root to Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """API health check endpoint."""
    pipeline, _ = load_model()

    status = "healthy" if pipeline is not None else "degraded (model not loaded)"
    uptime = time.time() - START_TIME

    model_version = "unknown"
    if FINAL_PIPELINE_PATH.exists():
        mtime = FINAL_PIPELINE_PATH.stat().st_mtime
        model_version = datetime.datetime.fromtimestamp(mtime).isoformat()

    return HealthResponse(
        status=status,
        model_version=model_version,
        uptime_seconds=uptime,
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
    )


@app.get("/metrics")
def get_metrics():
    """Return basic API metrics."""
    return metrics


@app.post("/predict", response_model=ForecastResponse)
def predict_demand(forecast_request: ForecastInput):
    """Forecast demand for a single (date, store, item)."""
    pipeline, _ = load_model()

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")

    try:
        df = pd.DataFrame([forecast_request.model_dump()])
        result = process_prediction(df, pipeline)[0]
        metrics["total_predictions"] += 1
        return result

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}") from e


@app.post("/predict/batch", response_model=BatchForecastResponse)
def predict_batch(file: UploadFile = File(...)):
    """Forecast demand for a batch of (date, store, item) rows via CSV."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    pipeline, _ = load_model()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")

    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required = {"date", "store", "item"}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"CSV is missing required columns: {sorted(missing)}",
            )

        results = process_prediction(df, pipeline)
        metrics["batch_requests"] += 1
        metrics["total_predictions"] += len(results)

        return BatchForecastResponse(
            predictions=results,
            total_processed=len(results),
        )

    except HTTPException:
        raise
    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}") from e

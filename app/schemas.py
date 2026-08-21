"""
Pydantic Schemas — API Request and Response Models.

Defines the expected structure and validation rules for:
1. Single forecast requests
2. Forecast responses
3. API health status
"""

import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ForecastInput(BaseModel):
    """
    Pydantic model for a single demand-forecast request.
    Validates entity ranges and date format before anything reaches the model.
    """

    date: str = Field(..., description="Forecast date in YYYY-MM-DD format")
    store: int = Field(..., ge=1, le=10, description="Store ID (1-10)")
    item: int = Field(..., ge=1, le=50, description="Item ID (1-50)")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError("date must be a valid YYYY-MM-DD date") from exc
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "date": "2018-01-15",
                "store": 1,
                "item": 1,
            }
        }
    )


class ForecastResponse(BaseModel):
    """Pydantic model for the forecast response."""
    predicted_sales: float = Field(..., description="Forecasted units sold (>= 0)")
    demand_level: str = Field(
        ...,
        description="Demand vs this store-item's historical average: Low / Normal / High",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_sales": 21.4,
                "demand_level": "Normal",
            }
        }
    )


class BatchForecastResponse(BaseModel):
    """Pydantic model for batch forecast response."""
    predictions: list[ForecastResponse]
    total_processed: int


class HealthResponse(BaseModel):
    """Pydantic model for API health check response."""
    status: str = Field(..., description="API operational status")
    model_version: str = Field(..., description="Loaded model version/timestamp")
    uptime_seconds: float = Field(..., description="How long the API has been running")
    timestamp: str = Field(..., description="Current ISO-8601 timestamp")

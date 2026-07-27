import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


@pytest.fixture
def valid_payload():
    return {
        "date": "2018-01-15",
        "store": 1,
        "item": 1,
    }


def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "model_version" in data
    assert "uptime_seconds" in data
    assert "timestamp" in data


def test_predict_endpoint_validation_error():
    """Missing fields must return 422 Unprocessable Entity."""
    response = client.post("/predict", json={"date": "2018-01-15"})
    assert response.status_code == 422


def test_predict_endpoint_rejects_out_of_range_entities(valid_payload):
    """Store 99 / item 0 don't exist — pydantic must reject them."""
    bad_store = dict(valid_payload, store=99)
    assert client.post("/predict", json=bad_store).status_code == 422

    bad_item = dict(valid_payload, item=0)
    assert client.post("/predict", json=bad_item).status_code == 422


def test_predict_endpoint_rejects_bad_date(valid_payload):
    """Malformed dates must be rejected before reaching the model."""
    bad_date = dict(valid_payload, date="15/01/2018")
    assert client.post("/predict", json=bad_date).status_code == 422


def test_predict_endpoint_no_model(valid_payload, monkeypatch):
    """The predict endpoint must 503 cleanly when the model isn't loaded."""
    monkeypatch.setattr("app.api.load_model", lambda: (None, None))

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 503
    assert "Model is not loaded" in response.json()["detail"]

# tests/test_api.py

from fastapi.testclient import TestClient
from app.fastapi_backend import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()

    
# tests/test_api.py
from starlette.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
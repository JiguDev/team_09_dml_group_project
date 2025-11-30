# tests/test_api.py
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json().get("status") == "ok"
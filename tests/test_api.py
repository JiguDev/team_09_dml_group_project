# tests/test_api.py
import requests
def test_health():
    r = requests.get("http://localhost:8000/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

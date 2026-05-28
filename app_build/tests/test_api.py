import pytest
from fastapi.testclient import TestClient
import sys, os

# Ensure backend module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from main import app

client = TestClient(app)

def test_chat_endpoint():
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"query": "What is the law?", "mode": "auto"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

def test_ingest_endpoint():
    response = client.post("/api/ingest")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "queued" in data["message"]

def test_frontend_serve():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

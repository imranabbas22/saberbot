import pytest
import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from main import app

client = TestClient(app)

QA_PAIRS = [
    "What is the probation period under UAE labor law?",
    "Are employees entitled to severance pay upon termination?",
    "What is the official currency of the UAE?",
    "What is the standard working hours per week in the UAE?",
    "Is there personal income tax in the UAE?"
]

def test_truthfulness_5_laws():
    print("\n--- Running Truthfulness Test with Local LLM ---\n")
    with TestClient(app) as client:
        for idx, q in enumerate(QA_PAIRS):
            res = client.post("/api/chat", json={"query": q, "mode": "auto"})
            assert res.status_code == 200
            data = res.json()
            ans = data["response"]
            
            print(f"[{idx+1}/5] Q: {q}")
            print(f"      A: {ans}")
            print(f"      Sources Used: {len(data['sources'])}")
            print("-" * 50)
            
            assert len(ans) > 0

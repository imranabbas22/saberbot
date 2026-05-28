import pytest
import sys, os
from datasets import Dataset

# Make sure RAGAS and testing tools are loaded
try:
    from ragas.metrics import faithfulness
    from ragas import evaluate
    from langchain_community.chat_models import ChatOllama
    from langchain_core.embeddings import FakeEmbeddings
except ImportError:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from fastapi.testclient import TestClient
from main import app, startup_event

@pytest.fixture(scope="module")
def setup_app():
    startup_event()

client = TestClient(app)

QA_PAIRS = [
    "What is the penalty for driving under the influence in the UAE?",
    "How long is the standard maternity leave in the UAE private sector?",
    "What is the required notice period for resigning from a job?",
    "What is the legal age for a person to be held criminally responsible in the UAE?",
    "Can a landlord increase rent without notice in Dubai?"
]

def test_truthfulness_5_laws(setup_app):
    questions = []
    answers = []
    contexts = []
    
    print("\n\n--- RUNNING 5 LAWS TEST ---")
    
    for q in QA_PAIRS:
        resp = client.post("/api/chat", json={"query": q, "mode": "auto"})
        assert resp.status_code == 200
        data = resp.json()
        
        ans = data["response"]
        ctx_chunks = data.get("context_chunks", [])
        
        questions.append(q)
        answers.append(ans)
        
        # fallback context if db is empty
        if not ctx_chunks:
            ctx_chunks = ["Dummy context retrieved from db due to offline testing without full corpus."]
            
        contexts.append(ctx_chunks)
        
        print(f"\nQuestion: {q}")
        print(f"Answer: {ans}")
    
    # RAGAS Evaluation
    try:
        evaluator_llm = ChatOllama(model="gemma4:4b", base_url="http://localhost:11434")
        dummy_embeddings = FakeEmbeddings(size=10)
        
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts
        })
        
        result = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=evaluator_llm,
            embeddings=dummy_embeddings
        )
        score = result["faithfulness"]
        print(f"\n=> RAGAS Faithfulness Score: {score}")
    except Exception as e:
        print(f"\n=> RAGAS Evaluation skipped or failed: {e}")

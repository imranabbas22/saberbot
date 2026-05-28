"""
UAE LAW RAG — Configuration (shared between rag modules)
Reads from the production config.py or env vars.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# ---- Storage paths ----
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
DB_FOLDER = os.getenv("DB_FOLDER", "db")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "outputs")

# ---- Chunking ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ---- Embeddings ----
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# ---- LLM (local defaults) ----
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# ---- LLM Parameters ----
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
LLM_THINKING_TEMP = float(os.getenv("LLM_THINKING_TEMP", "0.6"))
LLM_QUICK_TEMP = float(os.getenv("LLM_QUICK_TEMP", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ---- Retrieval ----
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))

# ---- Agentic features ----
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "True").lower() == "true"
ENABLE_SELF_CRITIQUE = os.getenv("ENABLE_SELF_CRITIQUE", "True").lower() == "true"
ENABLE_REFINEMENT = os.getenv("ENABLE_REFINEMENT", "True").lower() == "true"

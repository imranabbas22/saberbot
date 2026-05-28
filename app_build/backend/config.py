"""
UAE Law RAG — Production Configuration
Reads from .env file with sensible defaults.
"""
import os
from pathlib import Path

# Try loading python-dotenv (optional — works with env vars directly too)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# ------------------------------------------------------------------ #
# Server
# ------------------------------------------------------------------ #
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))
WORKERS = int(os.getenv("WORKERS", "1"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# ------------------------------------------------------------------ #
# CORS
# ------------------------------------------------------------------ #
_CORS_RAW = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _CORS_RAW.split(",") if o.strip()] or [
    "http://localhost:5173",  # Vite dev
    "http://localhost:8002",  # Self
]

# ------------------------------------------------------------------ #
# LLM Provider
# ------------------------------------------------------------------ #
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "api"

# Ollama
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Cloud API
API_PROVIDER = os.getenv("API_PROVIDER", "groq")  # groq, openai, together, openrouter
API_MODEL = os.getenv("API_MODEL", "llama-3.3-70b-versatile")
API_KEY = os.getenv("API_KEY", "")

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()  # UAE_RAG/
DB_DIR = str(PROJECT_ROOT / "db" / "chroma")
PAGETREE_PATH = str(PROJECT_ROOT / "db" / "pagetree_index.json")
BM25_PATH = str(PROJECT_ROOT / "db" / "bm25_index.pkl")
FRONTEND_DIR = str(PROJECT_ROOT / os.getenv("FRONTEND_DIR", "app_build/frontend/dist"))

# ------------------------------------------------------------------ #
# Rate Limiting
# ------------------------------------------------------------------ #
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# ------------------------------------------------------------------ #
# Embedding
# ------------------------------------------------------------------ #
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
TOP_K = int(os.getenv("TOP_K", "5"))

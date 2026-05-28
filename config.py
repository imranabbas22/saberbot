# -----------------------------
# UAE LAW RAG — GLOBAL CONFIG
# -----------------------------

# -----------------------------
# DATA & STORAGE
# -----------------------------
DATA_FOLDER = "data"        # folder containing PDFs
DB_FOLDER = "db"            # ChromaDB storage
OUTPUT_FOLDER = "outputs"   # generated reports, email drafts, and print files

# -----------------------------
# CHUNKING
# -----------------------------
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# -----------------------------
# EMBEDDINGS
# -----------------------------
EMBEDDING_BACKEND = "sentence_transformers"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# -----------------------------
# LLM
# -----------------------------
LLM_BACKEND = "llama-cpp-python"     # local GGUF inference (no external APIs)
LLM_MODEL = "Qwen3-8B-Q4_K_M"       # GGUF model in models/
LLM_MODEL_FILE = "Qwen3-8B-Q4_K_M.gguf"

# -----------------------------
# LLM PARAMETERS
# -----------------------------
LLM_TEMPERATURE = 0.6      # thinking mode recommended temp
LLM_THINKING_TEMP = 0.6    # for legal reasoning (thinking mode)
LLM_QUICK_TEMP = 0.7       # for simple queries (non-thinking mode)
LLM_MAX_TOKENS = 4096

# -----------------------------
# RETRIEVAL
# -----------------------------
TOP_K = 6
MIN_SIMILARITY = 0.25      # below this → refuse to answer

# -----------------------------
# AGENTIC FEATURES
# -----------------------------
ENABLE_QUERY_REWRITE = True
ENABLE_SELF_CRITIQUE = True
ENABLE_REFINEMENT = True

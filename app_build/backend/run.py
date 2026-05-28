"""
UAE Law RAG — Production Server Launcher
=========================================
Reads configuration from config.py / .env.
"""
import os
import sys
from pathlib import Path

# Ensure backend dir is on path
backend_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(backend_dir))

import config as cfg
import uvicorn

if __name__ == "__main__":
    print(f"Starting UAE Law RAG ({cfg.LLM_PROVIDER} mode) on port {cfg.PORT}...")
    uvicorn.run(
        "main:app",
        host=cfg.HOST,
        port=cfg.PORT,
        reload=False,
        log_level=cfg.LOG_LEVEL,
        workers=cfg.WORKERS,
    )

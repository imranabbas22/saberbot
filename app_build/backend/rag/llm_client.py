"""
LLM Client — Ollama backend (GPU-accelerated)
===============================================
Uses Ollama's OpenAI-compatible API to run Qwen3.5:9B.
Ollama handles CUDA/GPU acceleration automatically.
No model loading at startup — just API calls.
"""

import json
import re

import requests

OLLAMA_BASE = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"  # Fast mode for faithfulness testing — no thinking mode, seconds per query
MODEL_NAME_FAST = "qwen2.5:7b"  # 4.7GB — no thinking mode, faster for structured output

_client_initialized = False


def ensure_model_available():
    """Verify the model is available in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = resp.json().get("models", [])
        for m in models:
            if m["name"] == MODEL_NAME:
                return True
        print(f"[WARN] Model '{MODEL_NAME}' not found in Ollama. Available: {[m['name'] for m in models]}")
        return False
    except requests.RequestException as e:
        print(f"[ERROR] Cannot reach Ollama at {OLLAMA_BASE}: {e}")
        return False


def generate(prompt: str, max_tokens: int = 1024, temperature: float = 0.6, **kwargs) -> dict:
    """
    Generate text using Ollama's /api/generate endpoint.

    Handles Qwen3.5 thinking mode: when response is empty but thinking
    exists, extracts the useful analysis from the thinking block.

    Returns dict in llama-cpp format: {"choices": [{"text": "..."}]}
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=300)
    data = resp.json()

    text = data.get("response", "")
    thinking = data.get("thinking", "")

    if thinking:
        # Qwen3.5 stores deep analysis in "thinking". Wrap it with <think> tags
        # so strip_thinking() can extract the final answer later.
        if text:
            text = f"<think>\n{thinking}\n</think>\n\n{text}"
        else:
            # All tokens went to thinking — strip the "Thinking Process:" header
            # and use the body as the actual response. Qwen3 thinking almost
            # always contains the full reasoning + final response.
            cleaned = re.sub(r"^Thinking Process:\s*", "", thinking, flags=re.IGNORECASE)
            text = f"<think>\n{thinking}\n</think>\n\n{cleaned}"
    elif not text:
        text = "No response generated. Try rephrasing your question."

    return {
        "choices": [
            {
                "text": text,
                "index": 0,
            }
        ],
        "usage": {
            "total_tokens": data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
            "eval_count": data.get("eval_count", 0),
            "tokens_per_second": data.get("eval_count", 0) / data.get("eval_duration", 1) * 1e9 if data.get("eval_duration") else 0,
        }
    }


def generate_chat(messages: list, max_tokens: int = 1024, temperature: float = 0.6, **kwargs) -> dict:
    """
    Use Ollama's OpenAI-compatible /v1/chat/completions endpoint.

    Handles Qwen3.5 thinking mode — when the model outputs all tokens
    to the thinking field and leaves content empty, extracts the response
    from the thinking block.

    Returns same format as generate() for drop-in compatibility.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    resp = requests.post(f"{OLLAMA_BASE}/v1/chat/completions", json=payload, timeout=300)
    data = resp.json()

    # Try standard message content first
    content = data["choices"][0]["message"].get("content", "")

    # If content is empty but Ollama returned thinking, extract it
    if not content:
        thinking = data.get("thinking", data["choices"][0]["message"].get("thinking", ""))
        if thinking:
            cleaned = re.sub(r"^Thinking Process:\s*", "", thinking, flags=re.IGNORECASE)
            content = f"<think>\n{thinking}\n</think>\n\n{cleaned}"
        else:
            content = "I couldn't generate a response. Please try rephrasing your question."

    return {
        "choices": [
            {
                "text": content,
                "index": 0,
            }
        ],
        "usage": data.get("usage", {}),
    }


def check_health() -> dict:
    """Check Ollama service health."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return {
                "ok": True,
                "model": MODEL_NAME,
                "available": any(m["name"] == MODEL_NAME for m in models),
                "all_models": [m["name"] for m in models],
            }
    except:
        pass
    return {"ok": False, "error": "Ollama not reachable"}


def generate_chat_fast(messages: list, max_tokens: int = 1024, temperature: float = 0.2, **kwargs) -> dict:
    """
    Fast generation using qwen2.5:7b — no thinking mode.
    Use for structured output tasks (compliance, classification, JSON extraction)
    where Qwen3 thinking mode is unnecessary overhead.
    """
    payload = {
        "model": MODEL_NAME_FAST,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    resp = requests.post(f"{OLLAMA_BASE}/v1/chat/completions", json=payload, timeout=120)
    data = resp.json()

    content = data["choices"][0]["message"]["content"]

    return {
        "choices": [
            {
                "text": content,
                "index": 0,
            }
        ],
        "usage": data.get("usage", {}),
    }

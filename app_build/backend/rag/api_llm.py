"""
Cloud API LLM Client — for deployment without local Ollama.
Supports Groq, OpenAI, Together, and OpenRouter.
"""
import json
import os
import requests
from pathlib import Path

# Load config
try:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
except ImportError:
    pass

_API_KEY = os.getenv("API_KEY", "")
_PROVIDER = os.getenv("API_PROVIDER", "groq")
_MODEL = os.getenv("API_MODEL", "llama-3.3-70b-versatile")

_API_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "together": "https://api.together.xyz/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


def generate_chat(messages: list, max_tokens: int = 1024, temperature: float = 0.6) -> dict:
    """Call cloud LLM API (OpenAI-compatible endpoint)."""
    if not _API_KEY:
        return _offline_fallback(messages)

    url = _API_ENDPOINTS.get(_PROVIDER)
    if not url:
        return _offline_fallback(messages, error=f"Unknown provider: {_PROVIDER}")

    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": _MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return _offline_fallback(messages, error="API timeout")
    except Exception as e:
        return _offline_fallback(messages, error=str(e))

    return {
        "choices": [{"text": text, "index": 0}],
        "usage": data.get("usage", {}),
    }


def check_health() -> dict:
    """Check if API key is configured."""
    if not _API_KEY:
        return {"ok": False, "error": "No API key configured"}
    return {
        "ok": True,
        "model": _MODEL,
        "provider": _PROVIDER,
        "available": True,
    }


def _offline_fallback(messages, error=None):
    """Return a graceful error when API is unavailable."""
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break

    err_msg = f" ({error})" if error else ""
    return {
        "choices": [{
            "text": (
                f"⚠️ The AI service is temporarily unavailable{err_msg}.\n\n"
                f"Please try again in a few moments. Your question was:\n> {last_user[:200]}"
            ),
            "index": 0,
        }],
        "usage": {},
    }

"""
UAE Law RAG — Production FastAPI Server
========================================
Serves:
  - /api/chat        — Query UAE laws with citations
  - /api/compliance  — Upload documents for compliance checking
  - /api/health      — Full system health
  - Frontend (React SPA)

LLM: Ollama (local) or cloud API (Groq/OpenAI/Together/OpenRouter)
"""
import os
import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ensure import paths
backend_dir = Path(__file__).parent
rag_dir = backend_dir / "rag"
sys.path.insert(0, str(rag_dir))
sys.path.insert(0, str(backend_dir))

# Production config
import config as cfg  # app_build/backend/config.py

# Feedback & session tracking
from feedback import (
    init_db, get_or_create_session, increment_request,
    submit_feedback, get_global_stats, MAX_REQUESTS_PER_SESSION,
    heartbeat, extend_session, expire_session,
)

from pipeline import RAGPipeline
import routers.compliance
from chat_engine import (
    analyze_query, compute_confidence,
    strip_thinking, CITATION_SYSTEM_PROMPT,
)

# LLM client — Ollama or cloud API
LLM_PROVIDER = cfg.LLM_PROVIDER
if LLM_PROVIDER == "api":
    from rag.api_llm import generate_chat, check_health as api_health
    _llm_health = lambda: api_health()
    _chat_fn = lambda msgs, **kw: generate_chat(msgs, **kw)
else:
    from rag.llm_client import generate_chat, check_health as ollama_health, ensure_model_available, MODEL_NAME
    _llm_health = lambda: ollama_health()
    _chat_fn = lambda msgs, **kw: generate_chat(msgs, **kw)

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("uaelaw")

# ------------------------------------------------------------------ #
# App Setup
# ------------------------------------------------------------------ #

class AppState:
    def __init__(self):
        self.rag_pipeline = None
        self.llm_available = False
        self.llm_model = cfg.OLLAMA_MODEL if LLM_PROVIDER == "ollama" else cfg.API_MODEL
        self.start_time = time.time()
        self.request_count = 0

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    logger.info("=" * 50)
    logger.info(f"  UAE Law RAG — Starting ({LLM_PROVIDER} mode)")
    logger.info("=" * 50)

    # Initialize RAG pipeline
    state.rag_pipeline = RAGPipeline(
        db_dir=cfg.DB_DIR,
        pagetree_path=cfg.PAGETREE_PATH,
        bm25_path=cfg.BM25_PATH,
    )

    # Initialize feedback/session DB
    init_db()
    logger.info("[OK] Feedback DB initialized")

    # Check LLM
    health = _llm_health()
    state.llm_available = health.get("ok", False) or health.get("available", False)
    if state.llm_available:
        logger.info(f"[OK] LLM ready: {state.llm_model} ({LLM_PROVIDER})")
    else:
        logger.warning(f"[WARN] LLM unavailable ({health.get('error', 'unknown')})")

    # Wire up compliance router
    routers.compliance.set_globals(state.rag_pipeline, state.llm_available)

    logger.info("Application startup complete.")
    yield
    logger.info("Application shutdown.")


app = FastAPI(
    title="UAE Law RAG — Offline Legal Assistant",
    description="100% offline UAE federal law query & compliance engine",
    version="3.2.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount compliance router
app.include_router(routers.compliance.router)

# ------------------------------------------------------------------ #
# Rate Limiting (simple in-memory)
# ------------------------------------------------------------------ #
_requests = {}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60  # 1 minute

    # Clean old entries
    _requests[client_ip] = [t for t in _requests.get(client_ip, []) if now - t < window]
    _requests[client_ip].append(now)

    if len(_requests[client_ip]) > cfg.RATE_LIMIT_PER_MINUTE:
        return JSONResponse(
            status_code=429,
            content={"error": f"Rate limit exceeded. Max {cfg.RATE_LIMIT_PER_MINUTE} requests/minute."}
        )

    response = await call_next(request)
    return response


# ------------------------------------------------------------------ #
# API Models
# ------------------------------------------------------------------ #

class ChatRequest(BaseModel):
    query: str
    mode: str = "auto"
    session_id: str = ""

class ChatResponse(BaseModel):
    response: str
    sources: list = []
    mode: str = ""
    model: str = ""
    timing_ms: int = 0
    session_id: str = ""
    remaining: int = 0


# ------------------------------------------------------------------ #
# /api/chat — Smart legal assistant with citation enforcement
# ------------------------------------------------------------------ #

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    state.request_count += 1
    t0 = time.time()

    # Track session
    session = get_or_create_session(request.session_id)
    session_id = session["session_id"]

    if not session["can_chat"]:
        elapsed = int((time.time() - t0) * 1000)
        return ChatResponse(
            response="You've used your 5 free queries. Please submit your feedback below to help us improve! 🙏",
            mode="limit_reached",
            model=state.llm_model,
            timing_ms=elapsed,
        )

    if not request.query.strip():
        return ChatResponse(
            response="Please ask a question about UAE law. I can help with federal laws including VAT, labor, data protection, corporate law, and more.",
            mode="empty",
            model=state.llm_model,
        )

    query = request.query.strip()

    # Step 1: Analyze query
    analysis = analyze_query(query)

    # Step 2: Retrieve context
    retrieved = state.rag_pipeline.retrieve(query, mode=request.mode, top_k=cfg.TOP_K)
    sources = [meta for _, meta, _ in retrieved]

    # Step 3: Compute confidence
    confidence = compute_confidence(retrieved)

    # Step 4: Route
    try:
        # CASE A: Vague + low confidence → clarify
        if analysis["needs_clarification"] and confidence["is_low_confidence"]:
            if not state.llm_available:
                response_text = (
                    f"I'd like to help you with your UAE legal question, but I need more details.\n\n"
                    f"{analysis['reason']}\n\n"
                    "Please tell me:\n"
                    "1. Which area of UAE law does your question relate to?\n"
                    "2. What specific situation are you facing?\n"
                    "3. Are you an individual, employee, or business owner?"
                )
                return ChatResponse(response=response_text, mode="clarify_offline", model=state.llm_model)

            messages = [
                {"role": "system", "content": _clarify_system_prompt(analysis)},
                {"role": "user", "content": query},
            ]
            output = _chat_fn(messages, max_tokens=256, temperature=0.7)
            clarification = strip_thinking(output["choices"][0]["text"].strip())
            return ChatResponse(response=clarification, mode="clarify", model=state.llm_model)

        # CASE B: Low confidence (specific query but no matches) → refuse
        if confidence["is_low_confidence"] and not analysis["needs_clarification"]:
            if not state.llm_available:
                return ChatResponse(
                    response=(
                        "I couldn't find relevant UAE law in my database to answer your question confidently.\n\n"
                        "Try rephrasing with specific legal terms or mentioning the specific law you're asking about."
                    ),
                    mode="refuse_offline",
                    model=state.llm_model,
                )

            messages = [
                {"role": "system", "content": _refuse_system_prompt(confidence, analysis)},
                {"role": "user", "content": query},
            ]
            output = _chat_fn(messages, max_tokens=256, temperature=0.7)
            refusal = strip_thinking(output["choices"][0]["text"].strip())
            return ChatResponse(response=refusal, mode="refuse", model=state.llm_model)

        # CASE C: Vague but some matches → clarify with context
        if analysis["needs_clarification"] and not confidence["is_low_confidence"]:
            if not state.llm_available:
                return ChatResponse(
                    response=f"I found {len(retrieved)} relevant legal documents, but your question could use more detail. Could you provide more specifics?",
                    sources=sources,
                    mode="clarify_found",
                    model=state.llm_model,
                )

            messages = [
                {"role": "system", "content": _clarify_system_prompt(analysis)},
                {"role": "user", "content": query},
            ]
            output = _chat_fn(messages, max_tokens=256, temperature=0.7)
            clarification = strip_thinking(output["choices"][0]["text"].strip())
            return ChatResponse(response=clarification, sources=sources, mode="clarify_found", model=state.llm_model)

        # CASE D: Good query + good context → ANSWER
        if not state.llm_available:
            context_preview = "\n".join(
                f"- **{meta.get('title', '?')}** | Article {meta.get('article', '?')}"
                for _, meta, _ in retrieved[:5]
            )
            return ChatResponse(
                response=f"I found {len(retrieved)} relevant legal documents:\n\n{context_preview}\n\nThe AI model is not reachable. Start Ollama to get full analysis.",
                sources=sources,
                mode="answer_offline",
                model=state.llm_model,
            )

        context = state.rag_pipeline.build_context(retrieved)
        messages = [
            {"role": "system", "content": f"{CITATION_SYSTEM_PROMPT}\n\nLegal Context (UAE Laws — ONLY use these as your source of truth):\n{context}"},
            {"role": "user", "content": query},
        ]
        output = _chat_fn(messages, max_tokens=2048, temperature=0.6)
        answer = strip_thinking(output["choices"][0]["text"].strip())

        # Log token usage
        usage = output.get("usage", {})
        if usage:
            tps = usage.get("tokens_per_second", 0)
            total_tokens = usage.get("total_tokens", 0)
            logger.info(f"[Chat] {total_tokens} tokens ({tps:.1f} tok/s) | mode=answer")

        elapsed = int((time.time() - t0) * 1000)

        # Track this request
        inc = increment_request(session_id)
        return ChatResponse(
            response=answer, sources=sources, mode="answer",
            model=state.llm_model, timing_ms=elapsed,
            session_id=session_id, remaining=inc["remaining"],
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return ChatResponse(
            response="I encountered an error processing your request. Please try rephrasing or try again later.",
            mode="error",
            model=state.llm_model,
        )


def _clarify_system_prompt(analysis: dict) -> str:
    domains = analysis.get("legal_domains", [])
    domain_hint = f" The user seems to be asking about: {', '.join(domains)}." if domains else ""
    return (
        "You are a helpful UAE legal assistant. The user asked a question, but you need "
        "more details before you can give them an accurate legal answer.\n\n"
        f"{analysis.get('reason', '')}{domain_hint}\n\n"
        "Ask 2-3 specific, short follow-up questions to narrow down:\n"
        "1. The specific legal domain (e.g., labor law, tax law, commercial companies law)\n"
        "2. The exact situation the user is facing\n"
        "3. Any missing details that would change the legal answer\n\n"
        "Be concise and helpful. Do NOT answer their legal question yet.\n"
        "Number your questions as 1, 2, 3."
    )


def _refuse_system_prompt(confidence: dict, analysis: dict) -> str:
    if not confidence.get("has_results"):
        return (
            "You are a helpful UAE legal assistant. The user asked a question but your "
            "legal database does NOT contain relevant information to answer it.\n\n"
            "Politely explain that you could not find the relevant UAE law, and suggest:\n"
            "1. Rephrasing the question with specific legal terms\n"
            "2. Mentioning the specific law or legal domain\n"
            "3. Checking with a qualified UAE legal professional\n\n"
            "Be honest. Do NOT fabricate laws or article numbers."
        )
    return (
        "You are a helpful UAE legal assistant. The user asked a question and you found "
        "some legal context, but it may not be directly relevant or the confidence is low.\n\n"
        "Explain that:\n"
        "1. You searched the UAE law database but could not find a direct match\n"
        "2. Share what partial information you have (if any)\n"
        "3. Ask the user to rephrase or provide more details\n"
        "4. Suggest they consult a UAE-qualified lawyer\n\n"
        "Be honest. Do NOT fabricate laws or article numbers."
    )


# ------------------------------------------------------------------ #
# /api/health
# ------------------------------------------------------------------ #

@app.get("/api/health")
async def health():
    health_data = _llm_health()
    return {
        "status": "ok",
        "version": "3.2.0",
        "llm_available": state.llm_available,
        "llm_model": state.llm_model,
        "llm_provider": LLM_PROVIDER,
        "llm_detail": health_data,
        "chroma_available": state.rag_pipeline.collection is not None if state.rag_pipeline else False,
        "pagetree_available": state.rag_pipeline.pagetree_available if state.rag_pipeline else False,
        "bm25_available": state.rag_pipeline.bm25_available if state.rag_pipeline else False,
        "uptime_seconds": int(time.time() - state.start_time),
        "total_requests": state.request_count,
    }


# ------------------------------------------------------------------ #
# Session & Feedback APIs
# ------------------------------------------------------------------ #

@app.get("/api/heartbeat/{session_id}")
async def session_heartbeat(session_id: str):
    """Check session status and refresh timeout if active."""
    return heartbeat(session_id)


@app.post("/api/session-extend/{session_id}")
async def session_extend(session_id: str):
    """User chose to continue — reset the 3-minute timer."""
    return extend_session(session_id)


@app.post("/api/session-expire/{session_id}")
async def session_expire(session_id: str):
    """User chose to exit — clean up files and expire session."""
    return expire_session(session_id)


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = get_or_create_session(session_id)
    return {
        "session_id": session["session_id"],
        "requests_used": session["requests_used"],
        "remaining": max(0, MAX_REQUESTS_PER_SESSION - session["requests_used"]),
        "can_chat": session["can_chat"],
        "max_requests": MAX_REQUESTS_PER_SESSION,
    }


class FeedbackRequest(BaseModel):
    session_id: str
    nps_score: int
    comment: str = ""


@app.post("/api/feedback")
async def feedback_endpoint(fb: FeedbackRequest):
    ok = submit_feedback(fb.session_id, fb.nps_score, fb.comment)
    return {"success": ok, "message": "Thank you for your feedback!" if ok else "Invalid."}


@app.get("/api/stats")
async def stats_endpoint():
    return get_global_stats()


# ------------------------------------------------------------------ #
# Serve Static Frontend
# ------------------------------------------------------------------ #

FRONTEND_DIST = Path(cfg.FRONTEND_DIR)
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    logger.info(f"[OK] Frontend mounted from: {FRONTEND_DIST}")
else:
    logger.warning(f"[INFO] No frontend dist at {FRONTEND_DIST}")

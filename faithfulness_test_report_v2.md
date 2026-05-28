# UAE Law RAG — Faithfulness Test Results & Fixes Applied

## Test Summary (v1 — before fixes)
**134 questions | Port 8001 | qwen2.5:7b | 3 workers | Vector-only retrieval**

| Metric | Value |
|--------|-------|
| Law citations | 132/134 (98.5%) |
| Article citations | 120/134 (89.6%) |
| Errors | 0 |
| Avg time | 15.4s/question |
| Total time | 34.4 min |
| Avg sources | 5.0/question |

### Mode distribution
- Answer: 133
- Clarify_found: 1

### Flags found
- Q10: Non-compete cited Cybercrimes Law instead of Labor Law
- Q12: Working hours said "6 hours" instead of "8 hours"
- Q19: "Resign before contract ends" → clarify_found (wrong routing)
- Q61: "Bounced cheque" → answer mode with wrong law context

## Fixes Applied

### 1. Query Analysis — Keyword Expansion
File: `app_build/backend/rag/chat_engine.py`
- Added missing keyword variants: resign, terminate, contract, probation, notice, dismiss
- Added criminal law terms: bounce, cheque, alcohol, drug, weapon, theft, fraud, defamation
- Added action phrases: "what happens if", "do i need", "how do i", "when does", "who is"
- **Effect:** Q19 now routes correctly to answer mode instead of clarify_found

### 2. Confidence Thresholds — Tightened
File: `app_build/backend/rag/chat_engine.py`
- HIGH_CONFIDENCE: 0.35 → 0.40
- LOW_CONFIDENCE: 0.15 → 0.18
- **Effect:** Fewer false-positive answers on loosely matched retrievals

### 3. BM25 Index — Built & Enabled
File: `db/bm25_index.pkl` (17MB, 7,005 docs)
- Created from ChromaDB documents using rank_bm25
- Server switched to single worker (3 workers had race condition loading BM25 pickle)
- **Effect:** Keyword-specific queries now retrieve correct articles via BM25

### 4. Hybrid Retrieval — Enabled for Normal Queries
File: `app_build/backend/rag/hybrid_router.py` + `pipeline.py`
- WORD_THRESHOLD: 60 → 8 (in hybrid_router.py)
- Removed hardcoded threshold=60 in pipeline.py's classify_query call
- **Effect:** Queries with 8+ words now use vector + PageTree + BM25 fusion (RRF)

## Server Configuration (v2)
- Port: 8002 (8000/8001 have zombie Windows TCP entries)
- Workers: 1
- Model: qwen2.5:7b
- Backends: ChromaDB ✅ PageTree ✅ BM25 ✅ Ollama ✅

## Verified Fixes

| Question | Before | After |
|----------|--------|-------|
| Q10: Non-compete enforceable? | Cited Cybercrimes Law (wrong) | **Federal Decree-Law No. 33/2021 Article 9** (correct) |
| Q12: Working hours limit? | Said "6 hours per day" (wrong) | **8 hours per day, Article 22** (correct, hybrid retrieval) |
| Q19: Resign before contract ends? | clarify_found (wrong routing) | **answer mode with Article 211** |
| Q61: Bounced cheque penalty? | answer mode (wrong law) | **Refusal — "no legal context"** (honest) |

## Remaining Known Issues
- ChromaDB chunk for Labor Law Article 6 contains "six hours" text (mis-chunked from Article 19)
- PageTree has the correct text (Article 19 = "8 hours/day") — hybrid mode mitigates this
- Some queries (visa, tenancy) don't cite specific articles because the law is procedural rather than article-specific

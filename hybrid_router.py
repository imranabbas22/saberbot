"""
Hybrid Router — classifies queries and merges retrieval results.

Routing logic:
    short query  (< threshold words) →  vector embeddings
    long query   (≥ threshold words) →  PageTree / BM25
    hybrid                           →  both, merged via RRF

Reciprocal Rank Fusion (RRF) is used to combine results from
heterogeneous scoring systems (cosine distance vs BM25 score).
"""

import hashlib
import re
from typing import List, Optional, Tuple, Dict, Literal

# ------------------------------------------------------------------ #
# Query classification
# ------------------------------------------------------------------ #

WORD_THRESHOLD = 60  # default; overridable from the UI


def classify_query(
    query: str,
    threshold: int = WORD_THRESHOLD,
    force_mode: str = "auto",
) -> Literal["vector", "pagetree", "bm25", "hybrid"]:
    """
    Decide which retrieval path to use.

    Parameters
    ----------
    query : str
        The user's question.
    threshold : int
        Word-count boundary between short (vector) and long (pagetree).
    force_mode : str
        "auto" | "vector" | "pagetree" | "bm25" — lets the user override.

    Returns
    -------
    "vector", "pagetree", "bm25", or "hybrid"
    """
    if force_mode in ("vector", "pagetree", "bm25"):
        return force_mode

    word_count = len(query.split())

    # Check for explicit legal references (law numbers, article numbers)
    has_legal_ref = bool(
        re.search(r"(article|law|decree|no\.\s*\(?\d+\)?)", query, re.IGNORECASE)
    )

    if word_count < threshold:
        # Short query with legal refs → hybrid to catch keywords too
        if has_legal_ref and word_count > 15:
            return "hybrid"
        return "vector"
    else:
        # Long query → PageTree primary, but also pull vectors for semantics
        return "hybrid"


# ------------------------------------------------------------------ #
# Reciprocal Rank Fusion
# ------------------------------------------------------------------ #

RRF_K = 60  # standard RRF constant


def _content_hash(text: str) -> str:
    """Deterministic hash for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def merge_results(
    vector_results: List[Tuple[str, Dict, float]],
    pagetree_results: List[Tuple[str, Dict, float]],
    bm25_results: Optional[List[Tuple[str, Dict, float]]] = None,
    k: int = 10,
) -> List[Tuple[str, Dict, float]]:
    """
    Merge ranked result lists using Reciprocal Rank Fusion.

    Each input is a list of (doc_text, metadata, score).
    Returns a unified list sorted by fused score, deduplicated.
    """
    # Map content_hash → {text, meta, rrf_score, sources}
    fused: Dict[str, dict] = {}

    def _add(results: List[Tuple[str, Dict, float]], source_tag: str):
        for rank, (text, meta, score) in enumerate(results):
            h = _content_hash(text)
            rrf_score = 1.0 / (rank + 1 + RRF_K)
            if h in fused:
                fused[h]["rrf_score"] += rrf_score
                fused[h]["sources"].add(source_tag)
            else:
                fused[h] = {
                    "text": text,
                    "meta": {**meta, "retrieval_method": source_tag},
                    "rrf_score": rrf_score,
                    "sources": {source_tag},
                }

    _add(vector_results, "vector")
    _add(pagetree_results, "pagetree")
    if bm25_results:
        _add(bm25_results, "bm25")

    # Update meta for items found by multiple methods
    for h, item in fused.items():
        if len(item["sources"]) > 1:
            methods = " + ".join(sorted(item["sources"]))
            item["meta"]["retrieval_method"] = f"hybrid ({methods})"

    # Sort by fused score descending
    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    return [
        (item["text"], item["meta"], item["rrf_score"])
        for item in ranked[:k]
    ]

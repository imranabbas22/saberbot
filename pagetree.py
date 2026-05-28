"""
PageTree — Vectorless hierarchical document index + BM25 retrieval engine.

Structure:
  Law (PDF) → Article blocks → Sub-chunks

Uses Okapi BM25 for keyword-based retrieval.  Article-level blocks
are capped at 3000 characters so they fit the 4096-token LLM window.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from rank_bm25 import BM25Okapi


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class PageNode:
    """A node in the page tree (article or sub-chunk)."""
    node_id: str
    text: str
    metadata: Dict[str, str]
    level: str  # "article" | "chunk"
    children_ids: List[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Index: load / query
# ------------------------------------------------------------------ #

INDEX_PATH = os.path.join("db", "pagetree_index.json")

_CACHES: Dict[str, Tuple[Optional[Dict], Optional[BM25Okapi], Optional[List[PageNode]]]] = {}


def _tokenize(text: str) -> List[str]:
    """Tokenize by extracting Unicode word characters (supports Arabic)."""
    return re.findall(r"\w+", text.lower())


def load_pagetree(index_path: str = INDEX_PATH):
    """Load the JSON index from disk and build BM25 corpus."""
    global _CACHES

    if index_path in _CACHES and _CACHES[index_path][1] is not None:
        return _CACHES[index_path][1], _CACHES[index_path][2]

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"PageTree index not found at {index_path}.  "
            "Run build_pagetree script first."
        )

    with open(index_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    nodes: List[PageNode] = []
    for entry in raw:
        nodes.append(PageNode(
            node_id=entry["node_id"],
            text=entry["text"],
            metadata=entry["metadata"],
            level=entry["level"],
            children_ids=entry.get("children_ids", []),
        ))

    corpus = [_tokenize(n.text) for n in nodes]
    bm25 = BM25Okapi(corpus)

    _CACHES[index_path] = (raw, bm25, nodes)

    return bm25, nodes


def invalidate_cache(index_path: str = INDEX_PATH):
    """Force reload on next call (after re-ingestion)."""
    global _CACHES
    if index_path in _CACHES:
        del _CACHES[index_path]
    elif index_path is None:
        _CACHES.clear()


# ------------------------------------------------------------------ #
# Search
# ------------------------------------------------------------------ #

def search_pagetree(
    query: str,
    k: int = 10,
    index_path: str = INDEX_PATH,
) -> List[Tuple[str, Dict[str, str], float]]:
    """
    BM25 search over the PageTree index.

    Returns a list of (document_text, metadata_dict, bm25_score)
    in the same shape as the vector `retrieve()` function so
    downstream code (reranker, context builder) works unchanged.
    """
    bm25, nodes = load_pagetree(index_path)

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Pair scores with nodes, sort descending
    scored = sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)

    results = []
    for score, node in scored[:k]:
        if score <= 0:
            break
        results.append((node.text, node.metadata, float(score)))

    return results

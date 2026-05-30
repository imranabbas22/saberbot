"""
RAG Pipeline — Orchestrates retrieval from three backends.

Coordinated retrieval:
  1. Vector (ChromaDB)   — semantic similarity via E5 embeddings
  2. PageTree (BM25)     — article-level hierarchical index
  3. BM25 (standalone)   — independent BM25 index

Results are fused via Reciprocal Rank Fusion (RRF),
then filtered and sorted with temporal supersession awareness.
"""

import os
import re
import random
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from pagetree import search_pagetree, load_pagetree
from bm25_search import load_bm25, search_bm25
from hybrid_router import classify_query, merge_results
from config import EMBEDDING_MODEL


# ── Supersession patterns found in UAE law text ──────────────────────────
# These capture statements like "This Law supersedes Federal Law No. (8) of 2004"
# or "Federal Decree-Law No. (3) of 2024 abrogates Law No. (2) of 2019"
SUPERSEDES_PATTERNS = [
    re.compile(r"(?:supersedes?|replaces?|abrogates?|repeals?|cancels?|revokes?|annuls?|overrides?)"
               r"\s+(?:the\s+)?(?:provisions\s+of\s+)?"
               r"(?:Federal\s+)?(?:Decree[-\s])?(?:Law|Resolution|Decision|Order|Decree)"
               r"(?:\s+No\.?\s*\(?(\d+)\)?)?"
               r"(?:\s+of\s+(?:the\s+year\s+)?(\d{4}))?",
               re.IGNORECASE),
    re.compile(r"(?:Federal\s+)?(?:Decree[-\s])?(?:Law|Resolution|Decision|Order|Decree)"
               r"(?:\s+No\.?\s*\(?(\d+)\)?)?"
               r"(?:\s+of\s+(?:the\s+year\s+)?(\d{4}))?"
               r"\s+(?:is\s+(?:hereby\s+)?)?(?:superseded|replaced|abrogated|repealed|cancelled|revoked|annulled)",
               re.IGNORECASE),
]

# Pattern to detect introductory articles that declare what a law supersedes
# (often Article 1 or the preamble: "This Law applies to... and supersedes...")
INTRO_SUPERSESSION_PATTERN = re.compile(
    r"(?:This\s+(?:Law|Decree[-\s]Law|Resolution|Decision))\s+.*?"
    r"(?:supersedes?|replaces?|abrogates?|repeals?|cancels?)"
    r".*?(?:Law|Decree|Resolution|Decision).*?(?:\d{4})",
    re.IGNORECASE | re.DOTALL
)


def parse_supersession_from_text(text: str) -> list:
    """
    Parse supersession declarations from the beginning of a law's text.

    Returns a list of (superseded_law_number, superseded_law_year) tuples.
    """
    # Only scan first 3000 chars — supersession declarations are in Article 1
    head = text[:3000]
    findings = []

    for pattern in SUPERSEDES_PATTERNS:
        for match in pattern.finditer(head):
            num_str = match.group(1)
            year_str = match.group(2)
            num = int(num_str) if num_str else None
            year = int(year_str) if year_str else None
            if year is not None:
                findings.append((num, year))

    return findings


# ── Temporal supersession filtering ──────────────────────────────────────

def filter_supersession(retrieved: list) -> list:
    """
    Apply supersession awareness: group retrieved chunks by same law identity,
    keep the newest where there's a conflict, annotate supersession info,
    and sort by year descending (newest first).

    Parameters
    ----------
    retrieved : list of (doc_text, metadata_dict, score)
        Output from retrieve().

    Returns
    -------
    list of (doc_text, metadata_dict, score)
        Filtered and sorted, with 'supersession_info' added to metadata.
    """

    if not retrieved:
        return []

    def _get_year(meta) -> int:
        try:
            return int(meta.get("law_year", 0))
        except (ValueError, TypeError):
            return 0

    def _law_key(meta) -> str:
        """Identity key for a law: type + number (laws with same number are typically revisions)."""
        num = meta.get("law_number", "")
        ltype = meta.get("law_type", "")
        return f"{ltype}|{num}"

    # ── Step 1: Group by law identity ──
    groups: dict[str, list] = {}
    for doc, meta, score in retrieved:
        key = _law_key(meta)
        groups.setdefault(key, []).append((doc, meta, score))

    # ── Step 2: For each group, keep ALL chunks but tag them ──
    # If multiple versions of the same law exist, tag the older ones
    # as superseded and reorder so newer chunks come first.
    all_years = {}
    for key, items in groups.items():
        years = [_get_year(m) for _, m, _ in items]
        max_year = max(years) if years else 0
        all_years[key] = max_year

    # ── Step 3: Annotate and sort ──
    result = []
    seen_law_ids = set()
    for doc, meta, score in retrieved:
        key = _law_key(meta)
        year = _get_year(meta)
        max_year_for_law = all_years.get(key, year)

        # Detect if this law has been superseded by a newer version
        superseded_by = []
        if year < max_year_for_law:
            # There's a newer version — mark as possibly superseded
            superseded_by.append(f"Newer version exists: {key} ({max_year_for_law})")

        # Mark if this law itself supersedes something
        # Check the text content for supersession language
        supersedes_list = parse_supersession_from_text(doc)

        # Build annotation
        annotation = {}
        if superseded_by:
            annotation["superseded_by"] = "; ".join(superseded_by)
            annotation["is_superseded"] = "true"
        else:
            annotation["is_superseded"] = "false"

        if supersedes_list:
            supersedes_str = "; ".join(
                f"Law No. {n} of {y}" if n else f"Law of {y}"
                for n, y in supersedes_list
            )
            annotation["supersedes"] = supersedes_str

        # Merge annotation into metadata
        meta = {**meta, **annotation}
        result.append((doc, meta, (year, score)))

    # ── Step 4: Sort by year descending, then by original score ──
    result.sort(key=lambda x: (-x[2][0], -x[2][1]))

    # ── Step 5: Rebuild final list with original score type (preserve for confidence scoring) ──
    final = [(doc, meta, orig_score) for doc, meta, (_, orig_score) in result]

    return final


# ── RAG Pipeline class ───────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, db_dir="db/chroma", pagetree_path="db/pagetree_index.json", bm25_path="db/bm25_index.pkl", embed_model=None):
        # Resolve paths starting from the project root (the parent of app_build/)
        # __file__ is at app_build/backend/rag/pipeline.py
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        app_build_dir = os.path.abspath(os.path.join(backend_dir, '..'))

        # The actual DB/chroma lives at project root (sibling of app_build/)
        project_root = os.path.dirname(app_build_dir)

        self.db_dir = os.path.join(project_root, db_dir)
        self.pagetree_path = os.path.join(project_root, pagetree_path)
        self.bm25_path = os.path.join(project_root, bm25_path)

        # Load embedder (multilingual E5, supports English + Arabic)
        model_name = embed_model or EMBEDDING_MODEL
        try:
            self.embedder = SentenceTransformer(model_name)
            print(f"[OK] Embedder loaded: {model_name}")
        except Exception as e:
            print(f"[WARN] Embedder failed to load: {e}")
            self.embedder = None

        # Load Chroma (vector DB)
        try:
            self.client = chromadb.PersistentClient(
                path=self.db_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_collection("uae_laws")
            print(f"[OK] ChromaDB connected: {self.db_dir}")
            print(f"     Collection: uae_laws | Count: {self.collection.count()}")
        except Exception as e:
            print(f"[WARN] ChromaDB failed: {e}")
            self.collection = None

        # Load PageTree (BM25 over article-level index)
        self.pagetree_available = False
        try:
            load_pagetree(index_path=self.pagetree_path)
            self.pagetree_available = True
            print(f"[OK] PageTree loaded: {self.pagetree_path}")
        except Exception as e:
            print(f"[WARN] PageTree unavailable: {e}")

        # Load standalone BM25 index
        self.bm25_index, self.bm25_docs = load_bm25(index_path=self.bm25_path)
        self.bm25_available = self.bm25_index is not None
        if self.bm25_available:
            print(f"[OK] BM25 index loaded: {self.bm25_path}")

    def retrieve(self, query: str, mode: str = "auto", top_k: int = 5):
        """Retrieve relevant documents using the appropriate mode(s)."""
        force = mode if mode in ["vector", "pagetree", "bm25"] else "auto"
        active_mode = classify_query(query, force_mode=force)

        vector_results = []
        pagetree_results = []
        bm25_results = []

        # 1. Vector search (semantic)
        if active_mode in ("vector", "hybrid") and self.collection and self.embedder:
            try:
                q_emb = self.embedder.encode(
                    [f"query: {query}"],
                    normalize_embeddings=True,
                )[0]
                res = self.collection.query(
                    query_embeddings=[q_emb],
                    n_results=top_k,
                )
                if res and res["documents"] and len(res["documents"]) > 0:
                    docs = res["documents"][0]
                    metas = res["metadatas"][0]
                    dists = res["distances"][0]
                    vector_results = list(zip(docs, metas, dists))
            except Exception as e:
                print(f"[WARN] Vector search error: {e}")

        # 2. PageTree search (article-level BM25)
        if active_mode in ("pagetree", "hybrid") and self.pagetree_available:
            pagetree_results = search_pagetree(query, k=top_k, index_path=self.pagetree_path)

        # 3. Standalone BM25
        if active_mode in ("bm25", "hybrid") and self.bm25_available:
            bm25_results = search_bm25(query, self.bm25_index, self.bm25_docs, k=top_k)

        # Merge results based on mode
        if active_mode == "hybrid":
            retrieved = merge_results(vector_results, pagetree_results, bm25_results, k=top_k)
        elif active_mode == "pagetree":
            retrieved = pagetree_results
        elif active_mode == "bm25":
            retrieved = bm25_results
        else:
            retrieved = vector_results

        # ── Apply supersession filtering & temporal sorting ──
        retrieved = filter_supersession(retrieved)

        return retrieved

    def build_context(self, retrieved):
        """Build a formatted context string for the LLM prompt.

        Chunks are already sorted by year descending (newest first)
        from filter_supersession. This method adds supersession annotations
        to the header.
        """
        parts = []
        for doc, meta, score in retrieved:
            title = meta.get("title", "Unknown Law")
            article = meta.get("article", "")
            law_type = meta.get("law_type", "")
            law_number = meta.get("law_number", "")
            law_year = meta.get("law_year", "")
            method = meta.get("retrieval_method", "vector")
            is_superseded = meta.get("is_superseded", "false")
            supersedes = meta.get("supersedes", "")
            superseded_by = meta.get("superseded_by", "")

            header = f"[{title}"
            if law_number:
                header += f" | No. {law_number}"
            if law_year:
                header += f" | {law_year}"
            if article:
                header += f" | Article {article}"

            # Append supersession annotations
            notes = []
            if supersedes:
                notes.append(f"supersedes: {supersedes}")
            if is_superseded == "true" and superseded_by:
                notes.append(f"⚠️  SUPERSEDED by: {superseded_by}")
            if notes:
                header += f" | {'; '.join(notes)}"

            header += f"] (retrieved via {method})"

            parts.append(f"{header}\n{doc}")

        return "\n\n---\n\n".join(parts)

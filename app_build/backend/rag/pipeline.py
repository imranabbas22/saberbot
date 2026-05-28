"""
RAG Pipeline — Orchestrates retrieval from three backends.

Coordinated retrieval:
  1. Vector (ChromaDB)   — semantic similarity via E5 embeddings
  2. PageTree (BM25)     — article-level hierarchical index
  3. BM25 (standalone)   — independent BM25 index

Results are fused via Reciprocal Rank Fusion (RRF).
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from pagetree import search_pagetree, load_pagetree
from bm25_search import load_bm25, search_bm25
from hybrid_router import classify_query, merge_results
from config import EMBEDDING_MODEL


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

        return retrieved

    def build_context(self, retrieved):
        """Build a formatted context string for the LLM prompt."""
        parts = []
        for doc, meta, score in retrieved:
            title = meta.get("title", "Unknown Law")
            article = meta.get("article", "")
            law_type = meta.get("law_type", "")
            law_number = meta.get("law_number", "")
            law_year = meta.get("law_year", "")
            method = meta.get("retrieval_method", "vector")

            header = f"[{title}"
            if law_number:
                header += f" | No. {law_number}"
            if law_year:
                header += f" | {law_year}"
            if article:
                header += f" | Article {article}"
            header += f"] (retrieved via {method})"

            parts.append(f"{header}\n{doc}")

        return "\n\n---\n\n".join(parts)

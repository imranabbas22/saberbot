import pickle
import os
import re

def load_bm25(index_path="db/bm25_index.pkl"):
    if not os.path.exists(index_path):
        return None, None
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["docs"]

def search_bm25(query: str, bm25, docs, k=10):
    if not bm25 or not docs:
        return []
    tokenized_query = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    results = []
    for i in top_n:
        if scores[i] > 0:
            doc_info = docs[i]
            results.append((doc_info["text"], doc_info["metadata"], float(scores[i])))
    return results

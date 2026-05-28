"""Quick test: chat pipeline with real legal question + timing."""
import sys, os
import time

os.chdir("D:/UAE_RAG")
sys.path.insert(0, "D:/UAE_RAG/app_build/backend/rag")
sys.path.insert(0, "D:/UAE_RAG/app_build/backend")

from pipeline import RAGPipeline
from llama_cpp import Llama

# Load pipeline
print("Loading RAG pipeline...")
p = RAGPipeline()
print(f"ChromaDB: {p.collection.count() if p.collection else 'N/A'} docs")

# Test retrieval
t0 = time.time()
results = p.retrieve("What are the VAT requirements in UAE?")
print(f"Retrieval: {time.time()-t0:.1f}s | {len(results)} results")

context = p.build_context(results)
print(f"Context length: {len(context)} chars")

# Load LLM
print("Loading LLM...")
t0 = time.time()
llm = Llama(
    model_path="D:/UAE_RAG/app_build/backend/models/Qwen3-8B-Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=0,
    verbose=False,
)
print(f"LLM loaded in {time.time()-t0:.1f}s")

# Build prompt with context
system = """You are an expert UAE legal assistant. ALWAYS cite the exact Law number and Article. If unsure, say you don't know. Format your answer with:
- **Applicable Law:** [name, number, year]
- **Relevant Article(s):** [articles]
- **Analysis:** [step-by-step application]
- **Guidance:** [what the user should do]"""

prompt = f"<|im_start|>system\n{system}\n\nLegal Context:\n{context}\n<|im_end|>\n<|im_start|>user\nWhat are the VAT requirements for businesses in the UAE?<|im_end|>\n<|im_start|>assistant\n"

t0 = time.time()
output = llm(
    prompt,
    max_tokens=512,
    temperature=0.6,
    stop=["<|im_end|>", "<|im_start|>"],
    echo=False,
)
elapsed = time.time() - t0

raw = output["choices"][0]["text"].strip()
# Remove thinking block
import re
answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
token_count = len(raw.split())

print(f"\n{'='*60}")
print(f"Generated in {elapsed:.1f}s | ~{token_count} tokens | {token_count/elapsed:.1f} tok/s")
print(f"{'='*60}")
print(answer[:2000])
print(f"{'='*60}")

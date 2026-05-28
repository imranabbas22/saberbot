import time
from llama_cpp import Llama

t0 = time.time()
model_path = "D:/UAE_RAG/app_build/backend/models/Qwen3-8B-Q4_K_M.gguf"
print("Loading model...")

llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=8,    # Use more threads
    n_gpu_layers=0, # CPU only
    verbose=True,
    seed=42,
)
print(f"Loaded in {time.time()-t0:.1f}s")

# Test with a simple prompt
prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
t0 = time.time()
output = llm(prompt, max_tokens=50, temperature=0.6)
elapsed = time.time() - t0
tokens = len(output["choices"][0]["text"].split())
print(f"Response ({elapsed:.1f}s, ~{tokens} words): {output['choices'][0]['text'][:200]}")

import time
t0 = time.time()
from llama_cpp import Llama

model_path = "D:/UAE_RAG/app_build/backend/models/Qwen3-8B-Q4_K_M.gguf"
print(f'Loading Qwen3-8B Q4_K_M GGUF from {model_path}...')

llm = Llama(
    model_path=model_path,
    n_ctx=8192,
    n_threads=4,
    n_gpu_layers=-1,
    verbose=False,
    seed=42,
)
t1 = time.time()
print(f'Loaded in {t1-t0:.1f}s')

# Quick test
prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n'
output = llm(prompt, max_tokens=50, temperature=0.6, echo=False)
response = output['choices'][0]['text'].strip()
print(f'Test response: {response}')
print('OK - LLM works!')

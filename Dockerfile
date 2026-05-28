# UAE Law RAG — Production Dockerfile
# Multi-stage: build frontend, then create minimal runtime

# ---- Stage 1: Build frontend ----
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY app_build/frontend/package*.json ./
RUN npm ci
COPY app_build/frontend/ ./
RUN npm run build

# ---- Stage 2: Python runtime ----
FROM python:3.11-slim
WORKDIR /app

# Install system deps for sentence-transformers & chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app_build/ app_build/
COPY db/ db/
COPY config.py .
COPY pagetree.py .
COPY hybrid_router.py .
COPY faithfulness_test_runner.py .

# Install additional python packages for ingestion
RUN pip install --no-cache-dir pypdf2

# Copy frontend build from stage 1
COPY --from=frontend-builder /app/frontend/dist /app/app_build/frontend/dist

# Environment config (can be overridden)
ENV PORT=8002
ENV WORKERS=1
ENV LLM_PROVIDER=api
ENV CORS_ORIGINS=https://yourdomain.com

EXPOSE 8002

CMD ["python", "app_build/backend/run.py"]

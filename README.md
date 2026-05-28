# 🤖 SaberBot — UAE Law Guide

An educational AI tool for exploring UAE federal laws. Every answer cites specific articles. Zero data retention. Built as a portfolio project.

## Features

- **Citation-First Answers** — Every response includes exact law name, article number, and clause
- **Zero Data Retention** — No tracking, no cookies, no storage of queries
- **7,005 UAE Laws** — Federal laws across labor, tax, criminal, family, business, tenancy, visa, and data privacy
- **5 Free Queries** — Try before you commit, then share feedback

## Tech Stack

- **Frontend:** React 19 + Vite
- **Backend:** FastAPI + Uvicorn
- **AI:** Groq API (Llama 3.3 70B)
- **Retrieval:** Hybrid Tripartite (ChromaDB Vector Search, BM25 Lexical Search, PageTree Hierarchy)
- **Deployment:** Nginx reverse proxy → FastAPI on Oracle Cloud (ARM free tier)

## Live Demo

**https://saberbot.duckdns.org**

## Running Locally

```bash
# Backend
cd app_build/backend
pip install -r requirements.txt
python run.py

# Frontend (development)
cd app_build/frontend
npm install
npm run dev
```

## Disclaimer

⚠️ **This is a portfolio project and educational tool.** It is NOT a substitute for professional legal advice. Always consult a qualified UAE lawyer for legal decisions.

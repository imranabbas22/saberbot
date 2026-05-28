"""
Compliance Check Engine — Professional Document Auditor
=========================================================
Full document scan against UAE federal law with:
  - Red / Amber / Green per-clause classification
  - Exact Law + Article + Clause citations
  - Suggested compliant rewrites preserving user's intent
  - Overall score with executive summary

Supports: .pdf, .docx, .xlsx, .txt, .jpg/.png (OCR)

Speed: uses batched LLM analysis (1 call per 10 clauses instead of N calls).
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import tempfile
import json
import re
import time
from typing import Optional

router = APIRouter()

# Set by main.py at startup
rag_pipeline = None
llm_available = False

def set_globals(pipeline, available: bool):
    global rag_pipeline, llm_available
    rag_pipeline = pipeline
    llm_available = available

from llm_client import generate_chat_fast


# ═══════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_text(file_path: str, filename: str) -> str:
    """Extract text from any supported file format."""
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".pdf":
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages)

        elif ext == ".docx":
            from docx import Document
            doc = Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif ext == ".xlsx":
            import pandas as pd
            sheets = pd.read_excel(file_path, sheet_name=None)
            parts = []
            for name, df in sheets.items():
                parts.append(f"--- Sheet: {name} ---\n{df.to_string()}")
            text = "\n\n".join(parts)

        elif ext in (".png", ".jpg", ".jpeg"):
            import pytesseract
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(file_path))

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

        else:
            raise ValueError(f"Unsupported format: {ext}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Text extraction failed: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted. File may be empty or scanned image without OCR.")

    return text


# ═══════════════════════════════════════════════════════════════════
# BATCH ANALYSIS PROMPT
# ═══════════════════════════════════════════════════════════════════

BATCH_SYSTEM = """You are a UAE federal law compliance auditor. Evaluate document clauses against UAE law.

For each clause, determine the status (Compliant, Gray Area, or Non-Compliant), cite the exact law and articles, explain why, state accountability, and suggest a compliant rewrite if needed.

Base your evaluation STRICTLY on the provided legal context. Never fabricate laws.
For Compliant clauses, set suggested_rewrite to "None needed".
For Non-Compliant clauses, always provide a specific suggested rewrite.

Return ONLY a JSON array. No other text, no explanations, no markdown. Format:
[
  {
    "clause_index": 0,
    "status": "Compliant|Gray Area|Non-Compliant",
    "relevant_law": "Law No. (X) of Year",
    "relevant_articles": ["Article X - description"],
    "reason": "Brief explanation",
    "accountability": "Who is liable",
    "suggested_rewrite": "Compliant version or None needed"
  },
  ...
]
"""


# ═══════════════════════════════════════════════════════════════════
# JSON PARSING — robust extraction from LLM output
# ═══════════════════════════════════════════════════════════════════

def safe_parse_json_array(raw: str) -> list:
    """Extract and parse a JSON array from LLM output."""
    # Strip thinking blocks
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Remove markdown code fences
    clean = re.sub(r"```(?:json)?\s*", "", clean)
    clean = clean.replace("```", "")

    # Find JSON array boundaries
    start = clean.find("[")
    end = clean.rfind("]") + 1

    if start == -1 or end <= start:
        raise ValueError(f"No JSON array found in output: {raw[:200]}...")

    json_str = clean[start:end]

    # Fix common issues
    json_str = json_str.replace("True", "true").replace("False", "false")
    json_str = re.sub(r':\s*None\b', ': null', json_str)

    return json.loads(json_str)


def fallback_finding(chunk: str, idx: int, reason: str = "Analysis unavailable") -> dict:
    return {
        "chunk_index": idx,
        "clause": chunk[:300] + ("..." if len(chunk) > 300 else ""),
        "status": "Gray Area",
        "relevant_law": "N/A",
        "relevant_articles": [],
        "reason": reason,
        "accountability": "N/A",
        "suggested_rewrite": None,
    }


# ═══════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════

def compute_compliance_score(findings: list) -> dict:
    """Calculate overall compliance score from per-clause findings."""
    total = len(findings)
    if total == 0:
        return {"score": 100, "red": 0, "amber": 0, "green": 0, "summary": "No clauses to analyze."}

    red = [f for f in findings if f.get("status") == "Non-Compliant"]
    amber = [f for f in findings if f.get("status") == "Gray Area"]
    green = [f for f in findings if f.get("status") == "Compliant"]

    score = round(((len(green) * 100) + (len(amber) * 50)) / total)

    if score >= 80:
        grade = "A \u2014 Largely Compliant"
    elif score >= 60:
        grade = "B \u2014 Needs Review"
    elif score >= 40:
        grade = "C \u2014 Significant Issues"
    else:
        grade = "D \u2014 High Risk"

    summary_lines = [
        f"Grade: {grade} | Score: {score}%",
        f"Compliant: {len(green)} | Needs Review: {len(amber)} | Non-Compliant: {len(red)}",
    ]
    if red:
        summary_lines.append("")
        summary_lines.append(f"CRITICAL: {len(red)} clause(s) directly violate UAE law. Immediate review required.")
        summary_lines.append("Violations found:")
        for r in red[:5]:
            articles = ", ".join(r.get("relevant_articles", ["N/A"]))
            law = r.get("relevant_law", "N/A")
            clause = r.get("clause", "")[:60]
            summary_lines.append(f"  - {clause}... \u2192 {law} ({articles})")

    return {
        "overall_score": score,
        "grade": grade,
        "red_count": len(red),
        "amber_count": len(amber),
        "green_count": len(green),
        "total_clauses": total,
        "summary": "\n".join(summary_lines),
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN COMPLIANCE PIPELINE — BATCHED for speed
# ═══════════════════════════════════════════════════════════════════

CHUNK_LENGTH = 1500
MAX_CHUNKS = 50
CHUNK_OVERLAP = 200

def run_compliance_scan(document_text: str) -> dict:
    """Full compliance scan — chunk, batch-retrieve, batch-analyze, score, report."""
    if not rag_pipeline:
        return {"error": "RAG pipeline not initialized. Server still starting."}

    t_start = time.time()

    # ── Chunk the document into clauses ──
    # Split by clause boundaries (numbered clauses, section headers, or blank lines)
    raw_lines = document_text.split("\n")
    clauses = []
    current_clause = []
    for line in raw_lines:
        stripped = line.strip()
        # Detect clause boundaries: numbered clauses, section headers, or blank-line separated paragraphs
        if re.match(r"^(Clause|Section|Article|Paragraph)\s+\d+", stripped, re.IGNORECASE):
            if current_clause:
                clauses.append("\n".join(current_clause).strip())
            current_clause = [line]
        elif not stripped and current_clause:
            # Blank line — could be a clause separator
            if len("\n".join(current_clause)) > 300:
                clauses.append("\n".join(current_clause).strip())
                current_clause = []
            else:
                current_clause.append(line)
        else:
            current_clause.append(line)
    if current_clause:
        clauses.append("\n".join(current_clause).strip())

    # If clause detection found nothing useful, fall back to character chunking
    if len(clauses) <= 1:
        clauses = []
        i = 0
        while i < len(document_text) and len(clauses) < MAX_CHUNKS:
            clauses.append(document_text[i:i + CHUNK_LENGTH])
            i += CHUNK_LENGTH - CHUNK_OVERLAP

    total_clauses = len(clauses)
    print(f"[Compliance] Document chunked: {total_clauses} clauses from {len(document_text)} chars")

    if total_clauses == 0:
        return {"error": "Document is empty."}

    # ── Retrieve context for each chunk ──
    # Use the full document as the search query for better context retrieval
    doc_query = document_text[:2000]  # Use first 2000 chars as query
    doc_contexts = {}
    for idx, chunk in enumerate(clauses):
        retrieved = rag_pipeline.retrieve(chunk, mode="auto")
        if retrieved:
            doc_contexts[idx] = rag_pipeline.build_context(retrieved)
        else:
            doc_contexts[idx] = "No relevant UAE law found for this clause."

    # ── Batched analysis: send ALL clauses + ALL contexts in one LLM call ──
    if llm_available:
        clauses_text = "\n\n---\n\n".join(
            f"Clause {idx + 1}:\n{chunk[:800]}" for idx, chunk in enumerate(clauses)
        )
        contexts_text = "\n\n---\n\n".join(
            f"Context for Clause {idx + 1}:\n{ctx[:1200]}" for idx, ctx in doc_contexts.items()
        )

        messages = [
            {"role": "system", "content": BATCH_SYSTEM},
            {"role": "system", "content": f"Retrieved UAE Legal Context for each clause:\n\n{contexts_text}"},
            {"role": "user", "content": f"Analyze these document clauses for UAE legal compliance. Return a JSON array with one entry per clause.\n\n{clauses_text}"},
        ]

        print(f"[Compliance] Sending batch of {total_clauses} clauses to LLM...")
        output = generate_chat_fast(messages, max_tokens=2048, temperature=0.2)
        raw = output["choices"][0]["text"].strip()
        print(f"[Compliance] LLM response: {len(raw)} chars")

        # Parse batched JSON response
        try:
            results = safe_parse_json_array(raw)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[Compliance] JSON parse failed: {e}. Using fallbacks.")
            results = []

        # Map results back to clauses
        findings = []
        for idx in range(total_clauses):
            match = next((r for r in results if r.get("clause_index") == idx), None)
            if match:
                findings.append({
                    "chunk_index": idx,
                    "clause": clauses[idx][:300] + ("..." if len(clauses[idx]) > 300 else ""),
                    "status": match.get("status", "Gray Area"),
                    "relevant_law": match.get("relevant_law", "N/A"),
                    "relevant_articles": match.get("relevant_articles", []),
                    "reason": match.get("reason", ""),
                    "accountability": match.get("accountability", "N/A"),
                    "suggested_rewrite": match.get("suggested_rewrite"),
                })
            else:
                findings.append(fallback_finding(clauses[idx], idx, "LLM did not analyze this clause in its response."))
    else:
        # LLM offline — return context preview only
        findings = []
        for idx, chunk in enumerate(clauses):
            retrieved = rag_pipeline.retrieve(chunk, mode="auto")
            findings.append({
                "chunk_index": idx,
                "clause": chunk[:300] + ("..." if len(chunk) > 300 else ""),
                "status": "Gray Area",
                "relevant_law": ", ".join(
                    f"{m.get('title','?')}" for _, m, _ in retrieved[:3]
                ) if retrieved else "N/A",
                "relevant_articles": [
                    f"Article {m.get('article','?')} \u2014 {m.get('title','?')}"
                    for _, m, _ in retrieved[:3] if m.get("article")
                ] if retrieved else [],
                "reason": "LLM not loaded. Automatic analysis unavailable.",
                "accountability": "N/A",
                "suggested_rewrite": None,
            })

    # ── Score ──
    score = compute_compliance_score(findings)

    elapsed = time.time() - t_start
    print(f"[Compliance] Scan complete: {len(findings)} clauses in {elapsed:.1f}s \u2014 Score: {score['overall_score']}%")

    return {
        "overall_score": score["overall_score"],
        "grade": score["grade"],
        "summary": score["summary"],
        "red_count": score["red_count"],
        "amber_count": score["amber_count"],
        "green_count": score["green_count"],
        "total_clauses": score["total_clauses"],
        "scan_time_seconds": round(elapsed, 1),
        "llm_available": llm_available,
        "findings": findings,
    }


# ═══════════════════════════════════════════════════════════════════
# API ENDPOINT
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/compliance")
async def check_compliance(file: UploadFile = File(...)):
    """Upload any document for full UAE legal compliance audit."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Validate file size (50MB cap)
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum 50MB.")

    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = extract_text(tmp_path, file.filename)
    finally:
        os.unlink(tmp_path)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in the document.")

    print(f"[Compliance] Scanning: {file.filename} ({len(text)} chars extracted)")

    result = run_compliance_scan(text)

    return {
        "filename": file.filename,
        "extracted_chars": len(text),
        **result,
    }

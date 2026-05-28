#!/usr/bin/env python
"""
Faithfulness Test Runner for UAE Law RAG
Runs 134+ questions through /api/chat and logs responses for analysis.
Saves results incrementally so we don't lose progress on timeout.
"""
import json
import time
import requests
import os
from datetime import datetime

API_URL = "http://localhost:8001/api/chat"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(PROJECT_DIR, "faithfulness_test_results.jsonl")
QUESTIONS_FILE = os.path.join(PROJECT_DIR, "faithfulness_test_questions.txt")

def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def ask_question(query, timeout=300):
    """Send a question to the UAE RAG API and return the response."""
    try:
        resp = requests.post(
            API_URL,
            json={"query": query},
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except requests.Timeout:
        return {"error": "timeout"}
    except requests.ConnectionError:
        return {"error": "connection_refused"}
    except Exception as e:
        return {"error": str(e)}

def analyze_result(query, result):
    """Analyze a single result for faithfulness indicators."""
    analysis = {
        "query": query,
        "has_error": "error" in result,
        "mode": result.get("mode", "unknown"),
        "has_sources": len(result.get("sources", [])) > 0,
        "source_count": len(result.get("sources", [])),
        "response_length": len(result.get("response", "")),
        "cites_law": False,
        "cites_article": False,
        "has_analysis_section": False,
        "has_guidance_section": False,
        "response_preview": result.get("response", "")[:200],
        "timestamp": datetime.now().isoformat(),
    }
    
    response_text = result.get("response", "")
    
    # Check for citation patterns
    if re_search(r"(?i)(Federal Decree|Law No\.|Decree-Law|Article \d+|applicable law)", response_text):
        analysis["cites_law"] = True
    
    if re_search(r"(?i)Article\s+\d+", response_text):
        analysis["cites_article"] = True
    
    if "**Analysis:**" in response_text:
        analysis["has_analysis_section"] = True
    
    if "**Guidance:**" in response_text:
        analysis["has_guidance_section"] = True
    
    return analysis

def re_search(pattern, text):
    """Simple regex-like check without importing re."""
    import re
    return bool(re.search(pattern, text))

def main():
    questions = load_questions()
    total = len(questions)
    print(f"Loaded {total} questions")
    
    results = []
    start_time = time.time()
    
    for i, query in enumerate(questions):
        q_start = time.time()
        
        print(f"[{i+1}/{total}] Asking: {query[:80]}...", end=" ", flush=True)
        
        result = ask_question(query)
        analysis = analyze_result(query, result)
        
        elapsed = time.time() - q_start
        mode = analysis["mode"]
        has_law = "L" if analysis["cites_law"] else "-"
        has_art = "A" if analysis["cites_article"] else "-"
        status = "OK" if not analysis["has_error"] else f"ERR"
        
        print(f"{status} | mode={mode} | {has_law}{has_art} | {elapsed:.0f}s")
        
        # Save immediately
        record = {
            "index": i,
            "query": query,
            "result": result,
            "analysis": analysis,
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(record)
        
        # Write incrementally
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        
        # Small delay to avoid hammering the server
        if i < total - 1:
            time.sleep(0.5)
    
    total_elapsed = time.time() - start_time
    print(f"\n=== COMPLETE: {total} questions in {total_elapsed:.0f}s ===\n")
    
    # Summary statistics
    modes = {}
    law_citations = 0
    article_citations = 0
    errors = 0
    
    for r in results:
        a = r["analysis"]
        modes[a["mode"]] = modes.get(a["mode"], 0) + 1
        if a["cites_law"]:
            law_citations += 1
        if a["cites_article"]:
            article_citations += 1
        if a["has_error"]:
            errors += 1
    
    print("=== SUMMARY ===")
    print(f"Total questions: {total}")
    print(f"Mode distribution: {modes}")
    print(f"Law citations: {law_citations}/{total}")
    print(f"Article citations: {article_citations}/{total}")
    print(f"Errors: {errors}/{total}")
    print(f"Average time: {total_elapsed/total:.1f}s per question")
    
    # Generate report file
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "avg_time_per_question": round(total_elapsed / total, 1),
        "mode_distribution": modes,
        "law_citations": law_citations,
        "article_citations": article_citations,
        "errors": errors,
        "error_rate_pct": round(errors / total * 100, 1),
        "results_file": RESULTS_FILE,
    }
    
    report_path = os.path.join(PROJECT_DIR, "faithfulness_test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()

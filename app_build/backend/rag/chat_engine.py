"""
Chat Engine — Smart UAE Legal Chatbot
======================================
Handles:
  1. Query analysis — detect vague/ambiguous questions
  2. Confidence scoring — decide if we should answer or ask for clarification
  3. Clarification generation — 2-3 targeted follow-up questions
  4. Citation-first formatting — enforce Law/Article/Clause citations
  5. Context windowing — fit legal context + answer in 4096 tokens

All prompts are designed for Qwen3 thinking mode.
"""

import re
from typing import List, Tuple, Dict, Optional


# ------------------------------------------------------------------
# Query Analysis — detect vague, ambiguous, or underspecified queries
# ------------------------------------------------------------------

# Patterns indicating the user IS being specific enough
SPECIFIC_INDICATORS = [
    r"\b(VAT|tax|corporate tax|income tax|customs)\b",
    r"\b(labor|labour|employment|employee|employer|worker|salary|wage|overtime|terminat(e|ion)|resign(s|ation|ed)?|gratuity|contract|probation|notice|dismiss|hire|recruit)\b",
    r"\b(visa|residency|immigration|sponsorship)\b",
    r"\b(data protection|privacy|personal data|GDPR)\b",
    r"\b(company|commercial|business|trade|license|LLC|establishment)\b",
    r"\b(criminal|penalty|fine|offense|crime|cybercrime|bounce|cheque|alcohol|drug|weapon|theft|fraud|defamation|threat|assault|arrest|deport)\b",
    r"\b(rent|tenancy|lease|property|real estate|landlord|tenant)\b",
    r"\b(bank|finance|loan|mortgage|insurance)\b",
    r"\b(article|law|decree|resolution|federal)\b",
    r"\b(how much|what is the|what are the|is it legal|can i|am i allowed|what happens if|do i need|how do i|when does|who is)\b",
    r"\b(comply|compliance|requirement|obligation|mandatory)\b",
    r"\b(dubai|abu dhabi|sharjah|ajman|ras al|umm al|fujairah)\b",
    r"\b(uae|united arab emirates|federal)\b",
]

# Generic/vague patterns that suggest we need more context
VAGUE_PATTERNS = [
    r"^(help|law|laws|legal|rights)$",
    r"^(tell me|explain|what) (about|the) (law|laws)$",
    r"^(i have a question|i need help|i want to know)$",
    r"^(can you help|help me|assist me)$",
    r"^(tell me everything|what do i need to know)$",
    r"^(what are my rights)\s*$",           # too broad — which rights?
    r"^(what should i do)\s*$",              # no context
    r"^(is this legal)\s*$",                 # what is "this"?
    r"^(what does the law say)\s*$",         # which law?
]

# Very short queries (under 5 words) are often ambiguous
MIN_SPECIFIC_WORDS = 5
MIN_SPECIFIC_INDICATORS = 1  # at least one specific legal domain

def analyze_query(query: str) -> dict:
    """
    Analyze a user query for specificity, domain, and ambiguity.

    Returns:
        {
            "is_vague": bool,
            "is_specific_enough": bool,
            "legal_domains": list[str],      # VAT, labor, data protection, etc.
            "word_count": int,
            "has_legal_refs": bool,
            "needs_clarification": bool,
            "reason": str,
        }
    """
    q = query.strip().lower()
    q_clean = re.sub(r'[^\w\s]', ' ', q)
    words = q_clean.split()
    word_count = len(words)

    # Check if this is clearly a vague/one-liner
    is_trivially_vague = False
    for pattern in VAGUE_PATTERNS:
        if re.match(pattern, q, re.IGNORECASE):
            is_trivially_vague = True
            break

    # Count specific indicators
    found_domains = set()
    for pattern in SPECIFIC_INDICATORS:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            domain = match.group(0).lower()
            found_domains.add(domain)

    # Check for legal references (article numbers, law numbers)
    has_legal_refs = bool(re.search(
        r"(article|law|decree|resolution|no\.\s*\(?\d+\)?)",
        query, re.IGNORECASE
    ))

    # Determine if specific enough
    is_specific_enough = (
        word_count >= MIN_SPECIFIC_WORDS and
        len(found_domains) >= MIN_SPECIFIC_INDICATORS
    ) or has_legal_refs

    # Decision logic
    needs_clarification = False
    reason = ""

    if is_trivially_vague:
        needs_clarification = True
        reason = "Question is too broad — no legal domain identified."
    elif word_count < MIN_SPECIFIC_WORDS and not has_legal_refs:
        needs_clarification = True
        reason = "Question is very short — need more details to give an accurate answer."
    elif not is_specific_enough:
        needs_clarification = True
        reason = "No specific legal domain detected — which area of UAE law does your question relate to?"

    return {
        "is_vague": is_trivially_vague,
        "is_specific_enough": is_specific_enough,
        "legal_domains": list(found_domains),
        "word_count": word_count,
        "has_legal_refs": has_legal_refs,
        "needs_clarification": needs_clarification,
        "reason": reason,
    }


# ------------------------------------------------------------------
# Confidence scoring — decide if retrieved context is strong enough
# ------------------------------------------------------------------

def compute_confidence(retrieved: List[Tuple[str, Dict, float]]) -> dict:
    """
    Compute confidence based on retrieval results.

    Scores are typically cosine distances from ChromaDB (lower = better)
    or BM25 scores (higher = better). We normalize to 0-1 scale.

    Returns:
        {
            "has_results": bool,
            "count": int,
            "mean_score": float,
            "best_score": float,
            "is_high_confidence": bool,
            "is_low_confidence": bool,
            "threshold_met": bool,
        }
    """
    if not retrieved:
        return {
            "has_results": False,
            "count": 0,
            "mean_score": 0.0,
            "best_score": 0.0,
            "is_high_confidence": False,
            "is_low_confidence": True,
            "threshold_met": False,
        }

    # ChromaDB returns distances (lower = more similar). Normalize.
    # BM25 returns float scores (raw, not normalized).
    scores = []
    for _, _, score in retrieved:
        # Normalize: if score is a distance (< 2 typically), invert
        if score < 2.0:
            # Distance-based (ChromaDB): 1.0 = identical, 0.0 = unrelated
            normalized = max(0.0, 1.0 - score)
        elif score > 100:
            # Raw BM25 score — cap and normalize
            normalized = min(1.0, score / 200.0)
        else:
            # RRF fused score — already 0-1 range
            normalized = score
        scores.append(normalized)

    mean_score = sum(scores) / len(scores)
    best_score = max(scores)

    # Thresholds
    HIGH_CONFIDENCE = 0.40   # best score above this → confident
    LOW_CONFIDENCE = 0.18    # best score below this → refuse/ask

    return {
        "has_results": True,
        "count": len(retrieved),
        "mean_score": round(mean_score, 4),
        "best_score": round(best_score, 4),
        "is_high_confidence": best_score >= HIGH_CONFIDENCE,
        "is_low_confidence": best_score < LOW_CONFIDENCE,
        "threshold_met": best_score >= LOW_CONFIDENCE,
    }


# ------------------------------------------------------------------
# Clarification generation — ask 2-3 sharp follow-up questions
# ------------------------------------------------------------------

def build_clarification_prompt(query: str, analysis: dict, confidence: dict) -> str:
    """
    Build a prompt that asks the LLM to generate clarifying questions.
    Returns a plain English string (uses non-thinking mode for speed).
    """
    domains = analysis.get("legal_domains", [])
    reason = analysis.get("reason", "")

    domain_hint = ""
    if domains:
        domain_hint = f"The user seems to be asking about: {', '.join(domains)}."

    return f"""<|im_start|>system
You are a helpful UAE legal assistant. The user asked a question, but you need more details before you can give them an accurate legal answer.

{reason}
{domain_hint}

Your task: Ask 2-3 specific, short follow-up questions that will help you narrow down:
1. The specific legal domain (e.g., labor law, tax law, commercial companies law)
2. The exact situation the user is facing
3. Any missing details that would change the legal answer

Be concise and helpful. Do NOT answer their legal question yet — only ask clarifying questions.
Number your questions as 1, 2, 3.

Format:
I'd like to help you, but I need a few more details to give you an accurate answer under UAE law:

1. [First clarifying question]
2. [Second clarifying question]
3. [Third clarifying question if needed]
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""


# ------------------------------------------------------------------
# Answer formatting — enforce citation-first structure
# ------------------------------------------------------------------

CITATION_SYSTEM_PROMPT = """You are an expert UAE federal law assistant. Your responses MUST follow this exact structure.

CRITICAL RULES:
1. EVERY answer must cite at least one specific Law number AND Article number.
2. NEVER make up laws. If the context does not support your answer, say "I do not have enough legal context to answer this confidently."
3. ALWAYS apply the law to the user's specific scenario.
4. ALWAYS provide practical next-step guidance.
5. Use the EXACT law names, article numbers, and clauses found in the context.
6. If multiple laws apply, cite all relevant ones.
7. Write your answer as plain text with markdown formatting. Do NOT use JSON, code blocks, or any structured data format.

Write your answer in this format (markdown text, not JSON):

**Applicable Law:** [Exact law name, Law No. (X) of Year YYYY]

**Relevant Article(s):**
- Article [X]: [Brief summary of what this article covers]
- Article [Y]: [Brief summary if applicable]

**Analysis:** [Step-by-step application of the law to the user's specific situation. Explain HOW the article applies to their facts.]

**Guidance:** [Clear, actionable next steps the user should take based on this law.]

**Important:** [Any caveats, deadlines, or exceptions the user should know about.]"""


def build_answer_prompt(query: str, context: str) -> str:
    """Build a prompt for answering with citations using thinking mode."""
    return f"""<|im_start|>system
{CITATION_SYSTEM_PROMPT}

Legal Context (UAE Laws — ONLY use these as your source of truth):
{context}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""


# ------------------------------------------------------------------
# Refusal prompt — "I don't know" but helpful
# ------------------------------------------------------------------

def build_refusal_prompt(query: str, confidence: dict, analysis: dict) -> str:
    """Build a refusal that's honest but helpful."""
    if not confidence.get("has_results"):
        return f"""<|im_start|>system
You are a helpful UAE legal assistant. The user asked a question but your legal database does NOT contain relevant information to answer it.

Your task: Politely explain that you could not find the relevant UAE law, and suggest:
1. Rephrasing the question with specific legal terms
2. Mentioning the specific law or legal domain they're asking about
3. Checking with a qualified UAE legal professional for the most current information

Be honest that you don't know, but be helpful about how they can get the answer.<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
    else:
        return f"""<|im_start|>system
You are a helpful UAE legal assistant. The user asked a question and you found some legal context, but it may not be directly relevant or the confidence is low.

Your task: Explain that:
1. You searched the UAE law database but could not find a direct match
2. Share what partial information you have (if any)
3. Ask the user to rephrase or provide more details about their specific situation
4. Suggest they consult a UAE-qualified lawyer for definitive advice

Be honest. Do NOT fabricate laws or article numbers.<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""


# ------------------------------------------------------------------
# Strip thinking blocks from response
# ------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 output."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return stripped if stripped else text.strip()

"""
Build the PageTree vectorless index from PDFs.

Reads all PDFs in pdf_english/, extracts text, splits into
article-level blocks (capped at 3000 chars), and saves the
hierarchical index to db/pagetree_index.json.

Run:
    python ingest/build_pagetree.py
"""

import json
import os
import re
import sys

import pdfplumber

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

PDF_DIR = "pdf_gazette"
INDEX_PATH = os.path.join("db", "pagetree_gazette_index.json")
MAX_ARTICLE_CHARS = 3000  # cap to fit 4096-token context window


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def extract_text_from_pdf(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_law_metadata(filename: str) -> dict:
    """Extract law metadata from filename (mirrors ingest_laws.py)."""
    name, _ = os.path.splitext(filename)
    law_id = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")

    year_match = re.search(r"(19|20)\d{2}", name)
    law_year = int(year_match.group()) if year_match else None

    num_match = re.search(r"No\.?\s*\((\d+)\)", name, re.IGNORECASE)
    law_number = int(num_match.group(1)) if num_match else None

    if "decree" in name.lower() and "law" in name.lower():
        law_type = "Federal Decree-Law"
    elif "federal law" in name.lower():
        law_type = "Federal Law"
    elif "cabinet" in name.lower():
        law_type = "Cabinet Decision"
    else:
        law_type = "Other"

    return {
        "law_id": law_id,
        "law_type": law_type,
        "law_number": str(law_number) if law_number else "",
        "law_year": str(law_year) if law_year else "",
        "title": name.strip(),
    }


# ------------------------------------------------------------------ #
# Article splitter
# ------------------------------------------------------------------ #

_ARTICLE_PATTERN = re.compile(
    r"(?=Article\s*\(?(\d+)\)?)",
    re.IGNORECASE,
)


def split_into_articles(text: str):
    """
    Split law text into article-level blocks.

    Returns list of (article_number_or_None, block_text).
    If no articles are found, the full text is returned as one block.
    """
    splits = _ARTICLE_PATTERN.split(text)

    if len(splits) <= 1:
        # No article markers found → return full text as preamble
        return [(None, text)]

    articles = []

    # First segment is the preamble (before Article 1)
    preamble = splits[0].strip()
    if preamble:
        articles.append((None, preamble))

    # Remaining segments come in pairs: (article_num, article_text)
    i = 1
    while i < len(splits) - 1:
        art_num = splits[i]
        art_text = splits[i + 1].strip()
        articles.append((int(art_num), art_text))
        i += 2

    return articles


def cap_text(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> list:
    """
    If text exceeds max_chars, split into sub-chunks.
    Returns list of text blocks, each ≤ max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            current = s
        else:
            current = current + " " + s if current else s
    if current:
        chunks.append(current.strip())

    return chunks if chunks else [text[:max_chars]]


# ------------------------------------------------------------------ #
# Main builder
# ------------------------------------------------------------------ #

def build_index():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    pdf_files = [
        os.path.join(PDF_DIR, f)
        for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    print(f"Found {len(pdf_files)} PDF(s) in {PDF_DIR}/")

    all_nodes = []
    node_count = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        meta = parse_law_metadata(filename)
        print(f"\nProcessing: {filename}")

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            print("  -> Empty PDF, skipping.")
            continue

        cleaned = clean_text(raw_text)
        articles = split_into_articles(cleaned)
        print(f"  -> {len(articles)} article block(s)")

        law_children = []

        for art_num, art_text in articles:
            blocks = cap_text(art_text, MAX_ARTICLE_CHARS)

            for chunk_idx, block in enumerate(blocks):
                node_id = f"pt_{meta['law_id']}_art{art_num or 'pre'}_{chunk_idx}"
                node_count += 1

                node_meta = {
                    **meta,
                    "article": str(art_num) if art_num else "preamble",
                    "chunk_index": str(chunk_idx),
                    "filename": filename,
                    "lang": "ar",
                }

                all_nodes.append({
                    "node_id": node_id,
                    "text": block,
                    "metadata": node_meta,
                    "level": "article" if len(blocks) == 1 else "chunk",
                    "children_ids": [],
                })

                law_children.append(node_id)

    # Save index
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(all_nodes, f, ensure_ascii=False, indent=2)

    print(f"\n✅ PageTree index saved to {INDEX_PATH}")
    print(f"   Total nodes: {node_count}")


if __name__ == "__main__":
    build_index()

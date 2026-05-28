import os
import io
import re
import json
import chromadb
from sentence_transformers import SentenceTransformer
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PDF_DIR = "pdf_english"
DB_DIR = "db/chroma"
COLLECTION_NAME = "uae_laws"
MODEL_NAME = "intfloat/multilingual-e5-base"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

# ---------------------------------------------------------
# CLEANING
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------
# METADATA EXTRACTION FROM FILENAME
# ---------------------------------------------------------

def parse_law_metadata(filename: str):
    """
    Extract:
    - law_type
    - law_number
    - law_year
    - title
    - law_id
    """

    name, _ = os.path.splitext(filename)

    # Clean for ID
    law_id = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")

    # Extract year
    year_match = re.search(r"(19|20)\d{2}", name)
    law_year = int(year_match.group()) if year_match else None

    # Extract law number inside parentheses (8), (32), etc.
    num_match = re.search(r"No\.?\s*\((\d+)\)", name, re.IGNORECASE)
    law_number = int(num_match.group(1)) if num_match else None

    # Law type detection
    if "decree" in name.lower() and "law" in name.lower():
        law_type = "Federal Decree-Law"
    elif "federal law" in name.lower():
        law_type = "Federal Law"
    elif "cabinet" in name.lower():
        law_type = "Cabinet Decision"
    else:
        law_type = "Other"

    # Title = cleaned filename
    title = name.strip()

    return {
        "law_id": law_id,
        "law_type": law_type,
        "law_number": law_number,
        "law_year": law_year,
        "title": title,
    }


# ---------------------------------------------------------
# ARTICLE DETECTION
# ---------------------------------------------------------

def detect_article(text: str):
    """
    Detect article number inside a chunk.
    Example matches:
    - Article 1
    - ARTICLE (2)
    - Article (12)
    """
    match = re.search(r"Article\s*\(?(\d+)\)?", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


# ---------------------------------------------------------
# MAIN INGEST PIPELINE
# ---------------------------------------------------------

def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Initializing Chroma at: {DB_DIR}")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    manifest_path = "scraper/download_manifest.json"
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Gather all superseded target laws
    amended_targets = set()
    for entry in manifest.get("downloads", []):
        title = entry.get("title", "")
        # Match Amending Law No. (X) of Year (Y)
        match = re.search(r"Amending.*?(?:Law|Resolution|Decree|Order|Decision)\s*No\.?\s*\(?(\d+)\)?\s*(?:of|for|for the year)\s*(\d{4})", title, re.IGNORECASE)
        if match:
            num, year = match.groups()
            amended_targets.add((int(num), int(year)))

    print(f"Detected {len(amended_targets)} laws that have been amended and should be superseded.")

    # 2. Filter pdfs from manifest
    valid_entries = []
    for entry in manifest.get("downloads", []):
        # We only want English laws here
        if entry.get("source") != "browse_legislation":
            continue

        local_path = entry.get("local_path")
        if not local_path or not os.path.exists(local_path):
            continue

        title = entry.get("title", "")
        year = entry.get("year")
        filename = os.path.basename(local_path)
        raw_meta = parse_law_metadata(filename)
        num = raw_meta.get("law_number")

        # Skip if it is superseded AND it's not itself the amending law
        if num and year and (int(num), int(year)) in amended_targets:
            if "amending" not in title.lower():
                print(f"  [!] Excluding superseded law: {title}")
                continue

        merged_meta = {
            "law_id": raw_meta["law_id"],
            "law_type": entry.get("type") or raw_meta["law_type"],
            "law_number": num,
            "law_year": year or raw_meta["law_year"],
            "title": title or raw_meta["title"],
        }
        valid_entries.append((local_path, merged_meta))

    print(f"\nFound {len(valid_entries)} valid legislation PDF(s) to ingest.")

    for pdf_path, meta in valid_entries:
        filename = os.path.basename(pdf_path)

        print(f"\nProcessing: {filename}")
        print(f"Metadata extracted: {meta}")

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            print("  -> Empty PDF, skipping.")
            continue

        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)

        print(f"  -> {len(chunks)} chunk(s)")

        docs_for_embed = [f"passage: {c}" for c in chunks]
        embeddings = model.encode(docs_for_embed, normalize_embeddings=True)

        ids = [f"{meta['law_id']}_en_{i}" for i in range(len(chunks))]

        metadatas = []
        for i, chunk in enumerate(chunks):
            meta_clean = {
                "law_id": str(meta["law_id"]),
                "law_type": str(meta["law_type"]),
                "law_number": str(meta["law_number"]) if meta["law_number"] is not None else "",
                "law_year": str(meta["law_year"]) if meta["law_year"] is not None else "",
                "title": str(meta["title"]),
                "filename": str(filename),
                "lang": "en",
                "chunk_index": str(i),
                "article": str(detect_article(chunk)) if detect_article(chunk) is not None else "",
            }
            metadatas.append(meta_clean)

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print("  -> Stored in vector DB")

    print("\nIngest complete.")


if __name__ == "__main__":
    main()
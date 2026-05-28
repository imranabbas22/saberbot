import json
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_PATH = "db/chroma"
LAWS_FILE = "data/laws.jsonl"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_laws():
    with open(LAWS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

def main():
    print("🚀 Loading embedder...")
    embedder = SentenceTransformer(EMBED_MODEL)
    embedder.max_seq_length = 512

    print("📦 Connecting to Chroma...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("uae_laws")

    print("📚 Ingesting laws...")
    for law in load_laws():
        law_id = law["id"]

        for lang in ["en", "ar"]:
            text = (
                (law[lang].get("preamble_html") or "") + "\n\n" +
                (law[lang].get("law_text") or "")
            )

            chunks = chunk_text(text)
            embeddings = embedder.encode(chunks, normalize_embeddings=True)

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{law_id}_{lang}_{i}"

                collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=[emb.tolist()],
                    metadatas=[{
                        "law_id": law_id,
                        "lang": lang,
                        "chunk_index": i,
                        "title": law[lang].get("title", "")
                    }]
                )

        print(f"✔ Ingested law {law_id}")

    print("🎉 Ingestion complete.")

if __name__ == "__main__":
    main()
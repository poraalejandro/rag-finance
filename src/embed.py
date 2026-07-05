"""
Reads chunks from data/processed/chunks.jsonl, generates embeddings with
Gemini's embedding-001 model, and stores them in a local ChromaDB collection.
Ticker and source are stored as metadata to enable per-company filtering later.
"""

import json
import os
from pathlib import Path

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

COLLECTION_NAME = "finance_10k"
EMBEDDING_MODEL = "models/embedding-001"
BATCH_SIZE = 100  # chunks per API call — balances throughput vs. rate limits


def load_chunks(path: Path) -> list[dict]:
    """Load all chunk records from the JSONL file into a list of dicts."""
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Call the Gemini API to embed a list of texts in one request.

    task_type="RETRIEVAL_DOCUMENT" tells the model these are passages to be
    stored in an index, not queries. This produces different (better) vectors
    for document retrieval vs. using the default general-purpose task type.
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="RETRIEVAL_DOCUMENT",
    )
    return result["embedding"]


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Return the ChromaDB collection, creating it if it doesn't exist yet.

    hnsw:space="cosine" sets the distance metric used for nearest-neighbour
    search. Cosine similarity is standard for text embeddings because it
    measures the angle between vectors (direction), ignoring their magnitude.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env file")
    genai.configure(api_key=api_key)

    print(f"Loading chunks from {CHUNKS_PATH.resolve()}")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"  {len(chunks)} chunks loaded\n")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = get_or_create_collection(client)

    total = len(chunks)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        batch_end = batch_start + len(batch)
        print(f"  Embedding chunks {batch_start + 1}–{batch_end} / {total}...")

        texts = [c["text"] for c in batch]
        embeddings = embed_batch(texts)

        # IDs must be unique strings. ticker + chunk_index is deterministic,
        # so re-running this script will upsert (overwrite) rather than duplicate.
        ids = [f"{c['ticker']}_{c['chunk_index']}" for c in batch]
        metadatas = [{"ticker": c["ticker"], "source": c["source"]} for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"\nDone. {total} chunks indexed in ChromaDB at {CHROMA_DIR.resolve()}")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} entries.")


if __name__ == "__main__":
    main()

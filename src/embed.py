"""
Reads chunks from data/processed/chunks.jsonl, generates embeddings with
Gemini's gemini-embedding-001 model, and stores them in a local ChromaDB collection.
Ticker and source are stored as metadata to enable per-company filtering later.
"""

import json
import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

COLLECTION_NAME = "finance_10k"
EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 100


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in response.embeddings]


def get_or_create_collection(chroma_client: chromadb.PersistentClient) -> chromadb.Collection:
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env file")

    gemini_client = genai.Client(api_key=api_key)

    print(f"Loading chunks from {CHUNKS_PATH.resolve()}")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"  {len(chunks)} chunks loaded\n")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = get_or_create_collection(chroma_client)

    already_indexed = set(collection.get(include=[])["ids"])
    print(f"  {len(already_indexed)} chunks already in ChromaDB\n")

    total = len(chunks)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        batch_end = batch_start + len(batch)

        ids = [f"{c['ticker']}_{c['chunk_index']}" for c in batch]

        if all(id in already_indexed for id in ids):
            print(f"  Chunks {batch_start + 1}–{batch_end} already indexed, skipping...")
            continue

        print(f"  Embedding chunks {batch_start + 1}–{batch_end} / {total}...")

        texts = [c["text"] for c in batch]
        embeddings = embed_batch(gemini_client, texts)

        metadatas = [{"ticker": c["ticker"], "source": c["source"]} for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        if batch_end < total:
            print("  Rate limit: waiting 62s before next batch...")
            time.sleep(62)

    print(f"\nDone. {total} chunks indexed in ChromaDB at {CHROMA_DIR.resolve()}")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} entries.")


if __name__ == "__main__":
    main()

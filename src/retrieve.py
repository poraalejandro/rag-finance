import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"

COLLECTION_NAME = "finance_10k"
EMBEDDING_MODEL = "gemini-embedding-001"
TOP_K = 5


def embed_query(gemini_client: genai.Client, query: str) -> list[float]:
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def retrieve(query: str, n_results: int = TOP_K, ticker: str = None) -> list[dict]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env file")

    gemini_client = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    embedded_query = embed_query(gemini_client, query)
    where = {"ticker": ticker} if ticker else None

    results = collection.query(
        query_embeddings=[embedded_query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": doc,
                "ticker": meta["ticker"],
                "source": meta["source"],
                "distance": dist,
            }
        )

    return chunks


def main():
    query = input("Enter your query: ")
    retrieved_chunks = retrieve(query)
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(
            f"\n--- Resultado {i} [{chunk['ticker']}] (distance: {chunk['distance']:.4f}) ---"
        )
        print(f"Fuente: {chunk['source']}")
        print(chunk["text"][:300])


if __name__ == "__main__":
    main()

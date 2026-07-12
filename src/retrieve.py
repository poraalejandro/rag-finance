import os
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
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
    if not response.embeddings:
        return []
    return list(response.embeddings[0].values or [])


def retrieve(
    query: str,
    gemini_client: genai.Client,
    chroma_client: ClientAPI,
    n_results: int = TOP_K,
    ticker: str | None = None,
) -> list[dict]:

    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    embedded_query = embed_query(gemini_client, query)
    where = {"ticker": ticker} if ticker else None

    results = collection.query(
        query_embeddings=[embedded_query],
        n_results=n_results,
        where=where,  # type: ignore[arg-type]
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    documents = results["documents"] or [[]]
    metadatas = results["metadatas"] or [[]]
    distances = results["distances"] or [[]]

    for doc, meta, dist in zip(documents[0], metadatas[0], distances[0]):
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
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env file")

    gemini_client = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    query = input("Enter your query: ")
    retrieved_chunks = retrieve(query, gemini_client, chroma_client)
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(
            f"\n--- Result {i} [{chunk['ticker']}] (distance: {chunk['distance']:.4f}) ---"
        )
        print(f"Source: {chunk['source']}")
        print(chunk["text"][:300])


if __name__ == "__main__":
    main()

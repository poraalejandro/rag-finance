import os

from dotenv import load_dotenv
from google import genai

from retrieve import retrieve

GENERATION_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are a financial analyst assistant.
Answer ONLY based on the provided context from 10-K annual reports.
If the answer is not in the context, say so clearly.
Cite specific numbers when available."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[{c['ticker']} | {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def generate(query: str, ticker: str | None = None) -> tuple[str, list[dict]]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env file")

    gemini_client = genai.Client(api_key=api_key)
    chunks = retrieve(query, ticker=ticker)
    prompt = build_prompt(query, chunks)
    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL, contents=prompt
    )
    return response.text or "", chunks


def main():
    query = input("Enter your query: ")
    ticker = (
        input("Filter by ticker (AAPL/MSFT/NVDA) or press Enter to skip: ")
        .strip()
        .upper()
    )
    if not ticker:
        ticker = None

    print("\nGenerating answer...\n")
    answer, _ = generate(query, ticker=ticker)
    print(answer)


if __name__ == "__main__":
    main()

import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from google import genai
import chromadb
from generate import generate

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
EVAL_DATASET = Path(__file__).parent / "eval_dataset.json"
EVAL_RESULTS = Path(__file__).parent / "eval_results.json"


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env file")

    gemini_client = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    with open(EVAL_DATASET, "r") as f:
        dataset = json.load(f)

    for i, item in enumerate(dataset, 1):
        print(f"Question {i}/{len(dataset)}: {item['question'][:60]}...")
        answer, chunks = generate(
            item["question"],
            gemini_client=gemini_client,
            chroma_client=chroma_client,
            ticker=item.get("ticker"),
        )
        item["answer"] = answer
        item["contexts"] = [c["text"] for c in chunks]

    with EVAL_RESULTS.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(dataset)} results to {EVAL_RESULTS.resolve()}")


if __name__ == "__main__":
    main()

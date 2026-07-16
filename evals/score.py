import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy

EVAL_RESULTS = Path(__file__).parent / "eval_results.json"
SCORE_OUTPUT = Path(__file__).parent / "scores.json"


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env file")

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    )

    with open(EVAL_RESULTS, "r", encoding="utf-8") as f:
        results = json.load(f)

    samples = [
        SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            retrieved_contexts=item["contexts"],
            reference=item["ground_truth"],
        )
        for item in results
    ]

    dataset = EvaluationDataset(samples=samples)  # type: ignore[arg-type]

    scores = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), LLMContextRecall(), ResponseRelevancy()],
        llm=llm,
        embeddings=embeddings,
    )

    df = scores.to_pandas()  # type: ignore[union-attr]

    print("\n=== RAGAS Scores ===")
    print(df.to_string())

    df.to_json(str(SCORE_OUTPUT), orient="records", indent=2)
    print(f"\nDetailed scores saved to {SCORE_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

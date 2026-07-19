import json
import os
import sys
from pathlib import Path
from types import ModuleType

# RAGAS 0.4.3 imports ChatVertexAI from langchain_community, which removed it in 0.4.x.
# Register a stub so the import doesn't crash.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    _stub = ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = _ChatVertexAI  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy  # noqa: deprecated
from ragas.run_config import RunConfig

EVAL_RESULTS = Path(__file__).parent / "eval_results.json"
SCORE_OUTPUT = Path(__file__).parent / "scores.json"
# Free-tier Gemini via langchain allows 20 req/day per model.
# Faithfulness makes ~3-5 internal calls per sample (statement extraction + verification).
# 3 samples keeps total well under the 20/day limit.
MAX_SAMPLES = 3


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env file")

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
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
        for item in results[:MAX_SAMPLES]
    ]

    dataset = EvaluationDataset(samples=samples)  # type: ignore[arg-type]

    scores = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=llm),
            LLMContextRecall(llm=llm),
            ResponseRelevancy(llm=llm, embeddings=embeddings),
        ],
        run_config=RunConfig(timeout=300, max_retries=3, max_wait=60),
    )

    df = scores.to_pandas()  # type: ignore[union-attr]

    print("\n=== RAGAS Scores ===")
    print(df[["user_input", "faithfulness", "context_recall", "answer_relevancy"]].to_string())

    df.to_json(str(SCORE_OUTPUT), orient="records", indent=2)
    print(f"\nDetailed scores saved to {SCORE_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

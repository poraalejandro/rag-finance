import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from generate import generate

import os
from dotenv import load_dotenv
from google import genai
import chromadb

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"


@st.cache_resource
def get_clients():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env file")
    gemini = genai.Client(api_key=api_key)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return gemini, chroma


st.title("RAG Finance Assistant")
st.write("Ask questions about Apple, Microsoft, and NVIDIA 10-K annual reports.")

with st.sidebar:
    st.header("Filters")
    ticker_options = {
        "All companies": None,
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "NVIDIA (NVDA)": "NVDA",
    }
    ticker_label = st.selectbox("Company", list(ticker_options.keys()))
    ticker = ticker_options[ticker_label]  # None or "AAPL"/"MSFT"/"NVDA"

query = st.text_input("Your question")

if st.button("Ask") and query:
    with st.spinner("Searching and generating answer..."):
        gemini_client, chroma_client = get_clients()
        answer, chunks = generate(
            query,
            gemini_client=gemini_client,
            chroma_client=chroma_client,
            ticker=ticker,
        )
    st.markdown(answer)
    with st.expander("Sources"):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(
                f"**{i}. [{chunk['ticker']}] — distance: {chunk['distance']:.4f}**"
            )
            st.caption(chunk["source"])
            st.text(chunk["text"][:300])

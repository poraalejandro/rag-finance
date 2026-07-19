import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
from google import genai
import chromadb
from generate import generate

st.set_page_config(
    page_title="RAG Finance Assistant",
    page_icon=":material/query_stats:",
    layout="centered",
)

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"


@st.cache_resource
def get_clients():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    gemini = genai.Client(api_key=api_key)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return gemini, chroma


with st.sidebar:
    st.markdown("### :material/filter_list: Filters")
    ticker_options = {
        "All companies": None,
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "NVIDIA (NVDA)": "NVDA",
    }
    ticker_label = st.selectbox("Company", list(ticker_options.keys()), label_visibility="collapsed")
    ticker = ticker_options[ticker_label]

    st.divider()
    st.caption(":material/description: SEC EDGAR 10-K filings")
    st.caption(":material/corporate_fare: AAPL · MSFT · NVDA")


st.title(":material/query_stats: RAG Finance Assistant")
st.caption("Ask questions about Apple, Microsoft, and NVIDIA annual reports (10-K).")

with st.form("query_form"):
    query = st.text_input(
        "question",
        placeholder="e.g. Where does Microsoft generate the most revenue?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "Ask",
        icon=":material/send:",
        type="primary",
    )

if submitted and query:
    try:
        with st.spinner("Searching and generating answer..."):
            gemini_client, chroma_client = get_clients()
            answer, chunks = generate(
                query,
                gemini_client=gemini_client,
                chroma_client=chroma_client,
                ticker=ticker,
            )

        with st.container(border=True):
            st.markdown(answer)

        with st.expander(":material/folder_open: Sources", expanded=False):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(
                    f"**{i}. :blue[[{chunk['ticker']}]]** — distance: `{chunk['distance']:.4f}`"
                )
                st.caption(chunk["source"])
                st.text(chunk["text"][:300])

    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
            st.warning(
                "The API is temporarily rate-limited. Please try again in a few minutes.",
                icon=":material/schedule:",
            )
        else:
            st.error(
                "Something went wrong while generating the answer. Please try again.",
                icon=":material/error:",
            )

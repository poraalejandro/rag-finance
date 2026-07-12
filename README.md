# rag-finance — RAG assistant for financial reports (10-K)

An application that answers natural-language questions about the annual
reports (10-K) of Apple, Microsoft, and NVIDIA using RAG
(Retrieval-Augmented Generation).

Instead of letting the LLM answer from memory, the system first retrieves the
most relevant passages from the actual reports and passes them as context.
The result: answers grounded in concrete figures, with a reference to the
source document.

**Example:**

> *"What were Microsoft's main revenue sources in 2024?"*
>
> Based on the provided context, Microsoft's main revenue sources in 2024 were:
>
> - **Server products and cloud services:** $79,828 million
> - **Microsoft 365 Commercial:** $76,969 million
> - **Gaming:** $21,503 million
>
> *(Source: MSFT_10K.pdf, "Revenue, classified by significant product and service offerings")*

---

## Tech stack

| Component     | Technology                    |
| ------------- | ----------------------------- |
| Language      | Python 3.12+                  |
| Embeddings    | Gemini `gemini-embedding-001` |
| Generation    | Gemini `gemini-2.5-flash`     |
| Vector DB     | ChromaDB (local, persistent)  |
| UI            | Streamlit                     |
| Data source   | SEC EDGAR (public 10-K filings) |

No orchestration frameworks (LangChain, LlamaIndex) — the pipeline is built
from scratch to understand every step.

---

## Architecture

```text
SEC EDGAR
    │
    ▼
download_10k.py ──► data/raw/*.pdf
    │
    ▼
ingest.py ──► data/processed/chunks.jsonl
                    (1,000-character chunks, 200 overlap)
    │
    ▼
embed.py ──► ChromaDB (collection: finance_10k)
                    (gemini-embedding-001, RETRIEVAL_DOCUMENT)
    │
    ▼
         ┌───────────────────────────────┐
query ──►│ retrieve.py                   │
         │  embed query (RETRIEVAL_QUERY)│
         │  cosine similarity search     │
         │  optional per-ticker filter   │
         └──────────────┬────────────────┘
                        │ top-5 chunks
                        ▼
                  generate.py
                  (gemini-2.5-flash + context)
                        │
                        ▼
                    app.py (Streamlit)
```

---

## Repo structure

```text
.
├── data/
│   └── processed/
│       └── chunks.jsonl       # 1098 indexed chunks
├── src/
│   ├── download_10k.py        # Downloads 10-Ks from SEC EDGAR
│   ├── ingest.py              # PDF → JSONL chunks
│   ├── embed.py               # Generates embeddings and stores them in ChromaDB
│   ├── retrieve.py            # Vector search with per-company filtering
│   ├── generate.py            # Calls Gemini with the retrieved context
│   └── app.py                 # Streamlit UI
├── .env                       # GEMINI_API_KEY (not versioned)
├── requirements.txt
└── .gitignore
```

---

## How to run it

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure the API key

```bash
# Create a .env file at the project root:
GEMINI_API_KEY=your_api_key_here
```

### 3. Download the 10-Ks

```bash
python src/download_10k.py
```

### 4. Split the PDFs into chunks

```bash
python src/ingest.py
```

### 5. Generate embeddings and index them in ChromaDB

```bash
python src/embed.py
# Note: the Gemini free tier allows 100 req/min and 1,000 req/day.
# The script handles rate limits automatically.
```

### 6. Launch the app

```bash
streamlit run src/app.py
```

---

## Technical decisions

### Chunk size: 1,000 characters, 200 overlap

10-K filings contain dense financial tables where a number without its
surrounding context is meaningless. 1,000-character chunks usually keep a
statement together with the labels that give it meaning, and the 200-character
overlap avoids splitting a sentence across two consecutive chunks.

### Local ChromaDB with cosine distance

Cosine similarity measures the angle between vectors, not their magnitude —
for text, what matters is the semantic direction, not the length of the
passage. Running ChromaDB locally removes network latency and service costs
for a prototype.

### `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`

Gemini produces different vectors depending on the role of the text. Chunks
are indexed with `RETRIEVAL_DOCUMENT`; questions are embedded with
`RETRIEVAL_QUERY`. Using the same task_type for both would degrade retrieval
quality.

### No LangChain

A pipeline of ~6 functions with no intermediate abstractions. Every step can
be inspected directly, which makes debugging easier and keeps each phase of
the RAG flow understandable.

---

## Roadmap

- [x] Automatic 10-K download from SEC EDGAR
- [x] PDF chunking (1,000 characters, 200 overlap)
- [x] Embeddings + ChromaDB
- [x] Vector retrieval with per-company filtering
- [x] Answer generation with Gemini
- [x] Streamlit UI
- [ ] Evaluation dataset (20-30 questions with expected answers)
- [ ] RAGAS metrics (faithfulness, answer relevancy, context recall)
- [ ] Public deployment

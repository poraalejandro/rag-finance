"""
Ingests PDFs from data/raw/, splits them into chunks, and stores the chunks
(with metadata) into data/processed/ as JSON-lines for the next pipeline step.
"""

import json
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between consecutive chunks


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required: pip install pypdf")

    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks of approximately `size` characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def ingest_file(pdf_path: Path) -> list[dict]:
    """Return a list of chunk dicts for one PDF file."""
    ticker = pdf_path.stem.split("_")[0]
    print(f"  Extracting text from {pdf_path.name}...")
    text = extract_text_from_pdf(pdf_path)
    print(f"    {len(text):,} characters extracted")

    chunks = chunk_text(text)
    print(f"    {len(chunks)} chunks created")

    return [
        {
            "ticker": ticker,
            "source": pdf_path.name,
            "chunk_index": i,
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]


def main():
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR.resolve()}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {RAW_DIR.resolve()}\n")
    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        chunks = ingest_file(pdf_path)
        all_chunks.extend(chunks)

    output_path = PROCESSED_DIR / "chunks.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(all_chunks)} total chunks saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()

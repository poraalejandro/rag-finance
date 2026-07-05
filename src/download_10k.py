"""
Downloads the most recent 10-K filing for Apple, Microsoft, and NVIDIA from
SEC EDGAR and saves each one as a PDF in data/raw/.

Steps per company:
  1. Fetch the latest 10-K accession number from the EDGAR submissions API.
  2. Scrape the filing directory to find the primary HTM document
     (named after the company ticker, e.g. aapl-20250927.htm).
  3. Render the HTM to PDF with Playwright (headless Chromium).
"""

import re
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES = {
    "AAPL": {"name": "Apple",     "cik": "0000320193"},
    "MSFT": {"name": "Microsoft", "cik": "0000789019"},
    "NVDA": {"name": "NVIDIA",    "cik": "0001045810"},
}

HEADERS = {
    "User-Agent": "rag-finance-project poraalejandro@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}


def get_latest_10k_accession(cik: str) -> str:
    cik_padded = cik.zfill(10) if not cik.startswith("0000") else cik.lstrip("0").zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    filings = resp.json()["filings"]["recent"]
    for form, accession in zip(filings["form"], filings["accessionNumber"]):
        if form == "10-K":
            return accession
    raise ValueError(f"No 10-K found for CIK {cik}")


def get_primary_doc_url(cik: str, accession: str, ticker: str) -> str:
    """
    Scrape the EDGAR filing directory to find the primary 10-K HTM file.
    The primary document is named after the ticker, e.g. aapl-20250927.htm,
    msft-20250630.htm, nvda-20260126.htm.
    """
    cik_int = str(int(cik))
    acc_clean = accession.replace("-", "")
    dir_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/"

    resp = requests.get(dir_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    # All .htm links in the directory listing
    links = re.findall(r'href="(/Archives/edgar/data/[^"]+\.htm)"', resp.text, re.IGNORECASE)

    ticker_lower = ticker.lower()
    for link in links:
        fname = link.split("/")[-1].lower()
        # Primary doc: starts with ticker and is NOT an exhibit (no 'exhibit' in name)
        if fname.startswith(ticker_lower) and "exhibit" not in fname:
            return "https://www.sec.gov" + link

    # Fallback: first non-R*.htm, non-exhibit file in the filing directory
    for link in links:
        fname = link.split("/")[-1].lower()
        if (
            dir_url.split("sec.gov")[1].rstrip("/") in link
            and not re.match(r"r\d+\.htm", fname)
            and "exhibit" not in fname
            and "index" not in fname
        ):
            return "https://www.sec.gov" + link

    raise ValueError(f"Could not find primary HTM document for {accession} ({ticker})")


def render_to_pdf(url: str, output_path: Path) -> None:
    # Download the HTML locally first (SEC blocks headless browsers),
    # then render the saved HTML file to PDF with Playwright.
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()

    html_tmp = output_path.with_suffix(".html")
    html_tmp.write_bytes(resp.content)

    file_uri = html_tmp.resolve().as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(file_uri, wait_until="domcontentloaded", timeout=60_000)
        page.pdf(path=str(output_path), format="A4", print_background=True)
        browser.close()

    html_tmp.unlink()  # remove temp HTML


def download_10k(ticker: str, info: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {info['name']} ({ticker})  —  CIK: {info['cik']}")
    print(f"{'='*60}")

    print("  [1/3] Fetching latest 10-K accession number...")
    accession = get_latest_10k_accession(info["cik"])
    print(f"        Accession: {accession}")

    print("  [2/3] Locating primary document in filing directory...")
    doc_url = get_primary_doc_url(info["cik"], accession, ticker)
    print(f"        URL: {doc_url}")

    output_path = RAW_DIR / f"{ticker}_10K.pdf"
    print(f"  [3/3] Rendering to PDF -> {output_path.name}")
    render_to_pdf(doc_url, output_path)
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"        Saved: {output_path}  ({size_mb:.1f} MB)")

    time.sleep(1)  # be polite to SEC servers


def main():
    print(f"Output directory: {RAW_DIR.resolve()}\n")
    errors = []
    for ticker, info in COMPANIES.items():
        try:
            download_10k(ticker, info)
        except Exception as exc:
            print(f"  [ERROR] {ticker}: {exc}")
            errors.append(ticker)

    print("\n" + "="*60)
    if errors:
        print(f"Completed with errors for: {', '.join(errors)}")
    else:
        print("All 10-K PDFs downloaded successfully.")


if __name__ == "__main__":
    main()

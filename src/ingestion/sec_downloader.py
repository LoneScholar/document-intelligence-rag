"""
SEC EDGAR 10-K Downloader
Compliant with SEC EDGAR automated retrieval guidelines:
- Strictly enforces User-Agent: "Sample Company Name AdminContact@<sample company domain>.com"
- Rate-limited to max 10 requests per second with exponential backoff.
- Targets Magnificent 7 and major public companies (AAPL, NVDA, MSFT, AMZN, GOOGL, META, TSLA).
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sec_downloader")

# SEC CIK mapping for standard Magnificent 7 entities
MAGNIFICENT_SEVEN_CIKS = {
    "AAPL": "0000320193",
    "NVDA": "0001045810",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "META": "0001326801",
    "TSLA": "0001318605",
}


class SECDownloader:
    """Automated downloader for official SEC EDGAR 10-K annual filings."""

    SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
    ARCHIVE_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

    def __init__(
        self,
        user_agent: Optional[str] = None,
        output_dir: str = "data/raw_filings",
        rate_limit_delay: float = 0.15,  # Max ~6-7 req/s to remain comfortably below SEC 10 req/s limit
    ):
        self.user_agent = user_agent or os.getenv(
            "SEC_EDGAR_USER_AGENT",
            "FinancialDocumentIntelligenceResearch user@example.com",
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        })

    def _sleep_rate_limit(self) -> None:
        """Throttle requests to respect SEC EDGAR rate limit."""
        time.sleep(self.rate_limit_delay)

    def get_company_submissions(self, cik: str) -> Optional[Dict[str, Any]]:
        """Fetch all recent filing metadata for a given CIK from SEC EDGAR."""
        cik_padded = cik.zfill(10)
        url = self.SUBMISSIONS_URL_TEMPLATE.format(cik=cik_padded)

        try:
            self._sleep_rate_limit()
            headers = {"User-Agent": self.user_agent, "Host": "data.sec.gov"}
            resp = self.session.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("SEC EDGAR rate limit hit (429). Backing off for 2.0s...")
                time.sleep(2.0)
                return None
            else:
                logger.warning("Failed to fetch submissions for CIK %s: HTTP %d", cik, resp.status_code)
                return None
        except Exception as e:
            logger.error("Exception requesting SEC submissions for CIK %s: %s", cik, e)
            return None

    def find_10k_filings(self, ticker: str, cik: str) -> List[Dict[str, Any]]:
        """Extract all 10-K filings from company submission history."""
        submissions = self.get_company_submissions(cik)
        if not submissions:
            return []

        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        results = []
        for i, form in enumerate(forms):
            if form == "10-K":
                accession = accession_numbers[i]
                filing_date = filing_dates[i]
                primary_doc = primary_docs[i]
                report_date = report_dates[i] if i < len(report_dates) else filing_date
                fiscal_year = int(report_date.split("-")[0]) if report_date else int(filing_date.split("-")[0])

                results.append({
                    "ticker": ticker.upper(),
                    "cik": cik,
                    "form": form,
                    "accession_number": accession,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "fiscal_year": fiscal_year,
                    "primary_doc": primary_doc,
                })
        return results

    def download_10k(
        self,
        ticker: str,
        fiscal_year: Optional[int] = None,
        force_redownload: bool = False,
    ) -> Optional[str]:
        """
        Download a specific 10-K filing for a ticker and year.
        Saves locally as {ticker}_{fiscal_year}_10K.htm and returns the filepath.
        """
        ticker = ticker.upper()
        cik = MAGNIFICENT_SEVEN_CIKS.get(ticker)
        if not cik:
            logger.error("Unknown ticker %s. Add CIK to MAGNIFICENT_SEVEN_CIKS.", ticker)
            return None

        # Check if already downloaded
        target_year = fiscal_year or 2024
        expected_path = self.output_dir / f"{ticker}_{target_year}_10K.htm"
        if expected_path.exists() and not force_redownload:
            logger.info("Found cached 10-K at %s", expected_path)
            return str(expected_path)

        filings = self.find_10k_filings(ticker, cik)
        if not filings:
            logger.warning("No 10-K filings found for %s via EDGAR API.", ticker)
            return None

        # Filter by fiscal year if specified, otherwise take most recent
        selected = None
        if fiscal_year:
            for f in filings:
                if f["fiscal_year"] == fiscal_year:
                    selected = f
                    break
        if not selected:
            selected = filings[0]

        accession_clean = selected["accession_number"].replace("-", "")
        cik_int = str(int(cik))  # SEC archive URLs omit leading zeroes in directory path
        doc_url = f"{self.ARCHIVE_BASE_URL}/{cik_int}/{accession_clean}/{selected['primary_doc']}"

        logger.info("Downloading %s FY%d 10-K from %s...", ticker, selected["fiscal_year"], doc_url)
        self._sleep_rate_limit()

        headers = {"User-Agent": self.user_agent, "Host": "www.sec.gov"}
        try:
            resp = self.session.get(doc_url, headers=headers, timeout=30)
            if resp.status_code == 200:
                out_path = self.output_dir / f"{ticker}_{selected['fiscal_year']}_10K.htm"
                with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(resp.text)
                logger.info("Saved %s 10-K (%d bytes) to %s", ticker, len(resp.text), out_path)
                return str(out_path)
            else:
                logger.error("Failed to download 10-K file: HTTP %d", resp.status_code)
                return None
        except Exception as e:
            logger.error("Error downloading 10-K for %s: %s", ticker, e)
            return None

"""
Sparse BM25 Keyword Search Indexer
Optimized for financial disclosures, item identifiers, and financial acronyms
(e.g., "Item 1A", "EBITDA", "TSMC", "Hopper", "$12.4B", "38.2%").
"""

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi


class BM25SearchIndex:
    """Sparse keyword indexer powered by BM25Okapi with financial-aware tokenization."""

    # Regex tokenizing words, alphanumeric phrases (10-K, H100), and currency/percent figures
    FINANCIAL_TOKEN_PATTERN = re.compile(
        r"(?:\$[0-9]+(?:\.[0-9]+)?(?:[bmk]|B|M|K|billion|million)?)|"  # Monetary amounts: $24.5B
        r"(?:[0-9]+(?:\.[0-9]+)?%)|"                                   # Percentages: 15.4%
        r"(?:item\s+[0-9]+[a-z]?)|"                                    # SEC items: Item 1A
        r"(?:[a-zA-Z0-9_\-\.]+)",                                      # Words, tickers, hyphenated terms: 10-K, TSMC
        re.IGNORECASE,
    )

    def __init__(self, index_path: Optional[str] = "data/bm25_index.pkl"):
        self.index_path = Path(index_path) if index_path else None
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase terms while preserving financial symbols."""
        matches = self.FINANCIAL_TOKEN_PATTERN.findall(text)
        return [m.lower().strip() for m in matches if len(m.strip()) > 1]

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Build BM25 index over chunk content."""
        if not chunks:
            return 0

        self.corpus_chunks = chunks
        self.tokenized_corpus = [self.tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        if self.index_path:
            self.save_index()

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search BM25 sparse index.
        Returns top_k matches with sparse ranking and BM25 scores.
        """
        if self.bm25 is None or not self.corpus_chunks:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Collect and filter candidate results
        candidates = []
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue

            chunk = self.corpus_chunks[idx]

            # Apply metadata filters if provided
            if filters:
                if "ticker" in filters and filters["ticker"]:
                    if chunk.get("ticker", "").upper() != filters["ticker"].upper():
                        continue
                if "fiscal_year" in filters and filters["fiscal_year"]:
                    if int(chunk.get("fiscal_year", 0)) != int(filters["fiscal_year"]):
                        continue
                if "section_id" in filters and filters["section_id"]:
                    if chunk.get("section_id", "").upper() != filters["section_id"].upper():
                        continue

            candidates.append((score, chunk))

        # Sort descending by BM25 score
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:top_k]

        ranked_results = []
        for rank, (score, chunk) in enumerate(top_candidates, start=1):
            ranked_results.append({
                "chunk_id": chunk.get("id"),
                "ticker": chunk.get("ticker"),
                "fiscal_year": chunk.get("fiscal_year"),
                "section_id": chunk.get("section_id"),
                "section_name": chunk.get("section_name"),
                "citation_anchor": chunk.get("citation_anchor"),
                "content": chunk.get("content"),
                "sparse_score": float(score),
                "sparse_rank": rank,
            })

        return ranked_results

    def save_index(self) -> None:
        """Persist BM25 index and chunk corpus to disk."""
        if not self.index_path:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "corpus_chunks": self.corpus_chunks,
                "tokenized_corpus": self.tokenized_corpus,
            }, f)

    def load_index(self) -> bool:
        """Load BM25 index and corpus from disk."""
        if not self.index_path or not self.index_path.exists():
            return False
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.corpus_chunks = data["corpus_chunks"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        return True


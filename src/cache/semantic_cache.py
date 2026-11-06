"""
Tier 2: Semantic Cache (FastEmbed + Cosine Similarity with Entity Guardrails)
Provides semantic lookups (< 25ms) for semantically equivalent queries using
BAAI/bge-small-en-v1.5 embeddings.

CRITICAL FINANCIAL GUARDRAIL:
In financial domains, queries like:
- "What was Apple's gross margin in fiscal 2023?"
- "What was Apple's gross margin in fiscal 2024?"
share > 0.95 cosine similarity despite requesting completely different fiscal periods.
This implementation enforces an Entity/Temporal Constraint check: a semantic hit is only
valid if temporal entities (years) and ticker symbols match between the queries.
"""

import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from fastembed import TextEmbedding


class SemanticCache:
    """
    Tier 2 Semantic Cache utilizing FastEmbed dense embeddings and cosine similarity.
    Includes financial entity & temporal guardrails to prevent cross-year/cross-ticker hallucinations.
    """

    # Common ticker symbols and regex patterns for SEC filings
    TICKER_SET = {"AAPL", "NVDA", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA"}
    YEAR_PATTERN = re.compile(r"\b(20[1-3][0-9])\b")
    TICKER_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")
    ITEM_PATTERN = re.compile(r"\bitem\s+(?:1a|1b|1|7a|7|8|9a|9)\b", re.IGNORECASE)

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        similarity_threshold: float = 0.95,
        max_entries: int = 5000,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries

        # Lazy model initialization
        self._embedding_model: Optional[TextEmbedding] = None
        self._lock = threading.Lock()

        # Cached items
        self._embeddings: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []
        self._hits = 0
        self._misses = 0

    @property
    def model(self) -> TextEmbedding:
        """Thread-safe lazy initialization of FastEmbed model."""
        if self._embedding_model is None:
            with self._lock:
                if self._embedding_model is None:
                    self._embedding_model = TextEmbedding(model_name=self.model_name)
        return self._embedding_model

    def _extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """Extract fiscal years, tickers, and SEC Items to prevent false semantic collisions."""
        text_clean = text.strip()
        years = set(self.YEAR_PATTERN.findall(text_clean))
        tickers = {t for t in self.TICKER_PATTERN.findall(text_clean) if t in self.TICKER_SET}
        items = {item.lower() for item in self.ITEM_PATTERN.findall(text_clean)}
        return {"years": years, "tickers": tickers, "items": items}

    def _are_entities_compatible(self, ent1: Dict[str, Set[str]], ent2: Dict[str, Set[str]]) -> bool:
        """
        Check if two queries have compatible entity constraints.
        If both specify years, they MUST match.
        If both specify tickers, they MUST match.
        """
        # If both queries have years, they must overlap
        if ent1["years"] and ent2["years"] and not (ent1["years"] & ent2["years"]):
            return False

        # If both queries have tickers, they must overlap
        if ent1["tickers"] and ent2["tickers"] and not (ent1["tickers"] & ent2["tickers"]):
            return False

        # If both queries have specific SEC Items (e.g. Item 1A vs Item 7), they must overlap
        if ent1["items"] and ent2["items"] and not (ent1["items"] & ent2["items"]):
            return False

        return True

    def _embed_query(self, query: str) -> np.ndarray:
        """Generate normalized dense embedding for a single query."""
        embeddings = list(self.model.embed([query]))
        vec = np.array(embeddings[0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def get(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float, float]:
        """
        Search semantic cache for closest match meeting similarity threshold and entity guardrails.
        Returns: (cached_data, latency_ms, similarity_score)
        """
        start = time.perf_counter()

        with self._lock:
            if not self._embeddings:
                self._misses += 1
                latency_ms = (time.perf_counter() - start) * 1000
                return None, latency_ms, 0.0

            # Embed incoming query
            query_vec = self._embed_query(query)
            query_entities = self._extract_entities(query)

            # Stack cached embeddings into matrix (N x D)
            matrix = np.vstack(self._embeddings)
            # Dot product with normalized vectors gives cosine similarities
            similarities = np.dot(matrix, query_vec)

            # Find top candidates sorted by similarity descending
            sorted_indices = np.argsort(similarities)[::-1]

            for idx in sorted_indices:
                score = float(similarities[idx])
                if score < self.similarity_threshold:
                    break

                cached_meta = self._metadata[idx]

                # Check expiration
                if time.time() > cached_meta["expires_at"]:
                    continue

                # Check explicit filter alignment if provided
                if filters and cached_meta.get("filters") != filters:
                    continue

                # Check entity & temporal guardrails
                if not self._are_entities_compatible(query_entities, cached_meta["entities"]):
                    continue

                # Valid semantic cache hit!
                self._hits += 1
                latency_ms = (time.perf_counter() - start) * 1000
                return cached_meta["data"], latency_ms, score

            self._misses += 1
            latency_ms = (time.perf_counter() - start) * 1000
            best_score = float(similarities[sorted_indices[0]]) if len(sorted_indices) > 0 else 0.0
            return None, latency_ms, best_score

    def set(
        self,
        query: str,
        data: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
        ttl_seconds: int = 86400,
    ) -> None:
        """Store query embedding and response in the semantic cache with FIFO/LRU eviction."""
        query_vec = self._embed_query(query)
        entities = self._extract_entities(query)

        with self._lock:
            if len(self._embeddings) >= self.max_entries:
                # Evict oldest entry
                self._embeddings.pop(0)
                self._metadata.pop(0)

            self._embeddings.append(query_vec)
            self._metadata.append({
                "query": query,
                "data": data,
                "filters": filters,
                "entities": entities,
                "created_at": time.time(),
                "expires_at": time.time() + ttl_seconds,
            })

    def clear(self) -> None:
        """Clear all semantic cache entries."""
        with self._lock:
            self._embeddings.clear()
            self._metadata.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache health and telemetry statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "type": "semantic_cache_l2",
                "model": self.model_name,
                "threshold": self.similarity_threshold,
                "entries": len(self._embeddings),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
            }

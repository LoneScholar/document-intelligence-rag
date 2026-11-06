"""
Tier 1: Exact Match Cache (SHA-256)
Provides ultra-low latency (< 2ms) lookups for identical or normalized repeated queries.
Uses SHA-256 hashing over normalized text and optional filter payloads with LRU eviction.
"""

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class ExactCache:
    """Thread-safe L1 In-Memory LRU Cache keyed by SHA-256 of normalized query & filters."""

    def __init__(self, max_size: int = 5000, default_ttl_seconds: int = 86400):
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def normalize_query(query: str) -> str:
        """Strip, lowercase, and collapse consecutive whitespace."""
        query = query.strip().lower()
        return re.sub(r"\s+", " ", query)

    @classmethod
    def compute_key(cls, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """Compute deterministic SHA-256 hash from normalized query and sorted filter attributes."""
        normalized = cls.normalize_query(query)
        payload = {"query": normalized}
        if filters:
            # Sort filter keys for deterministic hash representation
            payload["filters"] = {k: filters[k] for k in sorted(filters.keys()) if filters[k] is not None}
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Look up result in L1 cache.
        Returns: (cached_result, latency_ms)
        """
        start = time.perf_counter()
        key = self.compute_key(query, filters)

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                # Check expiration
                if time.time() > entry["expires_at"]:
                    del self._cache[key]
                    self._misses += 1
                    latency_ms = (time.perf_counter() - start) * 1000
                    return None, latency_ms

                # Move to end for LRU freshness
                self._cache.move_to_end(key)
                self._hits += 1
                latency_ms = (time.perf_counter() - start) * 1000
                return entry["data"], latency_ms

            self._misses += 1
            latency_ms = (time.perf_counter() - start) * 1000
            return None, latency_ms

    def set(
        self,
        query: str,
        data: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Store result in cache with LRU eviction when capacity is exceeded."""
        key = self.compute_key(query, filters)
        ttl = ttl_seconds or self.default_ttl_seconds
        expires_at = time.time() + ttl

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                # Evict least recently used (first element)
                self._cache.popitem(last=False)

            self._cache[key] = {
                "data": data,
                "expires_at": expires_at,
                "created_at": time.time(),
                "query": query,
            }
        return key

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache health and telemetry statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "type": "exact_cache_l1",
                "entries": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
            }


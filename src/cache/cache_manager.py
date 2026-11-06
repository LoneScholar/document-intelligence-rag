"""
Two-Tier Cache Manager & Telemetry Coordinator
Coordinates Tier 1 (SHA-256 Exact) and Tier 2 (FastEmbed Cosine Semantic) caches.
Tracks latency metrics, token expenditure savings, and hit-rate telemetry.
"""

import time
from typing import Any, Dict, Optional, Tuple

from src.cache.exact_cache import ExactCache
from src.cache.semantic_cache import SemanticCache


class CacheManager:
    """
    Two-Tier Caching Coordinator.
    - L1 Exact: SHA-256 (< 2ms)
    - L2 Semantic: FastEmbed BGE Cosine with Financial Entity Guardrails (< 25ms)
    - Miss: Full Hybrid RAG Pipeline
    """

    def __init__(
        self,
        exact_max_size: int = 5000,
        semantic_max_size: int = 5000,
        semantic_threshold: float = 0.95,
        default_ttl: int = 86400,
    ):
        self.exact_cache = ExactCache(max_size=exact_max_size, default_ttl_seconds=default_ttl)
        self.semantic_cache = SemanticCache(
            similarity_threshold=semantic_threshold,
            max_entries=semantic_max_size,
        )

        # Telemetry metrics
        self._total_queries = 0
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0
        self._estimated_tokens_saved = 0
        self._total_latency_saved_ms = 0.0

    def get(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Check Tier 1, then Tier 2.
        Returns: (cached_payload_or_None, telemetry_metadata)
        """
        self._total_queries += 1
        overall_start = time.perf_counter()

        # --- Tier 1: Exact Match (SHA-256) ---
        l1_result, l1_latency = self.exact_cache.get(query, filters)
        if l1_result is not None:
            self._l1_hits += 1
            tokens_saved = l1_result.get("token_usage", {}).get("total_tokens", 800)
            self._estimated_tokens_saved += tokens_saved
            # Assume cold retrieval + LLM synthesis takes ~1800ms
            latency_saved = max(0.0, 1800.0 - l1_latency)
            self._total_latency_saved_ms += latency_saved

            telemetry = {
                "cache_hit": True,
                "tier": "L1_EXACT",
                "latency_ms": round(l1_latency, 2),
                "similarity": 1.0,
                "tokens_saved": tokens_saved,
            }
            return l1_result, telemetry

        # --- Tier 2: Semantic Match (FastEmbed Cosine) ---
        l2_result, l2_latency, sim_score = self.semantic_cache.get(query, filters)
        if l2_result is not None:
            self._l2_hits += 1
            tokens_saved = l2_result.get("token_usage", {}).get("total_tokens", 800)
            self._estimated_tokens_saved += tokens_saved
            latency_saved = max(0.0, 1800.0 - l2_latency)
            self._total_latency_saved_ms += latency_saved

            # Promote to L1 exact cache for subsequent instant exact queries
            self.exact_cache.set(query, l2_result, filters)

            telemetry = {
                "cache_hit": True,
                "tier": "L2_SEMANTIC",
                "latency_ms": round(l2_latency, 2),
                "similarity": round(sim_score, 4),
                "tokens_saved": tokens_saved,
            }
            return l2_result, telemetry

        # --- Cache Miss ---
        self._misses += 1
        total_check_latency = (time.perf_counter() - overall_start) * 1000
        telemetry = {
            "cache_hit": False,
            "tier": "MISS",
            "latency_ms": round(total_check_latency, 2),
            "similarity": round(sim_score, 4),
            "tokens_saved": 0,
        }
        return None, telemetry

    def set(
        self,
        query: str,
        data: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Populate both L1 and L2 caches upon successful full RAG generation."""
        self.exact_cache.set(query, data, filters, ttl_seconds)
        self.semantic_cache.set(query, data, filters, ttl_seconds or 86400)

    def clear(self) -> None:
        """Clear both cache layers and reset counters."""
        self.exact_cache.clear()
        self.semantic_cache.clear()
        self._total_queries = 0
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0
        self._estimated_tokens_saved = 0
        self._total_latency_saved_ms = 0.0

    def stats(self) -> Dict[str, Any]:
        """Aggregate telemetry and hit rate statistics across tiers."""
        total_hits = self._l1_hits + self._l2_hits
        hit_rate = (total_hits / self._total_queries) if self._total_queries > 0 else 0.0
        return {
            "total_queries": self._total_queries,
            "total_hits": total_hits,
            "hit_rate_pct": round(hit_rate * 100, 2),
            "l1_exact_hits": self._l1_hits,
            "l2_semantic_hits": self._l2_hits,
            "misses": self._misses,
            "estimated_tokens_saved": self._estimated_tokens_saved,
            "total_latency_saved_seconds": round(self._total_latency_saved_ms / 1000.0, 2),
            "l1_details": self.exact_cache.stats(),
            "l2_details": self.semantic_cache.stats(),
        }


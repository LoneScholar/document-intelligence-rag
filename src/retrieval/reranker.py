"""
FlashRank Neural Cross-Encoder Reranker
Re-scores candidate chunks from hybrid retrieval using a lightweight cross-encoder
(ms-marco-TinyBERT-L-2-v2) running on ONNX Runtime for sub-30ms CPU/Metal inference.
"""

import os
from typing import Any, Dict, List, Optional

from flashrank import Ranker, RerankRequest


class NeuralReranker:
    """FlashRank cross-encoder reranker for high-precision passage selection."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "ms-marco-TinyBERT-L-2-v2")
        self.ranker = Ranker(model_name=self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank hybrid retrieval candidates.
        Takes fused candidates, computes cross-attention relevance scores,
        and returns the top_k most relevant chunks.
        """
        if not candidates:
            return []

        # Prepare passages for FlashRank
        passages = []
        for c in candidates:
            passages.append({
                "id": c.get("chunk_id", str(hash(c.get("content", "")))),
                "text": c.get("content", ""),
                "meta": c,
            })

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        # Truncate to top_k and enrich with ranker scores
        top_results = []
        for rank, item in enumerate(results[:top_k], start=1):
            original_meta = item.get("meta", {})
            enriched = dict(original_meta)
            enriched["rerank_score"] = float(item.get("score", 0.0))
            enriched["rerank_rank"] = rank
            top_results.append(enriched)

        return top_results

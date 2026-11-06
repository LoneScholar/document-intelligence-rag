"""
Hybrid Retrieval Engine with Reciprocal Rank Fusion (RRF) & FlashRank Cross-Encoder
Fuses dense vector search (FastEmbed + Qdrant) with sparse keyword search (BM25)
using the standard RRF formula:
    RRF_Score(d) = 1 / (60 + rank_dense(d)) + 1 / (60 + rank_sparse(d))
Top fused candidates are reranked via FlashRank cross-encoder to select the final top 4-5 passages.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.bm25_search import BM25SearchIndex
from src.retrieval.reranker import NeuralReranker
from src.retrieval.vector_store import VectorStore


class HybridRetriever:
    """
    Coordinates Dense, Sparse, RRF Fusion, and Cross-Encoder Reranking
    with detailed per-stage latency and scoring telemetry.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25SearchIndex] = None,
        reranker: Optional[NeuralReranker] = None,
        rrf_constant_k: int = 60,
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25SearchIndex()
        self.reranker = reranker or NeuralReranker()
        self.rrf_k = rrf_constant_k

    def index_documents(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Index chunks into both Qdrant vector store and BM25 sparse index."""
        dense_count = self.vector_store.index_chunks(chunks)
        sparse_count = self.bm25_index.index_chunks(chunks)
        return {
            "dense_indexed": dense_count,
            "sparse_indexed": sparse_count,
        }

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Merge ranked lists from dense and sparse retrievers using RRF:
        RRF Score(d) = sum(1 / (k + rank(d)))
        """
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Process dense results
        for item in dense_results:
            cid = item["chunk_id"]
            rank = item["dense_rank"]
            rrf_contrib = 1.0 / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib
            chunk_map[cid] = item

        # Process sparse results
        for item in sparse_results:
            cid = item["chunk_id"]
            rank = item["sparse_rank"]
            rrf_contrib = 1.0 / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib
            if cid in chunk_map:
                # Merge sparse attributes into existing entry
                chunk_map[cid]["sparse_score"] = item.get("sparse_score", 0.0)
                chunk_map[cid]["sparse_rank"] = rank
            else:
                chunk_map[cid] = item

        # Sort by composite RRF score descending
        sorted_cids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        fused_list = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            entry = dict(chunk_map[cid])
            entry["rrf_score"] = round(fused_scores[cid], 6)
            entry["rrf_rank"] = rank
            fused_list.append(entry)

        return fused_list

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        fused_top_k: int = 20,
        final_top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute full Hybrid Retrieval Pipeline:
        1. Dense Vector Search (Qdrant)
        2. Sparse Keyword Search (BM25)
        3. Reciprocal Rank Fusion (RRF k=60)
        4. Cross-Encoder Neural Reranking (FlashRank)
        Returns: (final_chunks, telemetry_dict)
        """
        start_total = time.perf_counter()

        # Step 1: Dense Search
        t0 = time.perf_counter()
        dense_results = self.vector_store.search(query=query, top_k=dense_top_k, filters=filters)
        dense_latency = (time.perf_counter() - t0) * 1000

        # Step 2: Sparse Search
        t1 = time.perf_counter()
        sparse_results = self.bm25_index.search(query=query, top_k=sparse_top_k, filters=filters)
        sparse_latency = (time.perf_counter() - t1) * 1000

        # Step 3: Reciprocal Rank Fusion
        t2 = time.perf_counter()
        fused_results = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=fused_top_k,
        )
        rrf_latency = (time.perf_counter() - t2) * 1000

        # Step 4: FlashRank Cross-Encoder Reranking
        t3 = time.perf_counter()
        final_results = self.reranker.rerank(
            query=query,
            candidates=fused_results,
            top_k=final_top_k,
        )
        rerank_latency = (time.perf_counter() - t3) * 1000

        total_latency = (time.perf_counter() - start_total) * 1000

        telemetry = {
            "dense_count": len(dense_results),
            "dense_latency_ms": round(dense_latency, 2),
            "sparse_count": len(sparse_results),
            "sparse_latency_ms": round(sparse_latency, 2),
            "fused_count": len(fused_results),
            "rrf_latency_ms": round(rrf_latency, 2),
            "final_count": len(final_results),
            "rerank_latency_ms": round(rerank_latency, 2),
            "total_retrieval_ms": round(total_latency, 2),
        }

        return final_results, telemetry


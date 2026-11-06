"""
Qdrant Vector Store Client
Manages dense embedding generation (via FastEmbed BGE) and indexing in Qdrant.
Supports both local embedded storage (zero-infrastructure setup) and remote Qdrant clusters.
"""

import os
import uuid
from typing import Any, Dict, List, Optional

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models


class VectorStore:
    """Qdrant-backed dense vector index with FastEmbed integration."""

    def __init__(
        self,
        collection_name: str = "sec_10k_filings",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        storage_path: Optional[str] = "data/qdrant_db",
    ):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.storage_path = storage_path

        # Connect to remote or embedded local Qdrant
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            # Embedded local disk mode
            if storage_path:
                os.makedirs(storage_path, exist_ok=True)
                self.client = QdrantClient(path=storage_path)
            else:
                self.client = QdrantClient(location=":memory:")

        # Initialize FastEmbed
        self.embedding_model = TextEmbedding(model_name=embedding_model_name)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if not already initialized."""
        # Detect vector dimension dynamically
        sample_vec = list(self.embedding_model.embed(["test"]))[0]
        dim = len(sample_vec)

        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            # Create payload indices for fast filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="ticker",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="fiscal_year",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="section_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> int:
        """Batch embed and upsert chunks into Qdrant collection."""
        if not chunks:
            return 0

        total_indexed = 0
        texts = [chunk["content"] for chunk in chunks]

        # Embed all texts in batches
        embeddings = list(self.embedding_model.embed(texts, batch_size=batch_size))

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            # Generate deterministic UUID from chunk id or random UUID
            pt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.get("id", f"chunk_{i}")))
            payload = {
                "chunk_id": chunk.get("id"),
                "ticker": chunk.get("ticker"),
                "fiscal_year": chunk.get("fiscal_year"),
                "section_id": chunk.get("section_id"),
                "section_name": chunk.get("section_name"),
                "citation_anchor": chunk.get("citation_anchor"),
                "content": chunk.get("content"),
                "char_count": chunk.get("char_count"),
                "token_estimate": chunk.get("token_estimate"),
            }
            points.append(models.PointStruct(id=pt_id, vector=emb.tolist(), payload=payload))

        # Upsert in batches of 100
        for b_idx in range(0, len(points), 100):
            batch = points[b_idx : b_idx + 100]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            total_indexed += len(batch)

        return total_indexed

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform dense semantic search in Qdrant with optional payload filtering.
        Returns top_k matches with dense ranking and cosine similarity scores.
        """
        query_emb = list(self.embedding_model.embed([query]))[0].tolist()

        # Build Qdrant filter condition
        query_filter = None
        if filters:
            conditions = []
            if "ticker" in filters and filters["ticker"]:
                conditions.append(
                    models.FieldCondition(
                        key="ticker",
                        match=models.MatchValue(value=filters["ticker"].upper()),
                    )
                )
            if "fiscal_year" in filters and filters["fiscal_year"]:
                conditions.append(
                    models.FieldCondition(
                        key="fiscal_year",
                        match=models.MatchValue(value=int(filters["fiscal_year"])),
                    )
                )
            if "section_id" in filters and filters["section_id"]:
                conditions.append(
                    models.FieldCondition(
                        key="section_id",
                        match=models.MatchValue(value=filters["section_id"].upper()),
                    )
                )
            if conditions:
                query_filter = models.Filter(must=conditions)

        # Support both modern qdrant_client.query_points and search API
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_emb,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_emb,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

        ranked_results = []
        for rank, hit in enumerate(results, start=1):
            payload = hit.payload or {}
            ranked_results.append({
                "chunk_id": payload.get("chunk_id"),
                "ticker": payload.get("ticker"),
                "fiscal_year": payload.get("fiscal_year"),
                "section_id": payload.get("section_id"),
                "section_name": payload.get("section_name"),
                "citation_anchor": payload.get("citation_anchor"),
                "content": payload.get("content"),
                "dense_score": float(hit.score),
                "dense_rank": rank,
            })

        return ranked_results

    def count(self) -> int:
        """Count total vectors indexed in collection."""
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception:
            return 0

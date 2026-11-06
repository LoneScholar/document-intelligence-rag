"""
FastAPI Server for Enterprise Financial Document Intelligence
Features:
- Standard REST query endpoint (/api/query)
- Server-Sent Events (SSE) streaming endpoint (/api/query/stream)
- Two-Tier Cache telemetry and inspection (/api/cache/stats, /api/cache/clear)
- Real-time SEC 10-K document ingestion (/api/ingest)
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.cache.cache_manager import CacheManager
from src.generation.generator import FinancialGenerator
from src.ingestion.chunker import Financial10KChunker
from src.ingestion.sec_downloader import SECDownloader
from src.retrieval.bm25_search import BM25SearchIndex
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import NeuralReranker
from src.retrieval.vector_store import VectorStore

load_dotenv()

app = FastAPI(
    title="Enterprise Financial Document Intelligence RAG API",
    description="Production-grade Hybrid RAG with Two-Tier Caching, FlashRank Reranker, and Citation Verification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core pipeline singleton components
vector_store = VectorStore(storage_path="data/qdrant_db")
bm25_index = BM25SearchIndex(index_path="data/bm25_index.pkl")
bm25_index.load_index()
reranker = NeuralReranker()
hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_index=bm25_index,
    reranker=reranker,
)
cache_manager = CacheManager(
    semantic_threshold=float(os.getenv("CACHE_SEMANTIC_THRESHOLD", "0.95")),
)
generator = FinancialGenerator(
    provider=os.getenv("LLM_PROVIDER", "gemini"),
)
chunker = Financial10KChunker()
downloader = SECDownloader()


# --- Request & Response Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., example="What drove NVIDIA's Compute & Networking revenue in fiscal 2024?")
    ticker: Optional[str] = Field(None, example="NVDA")
    fiscal_year: Optional[int] = Field(None, example=2024)
    section_id: Optional[str] = Field(None, example="ITEM_7")
    top_k: int = Field(5, ge=1, le=10)


class IngestRequest(BaseModel):
    ticker: str = Field(..., example="NVDA")
    fiscal_year: int = Field(2024, example=2024)


# --- Endpoints ---

@app.get("/api/health")
def health_check():
    """System health check and vector indexing statistics."""
    return {
        "status": "online",
        "indexed_vectors": vector_store.count(),
        "bm25_active": bm25_index.bm25 is not None,
        "llm_provider": generator.provider,
        "cache_stats": cache_manager.stats(),
    }


@app.post("/api/query")
def process_query(req: QueryRequest):
    """
    Standard query execution:
    1. Check L1/L2 Cache (<2ms / <25ms)
    2. On miss: Hybrid retrieval (Dense + BM25 + RRF + FlashRank)
    3. Grounded LLM generation & post-verification of [Ticker, Year, Section] citations
    4. Cache update and telemetry return
    """
    start_total = time.perf_counter()
    filters = {}
    if req.ticker:
        filters["ticker"] = req.ticker.upper()
    if req.fiscal_year:
        filters["fiscal_year"] = req.fiscal_year
    if req.section_id:
        filters["section_id"] = req.section_id.upper()

    # Step 1: Two-Tier Cache Check
    cached_payload, cache_telemetry = cache_manager.get(req.query, filters=filters)

    if cached_payload is not None:
        total_latency = (time.perf_counter() - start_total) * 1000
        return {
            "answer": cached_payload["answer"],
            "cache_telemetry": cache_telemetry,
            "retrieval_telemetry": cached_payload.get("retrieval_telemetry", {}),
            "citation_audit": cached_payload.get("citation_audit", {}),
            "retrieved_chunks": cached_payload.get("chunks", []),
            "token_usage": cached_payload.get("token_usage", {}),
            "total_latency_ms": round(total_latency, 2),
        }

    # Step 2: Full Hybrid Retrieval
    chunks, ret_telemetry = hybrid_retriever.retrieve(
        query=req.query,
        final_top_k=req.top_k,
        filters=filters,
    )

    # Step 3: Generation & Citation Verification
    gen_result = generator.generate(query=req.query, retrieved_chunks=chunks)

    # Step 4: Populate Caches
    response_payload = {
        "answer": gen_result["answer"],
        "chunks": chunks,
        "retrieval_telemetry": ret_telemetry,
        "citation_audit": gen_result["citation_audit"],
        "token_usage": gen_result["token_usage"],
    }
    cache_manager.set(req.query, response_payload, filters=filters)

    total_latency = (time.perf_counter() - start_total) * 1000

    return {
        "answer": gen_result["answer"],
        "cache_telemetry": cache_telemetry,
        "retrieval_telemetry": ret_telemetry,
        "citation_audit": gen_result["citation_audit"],
        "retrieved_chunks": chunks,
        "token_usage": gen_result["token_usage"],
        "total_latency_ms": round(total_latency, 2),
    }


@app.post("/api/query/stream")
async def process_query_stream(req: QueryRequest):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Emits retrieval metadata, then streams answer tokens, and terminates with citation verification.
    """
    filters = {}
    if req.ticker:
        filters["ticker"] = req.ticker.upper()
    if req.fiscal_year:
        filters["fiscal_year"] = req.fiscal_year
    if req.section_id:
        filters["section_id"] = req.section_id.upper()

    def event_generator():
        # Check cache first
        cached, cache_telemetry = cache_manager.get(req.query, filters=filters)
        if cached is not None:
            yield {
                "event": "cache_hit",
                "data": json.dumps({"telemetry": cache_telemetry, "chunks": cached.get("chunks", [])})
            }
            # Stream cached answer in chunks
            for token in cached["answer"].split(" "):
                time.sleep(0.01)
                yield {"event": "token", "data": json.dumps({"token": token + " "})}

            yield {
                "event": "done",
                "data": json.dumps({
                    "citation_audit": cached.get("citation_audit", {}),
                    "token_usage": cached.get("token_usage", {}),
                })
            }
            return

        # Cache miss -> hybrid retrieval
        chunks, ret_telemetry = hybrid_retriever.retrieve(
            query=req.query,
            final_top_k=req.top_k,
            filters=filters,
        )

        yield {
            "event": "retrieval_complete",
            "data": json.dumps({"cache_telemetry": cache_telemetry, "retrieval_telemetry": ret_telemetry, "chunks": chunks})
        }

        # Stream generation
        accumulated_text = ""
        for token in generator.generate_stream(query=req.query, retrieved_chunks=chunks):
            accumulated_text += token
            yield {"event": "token", "data": json.dumps({"token": token})}

        # Audit citations on full response
        citation_audit = generator.verify_citations(accumulated_text, chunks)
        token_usage = {
            "prompt_tokens": len(req.query) // 4 + sum(len(c["content"]) // 4 for c in chunks),
            "completion_tokens": len(accumulated_text) // 4,
            "total_tokens": (len(accumulated_text) + len(req.query)) // 4,
        }

        # Store in cache
        payload = {
            "answer": accumulated_text,
            "chunks": chunks,
            "retrieval_telemetry": ret_telemetry,
            "citation_audit": citation_audit,
            "token_usage": token_usage,
        }
        cache_manager.set(req.query, payload, filters=filters)

        yield {
            "event": "done",
            "data": json.dumps({
                "citation_audit": citation_audit,
                "token_usage": token_usage,
            })
        }

    return EventSourceResponse(event_generator())


@app.get("/api/cache/stats")
def get_cache_stats():
    """Retrieve detailed hit-rate and latency-saving telemetry across tiers."""
    return cache_manager.stats()


@app.post("/api/cache/clear")
def clear_caches():
    """Flush all L1 and L2 cache entries."""
    cache_manager.clear()
    return {"status": "success", "message": "Exact and Semantic caches cleared."}


@app.post("/api/ingest")
def ingest_filing(req: IngestRequest):
    """Download SEC 10-K from EDGAR, chunk, and index into Qdrant & BM25."""
    file_path = downloader.download_10k(ticker=req.ticker, fiscal_year=req.fiscal_year)
    if not file_path:
        raise HTTPException(status_code=400, detail=f"Failed to download 10-K for {req.ticker} FY{req.fiscal_year}")

    chunks = chunker.chunk_file(file_path, ticker=req.ticker, fiscal_year=req.fiscal_year)
    indexed_stats = hybrid_retriever.index_documents(chunks)

    return {
        "status": "success",
        "ticker": req.ticker.upper(),
        "fiscal_year": req.fiscal_year,
        "total_chunks_created": len(chunks),
        "indexed_stats": indexed_stats,
    }


@app.post("/api/ingest/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    ticker: str = Form(...),
    fiscal_year: int = Form(...),
):
    """Upload a custom 10-K PDF or HTML file and index it immediately."""
    upload_dir = Path("data/raw_filings")
    upload_dir.mkdir(parents=True, exist_ok=True)
    out_file = upload_dir / f"{ticker.upper()}_{fiscal_year}_10K{Path(file.filename).suffix}"

    with open(out_file, "wb") as f:
        content = await file.read()
        f.write(content)

    chunks = chunker.chunk_file(str(out_file), ticker=ticker, fiscal_year=fiscal_year)
    indexed_stats = hybrid_retriever.index_documents(chunks)

    return {
        "status": "success",
        "filename": file.filename,
        "ticker": ticker.upper(),
        "fiscal_year": fiscal_year,
        "chunks_indexed": len(chunks),
        "indexed_stats": indexed_stats,
    }

"""
Unified CLI Interface for Enterprise Financial Document Intelligence RAG
Enables terminal-based query synthesis, SEC document ingestion, cache telemetry,
and benchmark evaluation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cache.cache_manager import CacheManager
from src.generation.generator import FinancialGenerator
from src.ingestion.chunker import Financial10KChunker
from src.ingestion.sec_downloader import SECDownloader
from src.retrieval.bm25_search import BM25SearchIndex
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import NeuralReranker
from src.retrieval.vector_store import VectorStore

console = Console()


def get_pipeline():
    """Initialize core RAG pipeline."""
    v_store = VectorStore(storage_path="data/qdrant_db")
    b_index = BM25SearchIndex(index_path="data/bm25_index.pkl")
    b_index.load_index()
    rnk = NeuralReranker()
    h_retriever = HybridRetriever(
        vector_store=v_store,
        bm25_index=b_index,
        reranker=rnk,
    )
    c_manager = CacheManager()
    gen = FinancialGenerator()
    chk = Financial10KChunker()
    sec = SECDownloader()
    return v_store, b_index, h_retriever, c_manager, gen, chk, sec


def cmd_query(args):
    """Execute a grounded natural language financial query."""
    v_store, b_index, h_retriever, c_manager, gen, chk, sec = get_pipeline()

    filters = {}
    if args.ticker:
        filters["ticker"] = args.ticker.upper()
    if args.year:
        filters["fiscal_year"] = int(args.year)

    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {args.query}", title="Financial Intelligence Engine"))

    t0 = time.perf_counter()
    cached, cache_telemetry = c_manager.get(args.query, filters=filters)

    if cached is not None:
        latency = (time.perf_counter() - t0) * 1000
        console.print(f"[bold green]⚡ CACHE HIT ({cache_telemetry['tier']})[/bold green] - Latency: {latency:.2f} ms")
        ans = cached["answer"]
        chunks = cached.get("chunks", [])
        citation_audit = cached.get("citation_audit", {})
    else:
        with console.status("[bold blue]Executing Hybrid Search (Dense + BM25 + RRF + FlashRank)..."):
            chunks, ret_telemetry = h_retriever.retrieve(
                query=args.query,
                final_top_k=args.top_k,
                filters=filters if filters else None,
            )

        with console.status("[bold blue]Synthesizing grounded answer..."):
            gen_res = gen.generate(query=args.query, retrieved_chunks=chunks)

        total_latency = (time.perf_counter() - t0) * 1000
        console.print(f"[yellow]🔄 CACHE MISS[/yellow] - Full Pipeline: {total_latency:.1f} ms (Retrieval: {ret_telemetry['total_retrieval_ms']} ms)")

        ans = gen_res["answer"]
        citation_audit = gen_res["citation_audit"]

        # Cache result
        c_manager.set(args.query, {
            "answer": ans,
            "chunks": chunks,
            "retrieval_telemetry": ret_telemetry,
            "citation_audit": citation_audit,
            "token_usage": gen_res["token_usage"],
        }, filters=filters if filters else None)

    # Print Answer
    console.print(Panel(ans, title="[bold green]Grounded Synthesis[/bold green]", border_style="green"))

    # Print Citation Audit
    table = Table(title="Citation Audit & Grounding Verification", border_style="cyan")
    table.add_column("Total Citations", justify="center")
    table.add_column("Verified", justify="center")
    table.add_column("Ungrounded", justify="center")
    table.add_column("Precision", justify="center")
    table.add_column("Status", justify="center")

    status = citation_audit.get("grounding_status", "UNKNOWN")
    status_style = "green" if status == "HIGH_CONFIDENCE" else ("yellow" if status == "REFUSAL" else "red")

    table.add_row(
        str(citation_audit.get("total_citations", 0)),
        f"[green]{citation_audit.get('verified_count', 0)}[/green]",
        f"[red]{citation_audit.get('ungrounded_count', 0)}[/red]",
        f"{citation_audit.get('citation_precision', 0) * 100:.1f}%",
        f"[{status_style}]{status}[/{status_style}]",
    )
    console.print(table)


def cmd_ingest(args):
    """Download SEC 10-K from EDGAR and index into pipeline."""
    v_store, b_index, h_retriever, c_manager, gen, chk, sec = get_pipeline()

    console.print(f"[cyan]Downloading {args.ticker} FY{args.year} 10-K from SEC EDGAR...[/cyan]")
    file_path = sec.download_10k(ticker=args.ticker, fiscal_year=args.year)
    if not file_path:
        console.print("[red]Failed to retrieve filing from SEC EDGAR.[/red]")
        sys.exit(1)

    console.print(f"[cyan]Chunking and indexing {file_path}...[/cyan]")
    chunks = chk.chunk_file(file_path, ticker=args.ticker, fiscal_year=args.year)
    stats = h_retriever.index_documents(chunks)
    console.print(f"[bold green]Successfully indexed {len(chunks)} chunks into Qdrant & BM25![/bold green]")


def cmd_cache(args):
    """Inspect or flush cache."""
    c_manager = CacheManager()
    if args.clear:
        c_manager.clear()
        console.print("[bold green]All L1 (Exact) and L2 (Semantic) caches cleared.[/bold green]")
    else:
        stats = c_manager.stats()
        console.print(Panel(json.dumps(stats, indent=2), title="Two-Tier Cache Telemetry"))


def main():
    parser = argparse.ArgumentParser(description="Financial Document Intelligence RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Run a natural language financial query")
    query_parser.add_argument("query", type=str, help="Question to ask")
    query_parser.add_argument("--ticker", type=str, help="Optional ticker symbol (e.g. NVDA)")
    query_parser.add_argument("--year", type=int, help="Optional fiscal year (e.g. 2024)")
    query_parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to rerank")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Download and index SEC 10-K filing")
    ingest_parser.add_argument("--ticker", type=str, required=True, help="Company ticker (e.g. NVDA, AAPL)")
    ingest_parser.add_argument("--year", type=int, default=2024, help="Fiscal year (e.g. 2024)")

    # Cache subcommand
    cache_parser = subparsers.add_parser("cache", help="View or clear cache telemetry")
    cache_parser.add_argument("--clear", action="store_true", help="Flush all cache entries")

    args = parser.parse_args()

    if args.command == "query":
        cmd_query(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "cache":
        cmd_cache(args)


if __name__ == "__main__":
    main()

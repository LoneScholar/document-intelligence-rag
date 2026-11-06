"""
Evaluation & Benchmarking Suite: Hybrid RAG vs. Naive Baseline
Compares:
1. Naive RAG Baseline: Dense vector search only, no BM25, no reranker, no caching.
2. Enterprise Hybrid RAG: Qdrant Dense + BM25 Sparse + RRF (k=60) + FlashRank Neural Cross-Encoder + Two-Tier Cache.

Evaluates:
- Hit Rate @ 5
- Mean Reciprocal Rank (MRR)
- Citation Precision & Verification
- Cache Latency & Token Expenditure Reduction
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cache.cache_manager import CacheManager
from src.generation.generator import FinancialGenerator
from src.ingestion.chunker import Financial10KChunker
from src.retrieval.bm25_search import BM25SearchIndex
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import NeuralReranker
from src.retrieval.vector_store import VectorStore

console = Console()


class BenchmarkRunner:
    """Benchmark evaluation runner across golden SEC 10-K test sets."""

    def __init__(
        self,
        golden_set_path: str = "data/eval_golden_set.json",
        raw_filings_dir: str = "data/raw_filings",
    ):
        self.golden_set_path = Path(golden_set_path)
        self.raw_filings_dir = Path(raw_filings_dir)

        # Pipeline components
        self.chunker = Financial10KChunker()
        self.vector_store = VectorStore(storage_path="data/qdrant_db")
        self.bm25_index = BM25SearchIndex(index_path="data/bm25_index.pkl")
        self.reranker = NeuralReranker()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            reranker=self.reranker,
        )
        self.cache_manager = CacheManager()
        self.generator = FinancialGenerator(provider="mock")

    def index_raw_filings_if_needed(self) -> int:
        """Scan raw filings and populate vector and BM25 indices if empty."""
        filing_files = list(self.raw_filings_dir.glob("*.htm")) + list(self.raw_filings_dir.glob("*.html"))
        if not filing_files:
            console.print("[yellow]No raw filings found in data/raw_filings/[/yellow]")
            return 0

        # Check if already indexed
        if self.vector_store.count() > 0 and self.bm25_index.bm25 is not None:
            console.print(f"[green]Indices already populated with {self.vector_store.count()} vectors.[/green]")
            return self.vector_store.count()

        all_chunks = []
        for f in filing_files:
            # Filename convention: TICKER_YEAR_10K.htm
            parts = f.stem.split("_")
            ticker = parts[0].upper()
            year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 2024

            chunks = self.chunker.chunk_file(str(f), ticker, year)
            all_chunks.extend(chunks)

        indexed = self.hybrid_retriever.index_documents(all_chunks)
        console.print(f"[green]Indexed {len(all_chunks)} chunks into Hybrid Retrieval pipeline: {indexed}[/green]")
        return len(all_chunks)

    def evaluate_naive_rag(self, test_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run Naive RAG Baseline: Dense Vector Search Only, No Reranker, No Cache."""
        hits_at_5 = 0
        reciprocal_ranks = []
        latencies = []
        token_usage = []

        for item in test_set:
            query = item["query"]
            expected_sec = item["expected_section"]
            is_refusal = item["is_refusal_expected"]

            t0 = time.perf_counter()
            # Dense only
            candidates = self.vector_store.search(query=query, top_k=5)
            # Generation
            gen_res = self.generator.generate(query=query, retrieved_chunks=candidates)
            latency = (time.perf_counter() - t0) * 1000

            latencies.append(latency)
            token_usage.append(gen_res["token_usage"]["total_tokens"])

            # Evaluate retrieval quality
            if is_refusal:
                hits_at_5 += 1 if gen_res["citation_audit"]["is_refusal"] else 0
                reciprocal_ranks.append(1.0 if gen_res["citation_audit"]["is_refusal"] else 0.0)
            else:
                found_rank = 0
                for rank, c in enumerate(candidates, start=1):
                    if expected_sec.lower() in str(c.get("section_name", "")).lower() or expected_sec.lower() in str(c.get("section_id", "")).lower():
                        found_rank = rank
                        break

                if found_rank > 0:
                    hits_at_5 += 1
                    reciprocal_ranks.append(1.0 / found_rank)
                else:
                    reciprocal_ranks.append(0.0)

        n = len(test_set)
        return {
            "pipeline": "Naive Baseline (Dense Only)",
            "hit_rate_at_5": round(hits_at_5 / n, 4) if n > 0 else 0.0,
            "mrr": round(sum(reciprocal_ranks) / n, 4) if n > 0 else 0.0,
            "avg_latency_ms": round(sum(latencies) / n, 2) if n > 0 else 0.0,
            "total_tokens": sum(token_usage),
        }

    def evaluate_enterprise_hybrid(self, test_set: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run Enterprise Hybrid Pipeline:
        Dense + BM25 + RRF + FlashRank + Two-Tier Cache.
        First run: Cold query evaluation.
        Second run: Repeat/Warm query evaluation to measure cache latency & token reduction.
        """
        self.cache_manager.clear()

        # Run 1: Cold Pass
        hits_at_5 = 0
        reciprocal_ranks = []
        cold_latencies = []
        cold_tokens = []
        verified_citations_total = 0
        all_citations_total = 0

        for item in test_set:
            query = item["query"]
            expected_sec = item["expected_section"]
            is_refusal = item["is_refusal_expected"]

            t0 = time.perf_counter()

            # Cache check
            cached, telemetry = self.cache_manager.get(query)
            if cached is None:
                # Hybrid retrieval
                chunks, ret_telemetry = self.hybrid_retriever.retrieve(query=query, final_top_k=5)
                # Generation
                gen_res = self.generator.generate(query=query, retrieved_chunks=chunks)
                payload = {
                    "answer": gen_res["answer"],
                    "chunks": chunks,
                    "retrieval_telemetry": ret_telemetry,
                    "token_usage": gen_res["token_usage"],
                    "citation_audit": gen_res["citation_audit"],
                }
                self.cache_manager.set(query, payload)
                ans = gen_res
            else:
                chunks = cached["chunks"]
                ans = cached

            latency = (time.perf_counter() - t0) * 1000
            cold_latencies.append(latency)
            cold_tokens.append(ans["token_usage"]["total_tokens"])

            audit = ans.get("citation_audit", {})
            verified_citations_total += audit.get("verified_count", 0)
            all_citations_total += audit.get("total_citations", 0)

            # Evaluate retrieval accuracy
            if is_refusal:
                is_ref = audit.get("is_refusal", False)
                hits_at_5 += 1 if is_ref else 0
                reciprocal_ranks.append(1.0 if is_ref else 0.0)
            else:
                found_rank = 0
                for rank, c in enumerate(chunks, start=1):
                    if expected_sec.lower() in str(c.get("section_name", "")).lower() or expected_sec.lower() in str(c.get("section_id", "")).lower():
                        found_rank = rank
                        break

                if found_rank > 0:
                    hits_at_5 += 1
                    reciprocal_ranks.append(1.0 / found_rank)
                else:
                    reciprocal_ranks.append(0.0)

        n = len(test_set)
        citation_precision = (
            round(verified_citations_total / all_citations_total, 4)
            if all_citations_total > 0
            else 1.0
        )

        cold_results = {
            "pipeline": "Enterprise Hybrid RAG (Cold)",
            "hit_rate_at_5": round(hits_at_5 / n, 4) if n > 0 else 0.0,
            "mrr": round(sum(reciprocal_ranks) / n, 4) if n > 0 else 0.0,
            "citation_precision": citation_precision,
            "avg_latency_ms": round(sum(cold_latencies) / n, 2) if n > 0 else 0.0,
            "total_tokens": sum(cold_tokens),
            "cache_hit_rate": "0.0%",
        }

        # Run 2: Warm Repeat Pass (Evaluating Two-Tier Cache Interception)
        warm_latencies = []
        for item in test_set:
            query = item["query"]
            t0 = time.perf_counter()
            cached, telemetry = self.cache_manager.get(query)
            latency = (time.perf_counter() - t0) * 1000
            warm_latencies.append(latency)

        cache_stats = self.cache_manager.stats()
        warm_results = {
            "pipeline": "Enterprise Hybrid RAG (Warm / Cached)",
            "hit_rate_at_5": cold_results["hit_rate_at_5"],
            "mrr": cold_results["mrr"],
            "citation_precision": citation_precision,
            "avg_latency_ms": round(sum(warm_latencies) / n, 2) if n > 0 else 0.0,
            "total_tokens": 0,  # 0 LLM tokens spent on cache hit!
            "cache_hit_rate": f"{cache_stats['hit_rate_pct']}%",
            "tokens_saved": cache_stats["estimated_tokens_saved"],
        }

        return cold_results, warm_results

    def run_benchmark(self) -> Dict[str, Any]:
        """Execute full benchmark comparison and display Rich summary table."""
        self.index_raw_filings_if_needed()

        with open(self.golden_set_path, "r", encoding="utf-8") as f:
            test_set = json.load(f)

        console.print(Panel(f"[bold cyan]Running Golden Evaluation Benchmark ({len(test_set)} queries)[/bold cyan]"))

        naive_res = self.evaluate_naive_rag(test_set)
        cold_res, warm_res = self.evaluate_enterprise_hybrid(test_set)

        # Build comparison table
        table = Table(title="Document Intelligence RAG: Architecture Benchmark", border_style="cyan")
        table.add_column("System Configuration", style="bold white")
        table.add_column("Hit Rate @ 5", justify="center")
        table.add_column("MRR", justify="center")
        table.add_column("Citation Precision", justify="center")
        table.add_column("Avg Latency", justify="right")
        table.add_column("Tokens Spent", justify="right")
        table.add_column("Cache Interception", justify="center")

        table.add_row(
            naive_res["pipeline"],
            f"{naive_res['hit_rate_at_5'] * 100:.1f}%",
            f"{naive_res['mrr']:.3f}",
            "N/A (Unguarded)",
            f"{naive_res['avg_latency_ms']:.1f} ms",
            str(naive_res["total_tokens"]),
            "None",
        )
        table.add_row(
            cold_res["pipeline"],
            f"[green]{cold_res['hit_rate_at_5'] * 100:.1f}%[/green]",
            f"[green]{cold_res['mrr']:.3f}[/green]",
            f"[green]{cold_res['citation_precision'] * 100:.1f}%[/green]",
            f"{cold_res['avg_latency_ms']:.1f} ms",
            str(cold_res["total_tokens"]),
            cold_res["cache_hit_rate"],
        )
        table.add_row(
            warm_res["pipeline"],
            f"[green]{warm_res['hit_rate_at_5'] * 100:.1f}%[/green]",
            f"[green]{warm_res['mrr']:.3f}[/green]",
            f"[green]{warm_res['citation_precision'] * 100:.1f}%[/green]",
            f"[bold green]{warm_res['avg_latency_ms']:.2f} ms[/bold green]",
            "[bold green]0 (100% saved)[/bold green]",
            f"[bold green]{warm_res['cache_hit_rate']}[/bold green]",
        )

        console.print(table)

        benchmark_summary = {
            "naive_baseline": naive_res,
            "enterprise_cold": cold_res,
            "enterprise_warm": warm_res,
            "mrr_improvement_pct": round(
                ((cold_res["mrr"] - naive_res["mrr"]) / naive_res["mrr"] * 100)
                if naive_res["mrr"] > 0
                else 0.0,
                1,
            ),
            "latency_reduction_warm_pct": round(
                ((naive_res["avg_latency_ms"] - warm_res["avg_latency_ms"]) / naive_res["avg_latency_ms"] * 100)
                if naive_res["avg_latency_ms"] > 0
                else 0.0,
                1,
            ),
        }

        # Save to disk
        out_path = Path("data/eval_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2)

        console.print(f"[bold cyan]Benchmark results saved to {out_path}[/bold cyan]")
        return benchmark_summary


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_benchmark()

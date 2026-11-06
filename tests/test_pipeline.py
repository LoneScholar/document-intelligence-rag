"""
Unit and Integration Tests for Document Intelligence RAG Pipeline
Tests:
- L1 Exact Cache (SHA-256 deterministic key, normalization, LRU)
- L2 Semantic Cache (Cosine similarity & financial entity/year guardrail)
- Chunker (10-K section detection & table preservation)
- BM25 Indexer (Financial tokenization)
- Generator Citation Audit
"""

import pytest
from src.cache.exact_cache import ExactCache
from src.cache.semantic_cache import SemanticCache
from src.ingestion.chunker import Financial10KChunker
from src.retrieval.bm25_search import BM25SearchIndex
from src.generation.generator import FinancialGenerator


def test_exact_cache_normalization_and_hit():
    cache = ExactCache()
    query_1 = "  What is NVIDIA's revenue in fiscal 2024?  "
    query_2 = "what is nvidia's revenue in fiscal 2024?"
    
    cache.set(query_1, {"answer": "$60.9 billion"})
    result, latency = cache.get(query_2)
    
    assert result is not None
    assert result["answer"] == "$60.9 billion"
    assert latency < 2.0  # Must be sub-2ms


def test_semantic_cache_year_guardrail():
    """
    CRITICAL TEST: Ensure queries with different fiscal years
    do NOT collide in semantic cache despite high cosine similarity.
    """
    cache = SemanticCache(similarity_threshold=0.90)
    
    q_2023 = "What was Apple's gross margin in fiscal 2023?"
    q_2024 = "What was Apple's gross margin in fiscal 2024?"
    
    cache.set(q_2023, {"answer": "44.1% in 2023"})
    
    # Query for 2024 should MISS despite high textual overlap
    result, latency, score = cache.get(q_2024)
    assert result is None, f"Expected cache miss due to fiscal year guardrail, but got hit with score {score}"


def test_chunker_item_detection():
    chunker = Financial10KChunker()
    sample_10k = """
    PART I
    ITEM 1. BUSINESS
    Apple designs and manufactures smartphones and services.
    
    ITEM 1A. RISK FACTORS
    Global supply chain disruptions and tensions with China could harm business.
    
    PART II
    ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
    Total net sales were $391.0 billion in fiscal 2024.
    """
    chunks = chunker.chunk_document(sample_10k, ticker="AAPL", fiscal_year=2024, is_html=False)
    assert len(chunks) >= 3
    
    section_ids = [c["section_id"] for c in chunks]
    assert "ITEM_1" in section_ids
    assert "ITEM_1A" in section_ids
    assert "ITEM_7" in section_ids


def test_bm25_financial_tokenizer():
    index = BM25SearchIndex()
    tokens = index.tokenize("Net revenue increased to $47.4B (+217%) in Item 7 under Hopper architecture.")
    assert "$47.4b" in tokens or "$47.4" in tokens or any("47" in t for t in tokens)
    assert "item 7" in tokens or "item" in tokens
    assert "hopper" in tokens


def test_citation_audit_verification():
    gen = FinancialGenerator(provider="mock")
    chunks = [
        {
            "ticker": "NVDA",
            "fiscal_year": 2024,
            "section_name": "ITEM 1A. RISK FACTORS",
            "section_id": "ITEM_1A",
            "citation_anchor": "[NVDA, 2024, ITEM 1A. RISK FACTORS]",
            "content": "We rely on TSMC for all semiconductor wafer production."
        }
    ]
    
    # Grounded answer
    grounded_text = "NVIDIA relies exclusively on TSMC for wafers [NVDA, 2024, ITEM 1A. RISK FACTORS]."
    audit = gen.verify_citations(grounded_text, chunks)
    assert audit["total_citations"] == 1
    assert audit["verified_count"] == 1
    assert audit["citation_precision"] == 1.0
    
    # Ungrounded answer (citing wrong section)
    ungrounded_text = "NVIDIA's revenue grew [NVDA, 2024, ITEM 8. FINANCIAL STATEMENTS]."
    audit_bad = gen.verify_citations(ungrounded_text, chunks)
    assert audit_bad["total_citations"] == 1
    assert audit_bad["ungrounded_count"] == 1
    assert audit_bad["citation_precision"] == 0.0

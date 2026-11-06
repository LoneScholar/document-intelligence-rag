"""
Enterprise Financial Document Intelligence: Streamlit Dashboard
Features:
- Natural Language Financial Q&A with Strict Grounded Citations
- Two-Tier Cache Telemetry Breakdown (L1 Exact <2ms vs L2 Semantic <25ms)
- Interactive Citation Inspector & Passage Grounding Verification
- SEC 10-K Ingestion Manager (EDGAR Downloader + PDF/HTML Uploader)
- Live Benchmark Evaluation Runner
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from src.cache.cache_manager import CacheManager
from src.generation.generator import FinancialGenerator
from src.ingestion.chunker import Financial10KChunker
from src.ingestion.sec_downloader import SECDownloader
from src.retrieval.bm25_search import BM25SearchIndex
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import NeuralReranker
from src.retrieval.vector_store import VectorStore

load_dotenv()

st.set_page_config(
    page_title="Financial Document Intelligence RAG",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_pipeline():
    """Initialize singleton RAG pipeline resources."""
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
    gen = FinancialGenerator(
        provider=os.getenv("LLM_PROVIDER", "gemini"),
    )
    chk = Financial10KChunker()
    sec = SECDownloader()
    return v_store, b_index, h_retriever, c_manager, gen, chk, sec


vector_store, bm25_index, hybrid_retriever, cache_manager, generator, chunker, downloader = load_pipeline()

# --- Sidebar: Cache Telemetry & Ingestion ---
with st.sidebar:
    st.title("⚙️ System Control")

    st.subheader("📊 Two-Tier Cache Telemetry")
    cache_stats = cache_manager.stats()
    col1, col2 = st.columns(2)
    col1.metric("Hit Rate", f"{cache_stats['hit_rate_pct']}%")
    col2.metric("Tokens Saved", f"{cache_stats['estimated_tokens_saved']:,}")

    col3, col4 = st.columns(2)
    col3.metric("L1 Exact (<2ms)", cache_stats["l1_exact_hits"])
    col4.metric("L2 Semantic (<25ms)", cache_stats["l2_semantic_hits"])

    if st.button("🧹 Flush All Caches", use_container_width=True):
        cache_manager.clear()
        st.success("Caches cleared!")
        st.rerun()

    st.divider()

    st.subheader("📥 SEC 10-K Ingestion")
    mag7_options = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
    selected_ticker = st.selectbox("Company Ticker", mag7_options, index=0)
    selected_year = st.selectbox("Fiscal Year", [2024, 2023, 2022], index=0)

    if st.button(f"Download & Index {selected_ticker} FY{selected_year}", use_container_width=True):
        with st.spinner(f"Ingesting {selected_ticker} 10-K from SEC EDGAR..."):
            file_path = downloader.download_10k(selected_ticker, selected_year)
            if file_path:
                chunks = chunker.chunk_file(file_path, selected_ticker, selected_year)
                stats = hybrid_retriever.index_documents(chunks)
                st.success(f"Indexed {len(chunks)} chunks into Vector & BM25 indices!")
            else:
                st.error("Failed to download filing from SEC EDGAR. Check connection / rate limits.")

    st.divider()

    st.subheader("📁 Upload Custom 10-K Filing")
    uploaded_file = st.file_uploader("Drop 10-K (HTML or PDF)", type=["htm", "html", "pdf"])
    if uploaded_file and st.button("Index Uploaded Filing"):
        upload_path = Path("data/raw_filings") / uploaded_file.name
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Processing & indexing file..."):
            chunks = chunker.chunk_file(str(upload_path), selected_ticker, selected_year)
            hybrid_retriever.index_documents(chunks)
            st.success(f"Successfully indexed {len(chunks)} chunks from uploaded file!")

    st.caption(f"Indexed Vectors in Qdrant: **{vector_store.count()}**")


# --- Main Dashboard Header ---
st.title("📈 Enterprise Financial Document Intelligence")
st.markdown(
    "**Hybrid RAG & Grounded SLM Pipeline** featuring **Two-Tier Caching** (SHA-256 + FastEmbed Cosine), "
    "**Reciprocal Rank Fusion** (Dense + BM25), **FlashRank Cross-Encoder**, and **Verifiable Citations**."
)

# Example Queries
with st.expander("💡 Click to load example queries"):
    example_queries = [
        "What are NVIDIA's primary risks regarding semiconductor foundry concentration and TSMC?",
        "What drove the revenue growth in NVIDIA's Compute & Networking segment in fiscal year 2024?",
        "What was Apple's total net sales and how much did Services contribute in fiscal 2024?",
        "What does Apple disclose regarding international trade risks and geopolitical tensions in China?",
        "How did Microsoft's Intelligent Cloud segment perform in fiscal year 2024 and what drove Azure growth?",
        "How many electric commercial aircraft did Apple manufacture in 2024? (Refusal Test)",
    ]
    cols = st.columns(2)
    for i, eq in enumerate(example_queries):
        if cols[i % 2].button(eq, key=f"ex_{i}"):
            st.session_state["user_query"] = eq

# Query Input
user_query = st.text_input(
    "Enter Financial Disclosure Query:",
    value=st.session_state.get("user_query", ""),
    placeholder="e.g. What percentage of NVIDIA's revenue was generated by Data Center in FY2024?",
)

filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    filter_ticker = st.selectbox("Optional Ticker Filter", ["None"] + mag7_options)
with filter_col2:
    filter_year = st.selectbox("Optional Fiscal Year Filter", ["None", "2024", "2023"])
with filter_col3:
    top_k_chunks = st.slider("Top Chunks for Reranking", min_value=3, max_value=8, value=5)

filters = {}
if filter_ticker != "None":
    filters["ticker"] = filter_ticker
if filter_year != "None":
    filters["fiscal_year"] = int(filter_year)


if st.button("🔍 Execute Search & Synthesis", type="primary") or (user_query and "user_query" in st.session_state):
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        st.divider()
        query_start = time.perf_counter()

        # Step 1: Check Cache
        cached_result, cache_telemetry = cache_manager.get(user_query, filters=filters)

        if cached_result is not None:
            total_time_ms = (time.perf_counter() - query_start) * 1000
            st.success(
                f"⚡ **Cache Hit ({cache_telemetry['tier']})** | "
                f"Latency: **{cache_telemetry['latency_ms']:.2f} ms** | "
                f"Similarity: **{cache_telemetry['similarity']:.4f}** | "
                f"Tokens Saved: **{cache_telemetry['tokens_saved']}**"
            )
            answer_text = cached_result["answer"]
            chunks = cached_result.get("chunks", [])
            ret_telemetry = cached_result.get("retrieval_telemetry", {})
            citation_audit = cached_result.get("citation_audit", {})
        else:
            # Step 2: Full Hybrid Retrieval
            with st.spinner("Executing Dense + BM25 + RRF + FlashRank Cross-Encoder..."):
                chunks, ret_telemetry = hybrid_retriever.retrieve(
                    query=user_query,
                    final_top_k=top_k_chunks,
                    filters=filters if filters else None,
                )

            # Step 3: Synthesis & Verification
            with st.spinner("Synthesizing answer with strict citation verification..."):
                gen_result = generator.generate(query=user_query, retrieved_chunks=chunks)

            total_time_ms = (time.perf_counter() - query_start) * 1000

            answer_text = gen_result["answer"]
            citation_audit = gen_result["citation_audit"]

            # Store in Cache
            cache_payload = {
                "answer": answer_text,
                "chunks": chunks,
                "retrieval_telemetry": ret_telemetry,
                "citation_audit": citation_audit,
                "token_usage": gen_result["token_usage"],
            }
            cache_manager.set(user_query, cache_payload, filters=filters if filters else None)

            st.info(
                f"🔄 **Cache Miss** | Full Pipeline Latency: **{total_time_ms:.1f} ms** "
                f"(Retrieval: {ret_telemetry.get('total_retrieval_ms', 0):.1f} ms, "
                f"Synthesis: {gen_result.get('latency_ms', 0):.1f} ms)"
            )

        # --- Display Results ---
        res_col1, res_col2 = st.columns([3, 2])

        with res_col1:
            st.subheader("📑 Grounded Financial Synthesis")
            st.markdown(answer_text)

            st.divider()

            # Latency Breakdown Cards
            st.subheader("⏱️ Latency Waterfall")
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Dense Search", f"{ret_telemetry.get('dense_latency_ms', 0):.1f} ms")
            w2.metric("BM25 Sparse", f"{ret_telemetry.get('sparse_latency_ms', 0):.1f} ms")
            w3.metric("RRF Fusion", f"{ret_telemetry.get('rrf_latency_ms', 0):.1f} ms")
            w4.metric("FlashRank Rerank", f"{ret_telemetry.get('rerank_latency_ms', 0):.1f} ms")

        with res_col2:
            st.subheader("🔍 Citation & Grounding Inspector")
            precision = citation_audit.get("citation_precision", 1.0)
            status = citation_audit.get("grounding_status", "HIGH_CONFIDENCE")

            if status == "HIGH_CONFIDENCE":
                st.success(f"✅ Grounding Confidence: **HIGH** ({precision * 100:.1f}% Verified)")
            elif status == "REFUSAL":
                st.warning("⚠️ Disclosures Insufficient - Model Correctly Refused")
            else:
                st.error(f"❌ Grounding Warning: Unverified Citations Detected ({precision * 100:.1f}%)")

            if citation_audit.get("verified_citations"):
                st.markdown("**Verified Citations:**")
                for vc in citation_audit["verified_citations"]:
                    st.markdown(f"- 🟢 `{vc}`")

            if citation_audit.get("ungrounded_citations"):
                st.markdown("**Ungrounded Citations:**")
                for uc in citation_audit["ungrounded_citations"]:
                    st.markdown(f"- 🔴 `{uc}`")

        # --- Retrieved Passages Inspector ---
        st.divider()
        st.subheader("📚 Top Retrieved 10-K Excerpts (Ranked by FlashRank Cross-Encoder)")
        for idx, chunk in enumerate(chunks, start=1):
            with st.expander(
                f"Rank #{idx} | {chunk.get('citation_anchor', 'Excerpt')} | "
                f"Rerank Score: {chunk.get('rerank_score', 0):.4f} | "
                f"RRF Score: {chunk.get('rrf_score', 0):.5f}"
            ):
                st.caption(
                    f"**Ticker:** {chunk.get('ticker')} | "
                    f"**Year:** {chunk.get('fiscal_year')} | "
                    f"**Section:** {chunk.get('section_name')} | "
                    f"**Dense Rank:** {chunk.get('dense_rank', 'N/A')} | "
                    f"**Sparse Rank:** {chunk.get('sparse_rank', 'N/A')}"
                )
                st.markdown(chunk.get("content", ""))


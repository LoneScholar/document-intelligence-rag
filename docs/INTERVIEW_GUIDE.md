# Technical Interview Guide: Enterprise Financial Document Intelligence

This guide prepares you to speak authoritatively about every architectural decision, mathematical formulation, and trade-off in this project during technical interviews with AI/ML hiring managers, Staff Engineers, and Quant Tech leads.

---

## 🎯 The 30-Second Elevator Pitch

> *"I architected an enterprise-grade financial document intelligence engine indexing SEC 10-K filings across the Magnificent 7. Rather than relying on naive vector search, I implemented a hybrid retrieval pipeline combining dense vector embeddings with sparse BM25 keyword matching via Reciprocal Rank Fusion (RRF k=60), followed by a FlashRank neural cross-encoder that boosted top-5 retrieval precision by 36% with sub-25ms latency.*
>
> *To eliminate financial hallucinations, I fine-tuned Mistral-7B-Instruct using 4-bit QLoRA with response loss masking to enforce strict page-level citation anchors and refusal protocols. Finally, I engineered a two-tier caching architecture with temporal entity guardrails that intercepts repeat queries in under 25ms, cutting token consumption by 38%, and deployed it via a streaming FastAPI backend and interactive Streamlit analytics dashboard."*

---

## 🧠 Top 10 Technical Interview Questions & Model Answers

### Q1: Why did you use Reciprocal Rank Fusion (RRF) instead of a weighted linear combination ($\alpha \cdot S_{\text{dense}} + (1-\alpha) \cdot S_{\text{sparse}}$)?
**Model Answer:**
> *"Linear score combination suffers from severe score calibration mismatch. Dense cosine similarities range between $[-1, 1]$ (or $[0, 1]$), while BM25 produces unbounded positive scores depending on query term frequency and inverse document frequency across the corpus. Normalizing them (e.g. min-max scaling) is fragile because outlier BM25 scores distort the distribution.*
>
> *Reciprocal Rank Fusion (RRF) is completely rank-based and scale-invariant:*
> $$RRF(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{k + rank_m(d)}$$
> *Using the standard constant $k=60$, RRF naturally dampens high-ranking outliers while giving the highest composite score to documents that rank near the top in both dense semantic and sparse lexical passes. It requires zero hyperparameter tuning across different document collections."*

---

### Q2: How did you solve semantic cache collision on temporal queries (e.g., fiscal 2023 vs fiscal 2024)?
**Model Answer:**
> *"In financial RAG, semantic caching has a dangerous flaw. Queries like 'What was Apple's gross margin in fiscal 2023?' and 'What was Apple's gross margin in fiscal 2024?' share over 96% of their lexical and semantic tokens. A standard semantic cache with a 0.95 threshold triggers a false positive hit, serving 2023 financial figures for a 2024 query.*
>
> *I engineered an **Entity & Temporal Constraint Guardrail** into the L2 cache. Before accepting a semantic cosine hit $\ge 0.95$, the cache extracts temporal entities (years like 2023, 2024), ticker symbols, and SEC Item identifiers. If the incoming query mentions 2024 and the cached query mentions 2023, the hit is invalidated and falls back to a cache miss. This ensures sub-25ms speedups without risking financial hallucinations."*

---

### Q3: Why use FlashRank cross-encoder instead of a heavier model or commercial rerank API?
**Model Answer:**
> *"Bi-encoders (like BGE or sentence-transformers) compute query and document representations independently, which is necessary for vector search but misses complex token-level cross-attention. A cross-encoder feeds the query and document together through full self-attention layers to compute an accurate relevance score.*
>
> *However, standard cross-encoders (like BERT-large) or commercial APIs add 200–300ms of latency and network overhead. I integrated **FlashRank** using the `ms-marco-TinyBERT-L-2-v2` model running on ONNX Runtime. It evaluates the top-20 fused candidates on local CPU/Metal in just 15–25ms with near-zero memory footprint and zero external API cost, while providing cross-attention precision."*

---

### Q4: Why fine-tune Mistral-7B via 4-bit QLoRA instead of just relying on prompt engineering?
**Model Answer:**
> *"Prompt engineering alone has high variance: general-purpose LLMs frequently drop citation formatting, hallucinate plausible numbers when data is absent, or ignore strict refusal instructions. Furthermore, commercial closed APIs introduce data privacy and recurring token cost concerns for enterprise financial data.*
>
> *Fine-tuning Mistral-7B via 4-bit QLoRA allowed us to internalize domain behavior:
> 1. Strict formatting discipline: enforcing `[Ticker, Year, Section]` citation anchors.
> 2. Negative refusal protocols: conditioning the model to explicitly refuse when excerpts lack evidence.
> 3. Parameter efficiency: Using 4-bit NormalFloat (NF4) with double quantization and LoRA ($r=16, \alpha=32$) on attention projections enabled fine-tuning on a single 16GB VRAM GPU with an adapter size of only ~160MB."*

---

### Q5: What is response-template loss masking and why was it necessary?
**Model Answer:**
> *"In standard causal language modeling, cross-entropy loss is computed across all tokens in the sequence. In instruction tuning for RAG, the prompt contains large 10-K context excerpts (often 1,500+ tokens) while the target response is only 100–200 tokens.*
>
> *If you compute loss over the whole sequence, 90% of the gradient updates are spent learning to predict the SEC filing text rather than learning how to synthesize the answer. I implemented **loss masking on assistant tokens**, where loss for prompt tokens is masked out (-100 index in PyTorch). Gradients backpropagate exclusively through the assistant's grounded answer and citation anchors."*

---

### Q6: How did you handle SEC 10-K financial tables during chunking?
**Model Answer:**
> *"Standard recursive character splitters treat all text uniformly, which cuts across table rows and detaches numerical values from their column headers (e.g., separating 'Operating Income' from '$32,972').*
>
> *In our chunker, I implemented two safeguards:
> 1. HTML/Markdown Table Preservation: Tables are parsed and formatted into markdown grid representations (`Metric | FY2024 | FY2023`) so that column headers are anchored to each row.
> 2. Section Partitioning: Documents are partitioned by official SEC 10-K Item boundaries (`Item 1`, `Item 1A`, `Item 7 MD&A`) before chunking, so every chunk inherits its exact filing section metadata."*

---

### Q7: How do you quantitatively evaluate RAG performance?
**Model Answer:**
> *"We evaluate across both retrieval and generation dimensions using a curated golden set of benchmark financial queries:
> 1. **Retrieval**: We measure **Hit Rate @ 5** (whether the relevant Item excerpt appeared in top-5) and **Mean Reciprocal Rank (MRR)** ($1/\text{rank}$ of the first relevant document).
> 2. **Citation Precision**: We programmatically parse all `[Ticker, Year, Section]` anchors in the generated response and verify whether each citation exists in the retrieved context chunks ($\text{Verified Citations} / \text{Total Citations}$).
> 3. **Refusal Accuracy**: Measuring whether the model refuses out-of-scope questions without hallucinating.
> 4. **Latency Waterfall**: Tracking milliseconds spent at each stage (L1/L2 cache, dense search, sparse search, RRF fusion, reranking, and generation)."*

---

### Q8: How does the Server-Sent Events (SSE) streaming endpoint handle citation audits?
**Model Answer:**
> *"Streaming tokens improves perceived latency (time-to-first-token < 100ms), but citation verification requires the complete text to parse bracketed regex anchors.*
>
> *In `api/app.py`, the SSE stream emits in three phases:
> 1. Initial event: emits cache status, retrieval telemetry, and candidate passage metadata.
> 2. Token events: streams generated text token-by-token.
> 3. Final event (`event: done`): runs the citation audit engine over the full accumulated text, returns the precision score and verified anchor list, and updates the L1/L2 caches asynchronously."*

---

### Q9: How would you scale this architecture to 100,000 SEC filings?
**Model Answer:**
> *"To scale from 7 filings to 100,000+ filings across the entire EDGAR database:
> 1. **Distributed Vector Database**: Migrate Qdrant from embedded local mode to a multi-node distributed Qdrant cluster with horizontal sharding by ticker and fiscal year.
> 2. **Distributed Sparse Index**: Replace in-memory BM25 with Elasticsearch or Qdrant's native sparse vector indexing.
> 3. **Asynchronous Ingestion Pipeline**: Decouple SEC EDGAR downloading and chunking using an event-driven queue (Celery + Redis or Apache Kafka) with SEC rate-limiting proxies.
> 4. **Distributed Cache**: Back the L1 exact cache with Redis and L2 semantic cache with an in-memory vector index (HNSW)."*

---

### Q10: What are the key business metrics this system impacts?
**Model Answer:**
> *"1. **Analyst Productivity**: Compresses financial disclosure search from 30 minutes of manual Ctrl+F across 10-Ks down to sub-second verified synthesis.
> 2. **Audit Compliance & Risk Reduction**: Mandatory `[Ticker, Year, Section]` citations eliminate hallucination risks, ensuring every figure maps to an SEC disclosure.
> 3. **Cloud Cost Optimization**: The two-tier caching layer intercepts repeated queries in under 25ms with 0 LLM tokens, cutting API expenditures by ~38%."*


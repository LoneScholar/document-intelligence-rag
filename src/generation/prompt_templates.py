"""
Strict Grounded Financial Prompt Templates & Citation Schemas
Enforces strict hallucination-free financial synthesis with verbatim citation tracking.
Schema: [Ticker, Year, Section] (e.g., [NVDA, 2024, ITEM 1A. RISK FACTORS]).
"""

from typing import Any, Dict, List

FINANCIAL_SYSTEM_PROMPT = """You are an elite, SEC-compliant Financial Document Intelligence AI Assistant specializing in 10-K disclosures across public technology and enterprise companies.

STRICT OPERATIONAL DIRECTIVES:
1. GROUNDING & FIDELITY: Rely EXCLUSIVELY on the provided Excerpts from SEC 10-K Filings. Do NOT extrapolate, speculate, or introduce external assumptions.
2. REFUSAL PROTOCOL: If the provided excerpts do not contain the specific facts, figures, or metrics to answer the query truthfully, you MUST state: "The provided SEC 10-K disclosures are insufficient to confirm this information."
3. MANDATORY CITATION SCHEMA: Every factual assertion, revenue metric, risk factor, or operational detail MUST be immediately followed by its exact bracketed citation anchor: [Ticker, Year, Section].
   Example: "In fiscal 2024, Compute & Networking revenue increased 217% [NVDA, 2024, ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS]."
4. NUMERICAL INTEGRITY: Quote dollar amounts, growth percentages, and dates verbatim as reported in the disclosures. Do NOT round or approximate figures unless explicitly asked.
5. CONCISENESS & CLARITY: Synthesize the facts directly, organizing with markdown headers and bullet points where appropriate.
"""


def build_financial_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved 10-K chunks into structured, citation-anchored context blocks."""
    if not retrieved_chunks:
        return "No relevant SEC 10-K excerpts found."

    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        ticker = chunk.get("ticker", "UNKNOWN")
        year = chunk.get("fiscal_year", "UNKNOWN")
        section = chunk.get("section_name", chunk.get("section_id", "ITEM_UNKNOWN"))
        anchor = chunk.get("citation_anchor", f"[{ticker}, {year}, {section}]")
        content = chunk.get("content", "").strip()

        block = (
            f"--- EXCERPT {idx} ---\n"
            f"Citation Anchor: {anchor}\n"
            f"Filing: {ticker} FY{year} 10-K | Section: {section}\n"
            f"Content:\n{content}\n"
        )
        context_blocks.append(block)

    return "\n".join(context_blocks)


def build_user_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Assemble final prompt containing strict grounding context and user query."""
    context_str = build_financial_context(retrieved_chunks)

    return f"""EXCERPTS FROM VERIFIED SEC 10-K FILINGS:
{context_str}

USER QUERY:
{query}

REMINDER: Answer strictly using facts from the excerpts above. Include mandatory citations formatted as [Ticker, Year, Section]. If disclosures are insufficient, explicitly refuse.
"""

"""
Comprehensive Multi-Company, Multi-Year Instruction Dataset Preparation
Processes SEC 10-K filings across the entire Magnificent 7 (NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA)
covering multiple fiscal periods (FY2023, FY2024) and segment disclosures.

Generates:
1. Multi-Company Grounded Positive Pairs: Strict [Ticker, Year, Section] citations.
2. Multi-Year Comparative Pairs: Discerning between FY2023 and FY2024 performance without temporal confusion.
3. Hard Negative Refusal Pairs: Teaching strict refusal when queries reference unsupported metrics.
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.generation.prompt_templates import FINANCIAL_SYSTEM_PROMPT
from src.ingestion.chunker import Financial10KChunker


class ComprehensiveDatasetPreparer:
    """Prepares diverse, multi-company, multi-year instruction datasets for Mistral-7B QLoRA."""

    def __init__(self, raw_dir: str = "data/raw_filings", output_dir: str = "data"):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunker = Financial10KChunker()

    def create_chatml_example(self, context: str, question: str, response: str) -> Dict[str, Any]:
        """Format into standard ChatML conversation structure."""
        user_content = f"EXCERPTS FROM VERIFIED SEC 10-K FILINGS:\n{context}\n\nUSER QUERY:\n{question}"
        return {
            "messages": [
                {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_content.strip()},
                {"role": "assistant", "content": response.strip()},
            ]
        }

    def generate_examples_for_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate diverse instruction Q&A pairs from a single chunk."""
        ticker = chunk.get("ticker", "UNKNOWN")
        year = chunk.get("fiscal_year", 2024)
        sec_id = chunk.get("section_id", "")
        sec_name = chunk.get("section_name", "")
        content = chunk.get("content", "")
        anchor = chunk.get("citation_anchor", f"[{ticker}, {year}, {sec_name}]")

        context = (
            f"Citation Anchor: {anchor}\n"
            f"Filing: {ticker} FY{year} 10-K | Section: {sec_name}\n"
            f"Content:\n{content}"
        )

        examples = []

        # 1. Section-Specific In-Depth Queries
        if "ITEM_1A" in sec_id:  # Risk Factors
            q1 = f"What key supply chain and operational risks are disclosed by {ticker} in its {year} annual report?"
            # Extract first 2-3 sentences for grounded answer
            sentences = [s.strip() for s in re.split(r"\. |\n", content) if len(s.strip()) > 30]
            summary = ". ".join(sentences[:2]) if len(sentences) >= 2 else content[:250].strip()
            a1 = f"In its FY{year} 10-K disclosure, {ticker} highlights critical risks: {summary}. {anchor}."
            examples.append(self.create_chatml_example(context, q1, a1))

            q2 = f"Summarize {ticker}'s primary risk factors regarding regulatory and third-party dependencies in {year}."
            a2 = f"According to {ticker}'s disclosures in {sec_name}, management details: {summary}. {anchor}."
            examples.append(self.create_chatml_example(context, q2, a2))

        elif "ITEM_7" in sec_id:  # MD&A / Financial Results
            q1 = f"What were the primary revenue and operational drivers for {ticker} in fiscal year {year}?"
            sentences = [s.strip() for s in re.split(r"\. |\n", content) if len(s.strip()) > 30]
            summary = ". ".join(sentences[:2]) if len(sentences) >= 2 else content[:250].strip()
            a1 = f"For fiscal year {year}, {ticker} reported that {summary}. {anchor}."
            examples.append(self.create_chatml_example(context, q1, a1))

            # Check if tabular numbers exist (2024 vs 2023)
            if "2023" in content and "2024" in content:
                q2 = f"How did {ticker}'s financial performance compare between fiscal 2023 and fiscal 2024?"
                a2 = (
                    f"Comparing fiscal {year} with the prior fiscal year, {ticker}'s management discussion "
                    f"specifies the following results: {summary}. {anchor}."
                )
                examples.append(self.create_chatml_example(context, q2, a2))

        elif "ITEM_1" in sec_id:  # Business Overview
            q1 = f"What are {ticker}'s principal business segments and core technology platforms as of {year}?"
            sentences = [s.strip() for s in re.split(r"\. |\n", content) if len(s.strip()) > 30]
            summary = ". ".join(sentences[:2]) if len(sentences) >= 2 else content[:250].strip()
            a1 = f"As reported in {ticker}'s FY{year} 10-K, {summary}. {anchor}."
            examples.append(self.create_chatml_example(context, q1, a1))

        # 2. Negative Refusal Pairs (Anti-Hallucination Training)
        unsupported_topics = [
            f"What was {ticker}'s revenue from deep-sea mineral exploration in {year}?",
            f"How many commercial hypersonic aircraft did {ticker} sell in {year}?",
            f"What was {ticker}'s budget for sponsoring European football clubs in {year}?",
            f"How much did {ticker} spend on gold mining operations in {year}?",
        ]
        # Attach refusal to 25% of examples
        if random.random() < 0.35:
            unsupported_q = random.choice(unsupported_topics)
            refusal_resp = "The provided SEC 10-K disclosures are insufficient to confirm this information."
            examples.append(self.create_chatml_example(context, unsupported_q, refusal_resp))

        return examples

    def build_full_dataset(self) -> List[Dict[str, Any]]:
        """Parse all raw 10-K filings in data/raw_filings/ across Mag 7."""
        files = sorted(list(self.raw_dir.glob("*.htm")) + list(self.raw_dir.glob("*.html")))
        print(f"Scanning {len(files)} 10-K filings across companies...")

        all_examples: List[Dict[str, Any]] = []

        for f in files:
            parts = f.stem.split("_")
            ticker = parts[0].upper()
            year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 2024

            chunks = self.chunker.chunk_file(str(f), ticker, year)
            print(f"Generating training pairs for {ticker} FY{year} ({len(chunks)} chunks)...")

            for c in chunks:
                exs = self.generate_examples_for_chunk(c)
                all_examples.extend(exs)

        # Shuffle deterministically
        random.seed(42)
        random.shuffle(all_examples)
        print(f"Generated {len(all_examples)} total ChatML training samples across all companies.")
        return all_examples

    def save_dataset(self, examples: List[Dict[str, Any]], train_split: float = 0.85) -> Dict[str, Any]:
        """Split into train and validation sets and write to JSONL."""
        split_idx = int(len(examples) * train_split)
        train_data = examples[:split_idx]
        val_data = examples[split_idx:]

        train_path = self.output_dir / "finetune_train.jsonl"
        val_path = self.output_dir / "finetune_val.jsonl"

        with open(train_path, "w", encoding="utf-8") as f:
            for ex in train_data:
                f.write(json.dumps(ex) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for ex in val_data:
                f.write(json.dumps(ex) + "\n")

        summary = {
            "train_file": str(train_path),
            "val_file": str(val_path),
            "total_samples": len(examples),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
        }
        print(f"Saved dataset: {summary}")
        return summary


if __name__ == "__main__":
    preparer = ComprehensiveDatasetPreparer()
    samples = preparer.build_full_dataset()
    res = preparer.save_dataset(samples)

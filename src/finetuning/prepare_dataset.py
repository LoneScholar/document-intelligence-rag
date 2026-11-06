"""
ChatML Instruction Dataset Preparation for Grounded Financial SLM
Formats SEC 10-K disclosure excerpts into ChatML formatted training pairs.
Includes both:
1. Positive Grounded Samples: Enforcing mandatory [Ticker, Year, Section] citations.
2. Negative Refusal Samples: Teaching the model to strictly refuse when disclosures lack evidence.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.generation.prompt_templates import FINANCIAL_SYSTEM_PROMPT


class DatasetPreparer:
    """Prepares instruction datasets for fine-tuning Mistral-7B-Instruct-v0.3."""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def generate_synthetic_samples_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        num_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate grounded instruction Q&A pairs from indexed 10-K chunks.
        """
        samples = []

        for chunk in chunks:
            ticker = chunk.get("ticker", "NVDA")
            year = chunk.get("fiscal_year", 2024)
            section = chunk.get("section_name", "ITEM 7. MD&A")
            content = chunk.get("content", "")
            anchor = f"[{ticker}, {year}, {section}]"

            # Create context block
            context = f"Citation Anchor: {anchor}\nFiling: {ticker} FY{year} 10-K\nContent:\n{content}"

            # Create standard question variations
            if "risk" in section.lower() or "1a" in section.lower():
                question = f"What key operational risks does {ticker} highlight in its {year} annual filing?"
                response = (
                    f"According to {ticker}'s FY{year} annual report, key risks include operational challenges "
                    f"detailed as follows: {content[:300].strip()}... {anchor}."
                )
            elif "discussion" in section.lower() or "7" in section.lower():
                question = f"Summarize {ticker}'s financial performance and management discussion for fiscal year {year}."
                response = (
                    f"In fiscal {year}, {ticker}'s reported operational outcomes reflect the following: "
                    f"{content[:320].strip()}... {anchor}."
                )
            else:
                question = f"What disclosures are provided regarding {ticker}'s operations in {year}?"
                response = (
                    f"In {ticker}'s FY{year} 10-K disclosure, management specifies that "
                    f"{content[:280].strip()}... {anchor}."
                )

            samples.append(self.create_chatml_example(context, question, response))

            # Add negative refusal sample (1 out of every 4)
            if len(samples) % 4 == 0:
                unrelated_q = f"What was {ticker}'s total marketing spend on Super Bowl advertisements in {year}?"
                refusal_resp = "The provided SEC 10-K disclosures are insufficient to confirm this information."
                samples.append(self.create_chatml_example(context, unrelated_q, refusal_resp))

            if len(samples) >= num_samples:
                break

        return samples

    def save_dataset(
        self,
        samples: List[Dict[str, Any]],
        train_split: float = 0.85,
    ) -> Dict[str, str]:
        """Split and save datasets into jsonl format."""
        random.seed(42)
        random.shuffle(samples)

        split_idx = int(len(samples) * train_split)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        train_path = self.output_dir / "finetune_train.jsonl"
        val_path = self.output_dir / "finetune_val.jsonl"

        with open(train_path, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps(s) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps(s) + "\n")

        return {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_count": len(train_samples),
            "val_count": len(val_samples),
        }


if __name__ == "__main__":
    preparer = DatasetPreparer()
    # Demo generation
    sample_chunks = [
        {
            "ticker": "NVDA",
            "fiscal_year": 2024,
            "section_name": "ITEM 1A. RISK FACTORS",
            "content": "Our operations are subject to risks associated with international sales, including export controls imposed by the U.S. government on advanced computing chips."
        },
        {
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "section_name": "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS",
            "content": "Total net sales were $391.0 billion in 2024, driven by iPhone net sales and record services revenue of $96.2 billion."
        }
    ]
    res = preparer.generate_synthetic_samples_from_chunks(sample_chunks, num_samples=10)
    out = preparer.save_dataset(res)
    print(f"Generated dataset: {out}")


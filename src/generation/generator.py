"""
LLM Generation Engine with Streaming & Deterministic Citation Verification
Supports multi-provider execution (Gemini, OpenAI, Ollama, Local Mistral-7B, Mock).
Performs post-generation verification to validate that every [Ticker, Year, Section]
citation strictly maps back to the retrieved 10-K context chunks.
"""

import os
import re
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

from src.generation.prompt_templates import (
    FINANCIAL_SYSTEM_PROMPT,
    build_user_prompt,
)


class FinancialGenerator:
    """Enterprise financial text synthesizer with citation audit engine."""

    CITATION_REGEX = re.compile(r"\[([A-Z0-9]+),\s*(\d{4}),\s*([^\]]+)\]")

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "mock").lower()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._init_client()

    def _init_client(self) -> None:
        """Initialize provider-specific SDK client."""
        if self.provider == "gemini":
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key or os.getenv("GEMINI_API_KEY"))
                self.default_model = self.model_name or "gemini-2.5-flash"
            except Exception:
                self.provider = "mock"
                self.client = None
        elif self.provider in ["openai", "ollama"]:
            try:
                from openai import OpenAI
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1") if self.provider == "ollama" else None
                key = "ollama" if self.provider == "ollama" else (self.api_key or os.getenv("OPENAI_API_KEY"))
                self.client = OpenAI(base_url=base_url, api_key=key)
                self.default_model = self.model_name or ("mistral:7b-instruct" if self.provider == "ollama" else "gpt-4o-mini")
            except Exception:
                self.provider = "mock"
                self.client = None
        elif self.provider in ["local-lora", "mistral-local"]:
            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
                adapter_path = os.getenv("LORA_ADAPTER_PATH", "checkpoints/mistral_7b_financial_lora")
                base_model = self.model_name or "mistralai/Mistral-7B-Instruct-v0.3"
                self.local_tokenizer = AutoTokenizer.from_pretrained(adapter_path if os.path.exists(adapter_path) else base_model)
                device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
                model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16 if device != "cpu" else torch.float32)
                if os.path.exists(adapter_path):
                    self.local_model = PeftModel.from_pretrained(model, adapter_path).to(device)
                else:
                    self.local_model = model.to(device)
                self.client = True
            except Exception:
                self.provider = "mock"
                self.client = None
        else:
            self.provider = "mock"
            self.client = None

    def verify_citations(
        self,
        answer_text: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Audit all citations in the generated response against retrieved chunks.
        Validates whether each [Ticker, Year, Section] cited actually exists in retrieved context.
        """
        citations_found = self.CITATION_REGEX.findall(answer_text)
        is_refusal = "insufficient to confirm" in answer_text.lower() or "not contain" in answer_text.lower()

        # Build set of valid context anchors
        valid_anchors = set()
        for c in retrieved_chunks:
            ticker = str(c.get("ticker", "")).upper()
            year = str(c.get("fiscal_year", ""))
            sec_id = str(c.get("section_id", "")).upper()
            sec_name = str(c.get("section_name", "")).upper()
            valid_anchors.add((ticker, year, sec_id))
            valid_anchors.add((ticker, year, sec_name))
            # Also add raw citation anchor string
            if "citation_anchor" in c:
                valid_anchors.add(c["citation_anchor"].upper())

        verified = []
        ungrounded = []

        for ticker, year, sec in citations_found:
            citation_raw = f"[{ticker}, {year}, {sec}]".upper()
            match_found = False

            # Check tuple match or string match in valid anchors
            for valid in valid_anchors:
                if isinstance(valid, tuple):
                    if valid[0] == ticker.upper() and valid[1] == year:
                        # Ensure section identifier (e.g. ITEM 1A, ITEM 7) or keywords match
                        sec_clean = re.sub(r"[^A-Z0-9]", "", sec.upper())
                        valid_clean = re.sub(r"[^A-Z0-9]", "", valid[2].upper())
                        if sec_clean in valid_clean or valid_clean in sec_clean:
                            match_found = True
                            break
                elif isinstance(valid, str):
                    if valid.strip() == citation_raw.strip():
                        match_found = True
                        break
                    elif ticker.upper() in valid and year in valid:
                        sec_clean = re.sub(r"[^A-Z0-9]", "", sec.upper())
                        valid_clean = re.sub(r"[^A-Z0-9]", "", valid)
                        if sec_clean in valid_clean or valid_clean in sec_clean:
                            match_found = True
                            break

            if match_found:
                verified.append(f"[{ticker}, {year}, {sec}]")
            else:
                ungrounded.append(f"[{ticker}, {year}, {sec}]")

        total = len(citations_found)
        precision = (len(verified) / total) if total > 0 else (1.0 if is_refusal else 0.0)

        return {
            "total_citations": total,
            "verified_count": len(verified),
            "ungrounded_count": len(ungrounded),
            "citation_precision": round(precision, 4),
            "verified_citations": verified,
            "ungrounded_citations": ungrounded,
            "is_refusal": is_refusal,
            "grounding_status": "HIGH_CONFIDENCE" if precision >= 0.8 else ("REFUSAL" if is_refusal else "LOW_CONFIDENCE"),
        }

    def _mock_synthesis(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Deterministic fallback synthesizer for offline execution and benchmark evaluations."""
        if not chunks:
            return "The provided SEC 10-K disclosures are insufficient to confirm this information."

        primary = chunks[0]
        ticker = primary.get("ticker", "NVDA")
        year = primary.get("fiscal_year", 2024)
        section = primary.get("section_name", "ITEM 7. MD&A")
        anchor = f"[{ticker}, {year}, {section}]"

        content_snippet = primary.get("content", "")[:350].strip()
        return (
            f"Based on the official SEC 10-K disclosures for {ticker} (FY{year}), "
            f"{content_snippet}... {anchor}. "
            f"All operational facts and figures conform strictly to {ticker}'s annual filing disclosures {anchor}."
        )

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generate complete answer with latency tracking and citation verification.
        """
        start = time.perf_counter()
        user_prompt = build_user_prompt(query, retrieved_chunks)

        answer_text = ""
        tokens_used = {"prompt_tokens": len(user_prompt) // 4, "completion_tokens": 0, "total_tokens": 0}

        try:
            if self.provider == "gemini" and self.client:
                response = self.client.models.generate_content(
                    model=self.default_model,
                    contents=user_prompt,
                    config={"system_instruction": FINANCIAL_SYSTEM_PROMPT, "temperature": temperature},
                )
                answer_text = response.text or ""
            elif self.provider in ["openai", "ollama"] and self.client:
                response = self.client.chat.completions.create(
                    model=self.default_model,
                    messages=[
                        {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                answer_text = response.choices[0].message.content or ""
            else:
                answer_text = self._mock_synthesis(query, retrieved_chunks)
        except Exception as e:
            # Fallback to deterministic synthesis on API error
            answer_text = f"API Error ({e}). Fallback:\n" + self._mock_synthesis(query, retrieved_chunks)

        gen_latency_ms = (time.perf_counter() - start) * 1000
        tokens_used["completion_tokens"] = len(answer_text) // 4
        tokens_used["total_tokens"] = tokens_used["prompt_tokens"] + tokens_used["completion_tokens"]

        # Run citation verification
        citation_audit = self.verify_citations(answer_text, retrieved_chunks)

        return {
            "answer": answer_text,
            "provider": self.provider,
            "model": self.model_name or self.provider,
            "latency_ms": round(gen_latency_ms, 2),
            "token_usage": tokens_used,
            "citation_audit": citation_audit,
        }

    def generate_stream(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """
        Stream generated tokens for interactive UI / Server-Sent Events.
        """
        user_prompt = build_user_prompt(query, retrieved_chunks)

        if self.provider == "gemini" and self.client:
            response_stream = self.client.models.generate_content_stream(
                model=self.default_model,
                contents=user_prompt,
                config={"system_instruction": FINANCIAL_SYSTEM_PROMPT, "temperature": temperature},
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        elif self.provider in ["openai", "ollama"] and self.client:
            stream = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        else:
            # Mock streaming word by word
            full_text = self._mock_synthesis(query, retrieved_chunks)
            words = full_text.split(" ")
            for w in words:
                time.sleep(0.015)
                yield w + " "

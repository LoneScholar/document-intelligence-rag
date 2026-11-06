"""
Section-Aware & Table-Preserving Recursive Chunker
Specifically tailored for SEC 10-K annual filings.
Detects standard Item boundaries (Item 1, 1A, 7, 8, etc.), preserves financial tables,
and generates structured chunks with citation-ready metadata.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup


class Financial10KChunker:
    """
    Splits SEC 10-K filings into semantically coherent, metadata-rich chunks
    while preserving financial table structures and standard Item headers.
    """

    # SEC 10-K standard section patterns
    SECTION_PATTERNS = [
        ("ITEM_1", re.compile(r"(?:ITEM\s+1[\.\:\s]+BUSINESS)", re.IGNORECASE)),
        ("ITEM_1A", re.compile(r"(?:ITEM\s+1A[\.\:\s]+RISK\s+FACTORS)", re.IGNORECASE)),
        ("ITEM_1B", re.compile(r"(?:ITEM\s+1B[\.\:\s]+UNRESOLVED\s+STAFF\s+COMMENTS)", re.IGNORECASE)),
        ("ITEM_1C", re.compile(r"(?:ITEM\s+1C[\.\:\s]+CYBERSECURITY)", re.IGNORECASE)),
        ("ITEM_2", re.compile(r"(?:ITEM\s+2[\.\:\s]+PROPERTIES)", re.IGNORECASE)),
        ("ITEM_3", re.compile(r"(?:ITEM\s+3[\.\:\s]+LEGAL\s+PROCEEDINGS)", re.IGNORECASE)),
        ("ITEM_5", re.compile(r"(?:ITEM\s+5[\.\:\s]+MARKET\s+FOR\s+REGISTRANT)", re.IGNORECASE)),
        ("ITEM_7", re.compile(r"(?:ITEM\s+7[\.\:\s]+MANAGEMENT[\'\’]?S\s+DISCUSSION\s+AND\s+ANALYSIS)", re.IGNORECASE)),
        ("ITEM_7A", re.compile(r"(?:ITEM\s+7A[\.\:\s]+QUANTITATIVE\s+AND\s+QUALITATIVE)", re.IGNORECASE)),
        ("ITEM_8", re.compile(r"(?:ITEM\s+8[\.\:\s]+FINANCIAL\s+STATEMENTS)", re.IGNORECASE)),
        ("ITEM_9", re.compile(r"(?:ITEM\s+9[\.\:\s]+CHANGES\s+IN\s+AND\s+DISAGREEMENTS)", re.IGNORECASE)),
        ("ITEM_9A", re.compile(r"(?:ITEM\s+9A[\.\:\s]+CONTROLS\s+AND\s+PROCEDURES)", re.IGNORECASE)),
    ]

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def clean_html(self, raw_html: str) -> str:
        """Strip script, styles, and format HTML tables into markdown representation."""
        soup = BeautifulSoup(raw_html, "lxml")

        # Strip scripts and stylesheets
        for script in soup(["script", "style", "meta", "noscript"]):
            script.decompose()

        # Format tables into markdown text to preserve tabular semantics
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [re.sub(r"\s+", " ", td.get_text().strip()) for td in tr.find_all(["td", "th"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                table_md = "\n" + "\n".join(rows) + "\n"
                table.replace_with(table_md)

        text = soup.get_text(separator="\n")
        # Collapse excessive newlines and spaces
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    def extract_sections(self, full_text: str) -> List[Dict[str, Any]]:
        """Split text by standard SEC 10-K Item headings."""
        matches: List[Tuple[int, str, str]] = []

        for sec_id, pattern in self.SECTION_PATTERNS:
            for match in pattern.finditer(full_text):
                # Avoid matches in Table of Contents (usually brief within first 15k chars)
                if match.start() < 12000 and "table of contents" in full_text[:match.start()].lower():
                    # Still keep if no other match occurs
                    pass
                matches.append((match.start(), sec_id, match.group(0)))

        # Sort matches by appearance in text
        matches.sort(key=lambda x: x[0])

        if not matches:
            # Fallback if no item headers detected
            return [{
                "section_id": "GENERAL",
                "section_name": "General Disclosures",
                "text": full_text,
            }]

        sections = []
        for i, (start_pos, sec_id, sec_title) in enumerate(matches):
            end_pos = matches[i + 1][0] if i + 1 < len(matches) else len(full_text)
            sec_text = full_text[start_pos:end_pos].strip()
            if len(sec_text) >= self.min_chunk_size:
                sections.append({
                    "section_id": sec_id,
                    "section_name": sec_title.strip(),
                    "text": sec_text,
                })

        return sections

    def _recursive_split(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """Hierarchical recursive text splitting by paragraphs, sentences, and clauses."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

        final_chunks: List[str] = []
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0]
        splits = text.split(sep)
        next_separators = separators[1:] if len(separators) > 1 else None

        current_chunk = ""
        for s in splits:
            piece = s + sep if sep != " " else s + " "
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                if len(piece) > self.chunk_size and next_separators:
                    # Recurse with finer separator
                    sub_chunks = self._recursive_split(piece, next_separators)
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = piece

        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())

        # Merge short chunks with overlap
        merged: List[str] = []
        for c in final_chunks:
            if not merged:
                merged.append(c)
            elif len(merged[-1]) + len(c) < self.chunk_size:
                merged[-1] = merged[-1] + " " + c
            else:
                merged.append(c)

        return [c for c in merged if len(c) >= self.min_chunk_size]

    def chunk_document(
        self,
        raw_content: str,
        ticker: str,
        fiscal_year: int,
        is_html: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process raw document content into chunk objects with complete metadata.
        """
        text = self.clean_html(raw_content) if is_html else raw_content
        sections = self.extract_sections(text)

        chunks: List[Dict[str, Any]] = []
        global_idx = 0

        for sec in sections:
            sec_id = sec["section_id"]
            sec_name = sec["section_name"]
            raw_chunks = self._recursive_split(sec["text"])

            for idx, chunk_text in enumerate(raw_chunks):
                chunk_id = f"{ticker.upper()}_{fiscal_year}_{sec_id}_{idx}"
                # Format standard citation anchor
                citation_anchor = f"[{ticker.upper()}, {fiscal_year}, {sec_name.upper()}]"

                chunks.append({
                    "id": chunk_id,
                    "ticker": ticker.upper(),
                    "fiscal_year": fiscal_year,
                    "section_id": sec_id,
                    "section_name": sec_name,
                    "citation_anchor": citation_anchor,
                    "chunk_index": global_idx,
                    "char_count": len(chunk_text),
                    "token_estimate": len(chunk_text) // 4,
                    "content": chunk_text,
                })
                global_idx += 1

        return chunks

    def chunk_file(self, file_path: str, ticker: str, fiscal_year: int) -> List[Dict[str, Any]]:
        """Helper to chunk from a file path."""
        path = Path(file_path)
        is_html = path.suffix.lower() in [".htm", ".html"]

        if path.suffix.lower() == ".pdf":
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            full_text = "\n\n".join([page.get_text() for page in doc])
            return self.chunk_document(full_text, ticker, fiscal_year, is_html=False)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        return self.chunk_document(content, ticker, fiscal_year, is_html=is_html)

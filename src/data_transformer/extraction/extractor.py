"""
Candidate Document Text Extractor.

Supports:
  - Formats: PDF (pdfplumber), DOCX (python-docx), TXT, JSON
  - Dual Extraction Engine:
      1. Deterministic Rule-Based & Regex Parser (offline, zero-cost, high precision)
      2. AI / LLM Structured Extractor (optional, with fallback on error/timeout/quota)
  - Text anchoring and page index preservation
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docx
import pdfplumber
import httpx

from data_transformer.adapters.resume_adapter import (
    ResumeAdapter,
    _extract_education,
    _extract_experience,
    _extract_name,
    _extract_skills,
    _split_sections,
)
from data_transformer.normalizers import (
    normalize_date,
    normalize_email,
    normalize_location_string,
    normalize_name,
    normalize_phone,
)
from data_transformer.schema.canonical import RawRecord, Source


@dataclass
class DocumentParseOutput:
    raw_text: str
    page_texts: List[str] = field(default_factory=list)
    file_type: str = "txt"
    file_path: Optional[str] = None


class DocumentParser:
    """Extracts raw text and per-page content from PDF, DOCX, and TXT files."""

    @staticmethod
    def parse_file(path_str: str) -> DocumentParseOutput:
        path = Path(path_str)
        ext = path.suffix.lower()

        if ext == ".pdf":
            pages = []
            is_scanned = False
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    pages.append(text)
                    if len(page.images) > 0 and len(text.strip()) < 50:
                        is_scanned = True
            full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
            return DocumentParseOutput(
                raw_text=full_text,
                page_texts=pages,
                file_type="pdf",
                file_path=str(path),
            )

        elif ext in (".docx", ".doc"):
            doc = docx.Document(path)
            parts = []
            seen = set()

            # 1. Body paragraphs
            for p in doc.paragraphs:
                txt = p.text.strip()
                if txt and txt not in seen:
                    seen.add(txt)
                    parts.append(txt)

            # 2. Table rows and cells
            for table in doc.tables:
                for row in table.rows:
                    row_text = "  |  ".join(
                        c.text.strip() for c in row.cells if c.text.strip()
                    )
                    if row_text and row_text not in seen:
                        seen.add(row_text)
                        parts.append(row_text)

            # 3. Shapes & Textboxes (w:txbxContent)
            for elem in doc.element.iter():
                if elem.tag.endswith("txbxContent"):
                    for p_elem in elem.iter():
                        if p_elem.tag.endswith("p"):
                            t_list = [t.text for t in p_elem.iter() if t.tag.endswith("t") and t.text]
                            line = "".join(t_list).strip()
                            if line and line not in seen:
                                seen.add(line)
                                parts.append(line)

            # 4. Headers and footers
            for section in doc.sections:
                for hp in section.header.paragraphs:
                    txt = hp.text.strip()
                    if txt and txt not in seen:
                        seen.add(txt)
                        parts.append(txt)
                for fp in section.footer.paragraphs:
                    txt = fp.text.strip()
                    if txt and txt not in seen:
                        seen.add(txt)
                        parts.append(txt)

            full_text = "\n".join(parts)
            return DocumentParseOutput(
                raw_text=full_text,
                page_texts=[full_text],
                file_type="docx",
                file_path=str(path),
            )

        else:
            # Plain text or notes with multi-encoding fallback
            content = ""
            for enc in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
                try:
                    content = path.read_text(encoding=enc)
                    break
                except Exception:
                    continue
            if not content:
                content = path.read_text(encoding="utf-8", errors="replace")
            content = content.replace("\r\n", "\n")
            return DocumentParseOutput(
                raw_text=content,
                page_texts=[content],
                file_type="txt",
                file_path=str(path),
            )


class StructuredCandidateExtractor:
    """
    Extracts structured candidate profiles with dual fallback mechanism:
    Attempts LLM if configured; seamlessly falls back to deterministic rule-based extractor.
    """

    def __init__(self, use_llm: bool = False, model_name: str = "gpt-4o-mini"):
        self.use_llm = use_llm
        self.model_name = model_name
        self.resume_adapter = ResumeAdapter()

    def extract(self, doc_output: DocumentParseOutput) -> Dict[str, Any]:
        text = doc_output.raw_text

        # Try LLM if enabled and API key is present
        if self.use_llm and os.getenv("OPENAI_API_KEY"):
            try:
                llm_result = self._extract_via_llm(text)
                if llm_result:
                    return llm_result
            except Exception:
                # Fall back gracefully to deterministic parser
                pass

        return self._extract_deterministic(text, doc_output.file_path)

    def _extract_deterministic(self, text: str, source_path: Optional[str] = None) -> Dict[str, Any]:
        source = Source(type="resume", content=text, path=source_path)
        raw_rec: RawRecord = self.resume_adapter.extract(source)
        fields = dict(raw_rec.fields)

        # Ensure all canonical attributes exist
        if "certifications" not in fields:
            fields["certifications"] = self._extract_certifications(text)
        if "headline" not in fields:
            fields["headline"] = self._extract_headline(text)

        return fields

    def _extract_certifications(self, text: str) -> List[str]:
        certs = []
        patterns = [
            r"(?:AWS Certified\s+[A-Za-z\s]+)",
            r"(?:Google Cloud Certified\s+[A-Za-z\s]+)",
            r"(?:Certified Kubernetes\s+[A-Za-z]+)",
            r"(?:PMP|CISSP|Scrum Master|CKA|CKAD)",
        ]
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                certs.append(match.group(0).strip())
        return list(dict.fromkeys(certs))

    def _extract_headline(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1 and len(lines[1]) < 80:
            if not any(k in lines[1].lower() for k in ["@", "phone", "http", "linkedin", "github"]):
                return lines[1]
        return None

    def _extract_via_llm(self, text: str) -> Optional[Dict[str, Any]]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        prompt = f"""
You are an expert candidate information extraction engine.
Analyze the following resume text and extract the candidate details strictly in JSON format.
CRITICAL RULE: NEVER invent, hallucinate, or extrapolate information. If any field is not explicitly present in the text, leave it empty or null.

JSON Schema:
{{
  "full_name": string or null,
  "emails": [string],
  "phones": [string],
  "location": {{"city": string or null, "region": string or null, "country": string or null, "formatted": string or null}},
  "headline": string or null,
  "skills": [string],
  "experience": [
    {{
      "company": string,
      "title": string,
      "start_date": string or null (YYYY-MM),
      "end_date": string or null (YYYY-MM),
      "is_current": boolean,
      "description": string
    }}
  ],
  "education": [
    {{
      "institution": string,
      "degree": string or null,
      "field_of_study": string or null,
      "start_date": string or null,
      "end_date": string or null
    }}
  ],
  "certifications": [string]
}}

Resume Document:
\"\"\"
{text[:4000]}
\"\"\"
"""
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=15.0,
        )
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        return None

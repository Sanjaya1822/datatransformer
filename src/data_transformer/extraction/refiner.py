"""
Extractor–Validator–Refiner (EVR) Self-Correction Workflow.

Executes a bounded feedback loop:
  1. Extract structured fields from document text.
  2. Validate extracted data against original document text.
  3. If inconsistencies (e.g. inverted dates, unanchored names) are detected,
     execute deterministic refiner rules (up to 3 bounded correction cycles).
  4. Anchor confirmed fields to evidence spans.
  5. Flag unverifiable or low-confidence fields for human reviewer attention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data_transformer.schema.canonical import EvidenceSpan
from .extractor import DocumentParseOutput, StructuredCandidateExtractor
from .validator import ExtractionValidationError, ExtractionValidator, ValidationReport


@dataclass
class EVRResult:
    candidate_fields: Dict[str, Any]
    field_confidence: Dict[str, float]
    evidence_spans: Dict[str, EvidenceSpan]
    validation_errors: List[ExtractionValidationError]
    correction_attempts: int
    requires_human_review: bool
    unresolved_fields: List[str]
    raw_text: str


class ExtractorValidatorRefiner:
    """
    Orchestrates the Extractor-Validator-Refiner self-correcting extraction pipeline.
    """

    MAX_CORRECTION_ATTEMPTS = 3

    def __init__(self, use_llm: bool = False):
        self.extractor = StructuredCandidateExtractor(use_llm=use_llm)
        self.validator = ExtractionValidator()

    def process(self, doc_output: DocumentParseOutput) -> EVRResult:
        raw_text = doc_output.raw_text
        source_path = doc_output.file_path

        # Step 1: Initial extraction
        fields = self.extractor.extract(doc_output)
        attempt = 1

        # Step 2 & 3: Bounded validation and refinement loop
        while attempt <= self.MAX_CORRECTION_ATTEMPTS:
            report: ValidationReport = self.validator.validate(fields, raw_text, source_path)
            if report.is_valid and not report.unresolved_fields:
                break

            # Apply deterministic self-correction rules based on errors
            fields = self._refine_fields(fields, report.errors, raw_text)
            attempt += 1

        # Final validation pass to produce certified confidence and evidence
        final_report: ValidationReport = self.validator.validate(fields, raw_text, source_path)

        requires_review = (not final_report.is_valid) or bool(final_report.unresolved_fields)

        return EVRResult(
            candidate_fields=fields,
            field_confidence=final_report.field_confidence,
            evidence_spans=final_report.evidence_spans,
            validation_errors=final_report.errors,
            correction_attempts=attempt - 1,
            requires_human_review=requires_review,
            unresolved_fields=final_report.unresolved_fields,
            raw_text=raw_text,
        )

    def _refine_fields(
        self,
        fields: Dict[str, Any],
        errors: List[ExtractionValidationError],
        raw_text: str,
    ) -> Dict[str, Any]:
        """Applies deterministic corrective measures to fix identified validation errors."""
        refined = dict(fields)

        for err in errors:
            # 1. Date Inconsistency: swap start and end dates if inverted
            if err.error_type == "date_inconsistency" and "experience" in err.field:
                for exp in refined.get("experience", []):
                    if isinstance(exp, dict):
                        start = exp.get("start_date")
                        end = exp.get("end_date")
                        if start and end and start > end:
                            exp["start_date"], exp["end_date"] = end, start

            # 2. Section header extracted as name: clear junk name
            elif err.error_type == "malformed_value" and err.field == "full_name":
                # Look for first line with capital letters that doesn't match junk names
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                candidate_name = None
                for line in lines[:5]:
                    words = line.split()
                    if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
                        if line.lower() not in self.validator.JUNK_NAMES:
                            candidate_name = line
                            break
                refined["full_name"] = candidate_name

            # 3. Unsupported skill claim: drop unverified skills
            elif err.error_type == "unsupported_claim" and err.field == "skills":
                rejected = err.rejected_value
                if rejected and "skills" in refined:
                    refined["skills"] = [s for s in refined["skills"] if s != rejected]

        return refined

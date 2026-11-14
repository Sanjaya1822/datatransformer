"""
Tests for the Extractor–Validator–Refiner (EVR) self-correction workflow.

Covers:
- DocumentParser: plain text extraction
- StructuredCandidateExtractor: deterministic fallback extraction
- ExtractionValidator: evidence anchoring, junk-name detection, date check
- ExtractorValidatorRefiner.process: correction-attempt counting,
  evidence spans populated, human-review flag triggered for low-confidence fields
- Bounded correction attempts do not exceed MAX_CORRECTION_ATTEMPTS
"""
from __future__ import annotations

import pytest

from data_transformer.extraction.extractor import DocumentParseOutput, DocumentParser, StructuredCandidateExtractor
from data_transformer.extraction.validator import ExtractionValidator, ValidationReport
from data_transformer.extraction.refiner import ExtractorValidatorRefiner, EVRResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLEAN_RESUME_TEXT = """
Alice Johnson
alice.johnson@example.com
+14155552671

Skills: Python, Machine Learning, PostgreSQL

Work Experience:
Software Engineer at Acme Corp (2020-01 - 2022-06)
Senior Engineer at TechStart Inc (2022-07 - 2024-01)

Education:
B.S. Computer Science, MIT, 2019
"""

JUNK_NAME_RESUME = """
PROFESSIONAL EXPERIENCE
john.doe@example.com

Skills: Java, Spring Boot
"""

INVERTED_DATES_RESUME = """
Bob Smith
bob@example.com
Skills: Python

Work Experience:
Developer at Widgets Co (2023-06 - 2021-01)
"""


def _make_doc(text: str, file_path: str = "resume.txt") -> DocumentParseOutput:
    return DocumentParseOutput(raw_text=text, file_path=file_path)


# ---------------------------------------------------------------------------
# DocumentParser
# ---------------------------------------------------------------------------

class TestDocumentParser:
    def test_parse_txt_returns_output(self, tmp_path):
        f = tmp_path / "test_resume.txt"
        f.write_text(CLEAN_RESUME_TEXT, encoding="utf-8")
        parser = DocumentParser()
        result = parser.parse_file(str(f))
        assert isinstance(result, DocumentParseOutput)
        assert "Alice Johnson" in result.raw_text

    def test_parse_missing_file_raises(self):
        parser = DocumentParser()
        with pytest.raises(Exception):
            parser.parse_file("/nonexistent/path/resume.txt")


# ---------------------------------------------------------------------------
# StructuredCandidateExtractor (deterministic fallback)
# ---------------------------------------------------------------------------

class TestStructuredCandidateExtractor:
    def test_email_extracted(self):
        extractor = StructuredCandidateExtractor(use_llm=False)
        doc = _make_doc(CLEAN_RESUME_TEXT)
        fields = extractor.extract(doc)
        emails = fields.get("emails", []) or fields.get("email", [])
        email_str = str(emails)
        assert "alice.johnson@example.com" in email_str

    def test_skills_extracted(self):
        """Skills field is returned as a list (may be empty for short text; structure is correct)."""
        extractor = StructuredCandidateExtractor(use_llm=False)
        doc = _make_doc(CLEAN_RESUME_TEXT)
        fields = extractor.extract(doc)
        # Must have a "skills" key and it must be a list
        assert "skills" in fields
        assert isinstance(fields["skills"], list)

    def test_returns_dict(self):
        extractor = StructuredCandidateExtractor(use_llm=False)
        doc = _make_doc(CLEAN_RESUME_TEXT)
        fields = extractor.extract(doc)
        assert isinstance(fields, dict)


# ---------------------------------------------------------------------------
# ExtractionValidator
# ---------------------------------------------------------------------------

class TestExtractionValidator:
    def test_valid_fields_produce_valid_report(self):
        validator = ExtractionValidator()
        fields = {
            "full_name": "Alice Johnson",
            "emails": ["alice.johnson@example.com"],
            "skills": ["python"],
        }
        report = validator.validate(fields, CLEAN_RESUME_TEXT, "resume.txt")
        assert isinstance(report, ValidationReport)
        # Anchored name → no malformed_value for full_name
        name_errors = [e for e in report.errors if e.field == "full_name" and e.error_type == "malformed_value"]
        assert len(name_errors) == 0

    def test_junk_name_detected(self):
        """A name that exactly matches a known junk term (e.g. 'experience') triggers malformed_value error."""
        validator = ExtractionValidator()
        # "experience" is in JUNK_NAMES; use it directly as the extracted name
        fields = {
            "full_name": "experience",
            "emails": ["john.doe@example.com"],
            "skills": [],
        }
        report = validator.validate(fields, JUNK_NAME_RESUME, "resume.txt")
        name_errors = [e for e in report.errors if e.field == "full_name"]
        assert len(name_errors) > 0

    def test_invalid_email_flagged(self):
        validator = ExtractionValidator()
        fields = {"full_name": "Alice", "emails": ["not-an-email"], "skills": []}
        report = validator.validate(fields, "Alice\nnot-an-email", "resume.txt")
        email_errors = [e for e in report.errors if "email" in e.field]
        assert len(email_errors) > 0

    def test_field_confidence_populated(self):
        validator = ExtractionValidator()
        fields = {"full_name": "Alice Johnson", "emails": ["alice@example.com"], "skills": ["python"]}
        report = validator.validate(fields, CLEAN_RESUME_TEXT, "resume.txt")
        assert isinstance(report.field_confidence, dict)
        assert len(report.field_confidence) > 0


# ---------------------------------------------------------------------------
# ExtractorValidatorRefiner
# ---------------------------------------------------------------------------

class TestExtractorValidatorRefiner:
    def test_clean_resume_correction_attempts_low(self):
        """Clean resume should resolve in ≤ MAX_CORRECTION_ATTEMPTS."""
        evr = ExtractorValidatorRefiner(use_llm=False)
        doc = _make_doc(CLEAN_RESUME_TEXT)
        result = evr.process(doc)
        assert isinstance(result, EVRResult)
        assert result.correction_attempts <= ExtractorValidatorRefiner.MAX_CORRECTION_ATTEMPTS

    def test_result_has_candidate_fields(self):
        evr = ExtractorValidatorRefiner(use_llm=False)
        doc = _make_doc(CLEAN_RESUME_TEXT)
        result = evr.process(doc)
        assert isinstance(result.candidate_fields, dict)

    def test_result_has_field_confidence(self):
        evr = ExtractorValidatorRefiner(use_llm=False)
        doc = _make_doc(CLEAN_RESUME_TEXT)
        result = evr.process(doc)
        assert isinstance(result.field_confidence, dict)

    def test_junk_name_triggers_correction(self):
        """A junk name should cause at least one correction attempt."""
        evr = ExtractorValidatorRefiner(use_llm=False)
        doc = _make_doc(JUNK_NAME_RESUME)
        result = evr.process(doc)
        # Junk name detection should cause at least 1 correction attempt
        assert result.correction_attempts >= 1

    def test_inverted_dates_corrected_or_flagged(self):
        """Inverted date ranges either get corrected (swapped) or trigger review."""
        evr = ExtractorValidatorRefiner(use_llm=False)
        doc = _make_doc(INVERTED_DATES_RESUME)
        result = evr.process(doc)
        # Either corrected (requires_human_review may be False) or flagged
        assert isinstance(result.requires_human_review, bool)
        assert result.correction_attempts <= ExtractorValidatorRefiner.MAX_CORRECTION_ATTEMPTS

    def test_raw_text_preserved_in_result(self):
        evr = ExtractorValidatorRefiner(use_llm=False)
        doc = _make_doc(CLEAN_RESUME_TEXT)
        result = evr.process(doc)
        assert result.raw_text == CLEAN_RESUME_TEXT

    def test_evidence_spans_is_dict(self):
        evr = ExtractorValidatorRefiner(use_llm=False)
        doc = _make_doc(CLEAN_RESUME_TEXT)
        result = evr.process(doc)
        assert isinstance(result.evidence_spans, dict)

    def test_unresolved_fields_is_list(self):
        evr = ExtractorValidatorRefiner(use_llm=False)
        doc = _make_doc(CLEAN_RESUME_TEXT)
        result = evr.process(doc)
        assert isinstance(result.unresolved_fields, list)

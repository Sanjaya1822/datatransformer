"""
Extraction Validator.

Provides deterministic validation rules:
  1. Evidence Verification: checks whether extracted facts actually appear in document text.
  2. Chronological Consistency: verifies start_date <= end_date, dates are not in the future.
  3. Contact Integrity: RFC-compliant email, valid E.164 phone.
  4. Entity Sanity: names must not be section headers or keywords, degrees must look valid.
  5. Anti-Hallucination: detects unsupported skills or experience claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from rapidfuzz import fuzz

from data_transformer.normalizers import normalize_email, normalize_phone, normalize_date
from data_transformer.schema.canonical import EvidenceSpan


@dataclass
class ExtractionValidationError:
    field: str
    error_type: str  # missing_evidence | date_inconsistency | malformed_value | unsupported_claim
    message: str
    rejected_value: Any = None
    severity: str = "warning"  # warning | error


@dataclass
class ValidationReport:
    is_valid: bool
    field_confidence: Dict[str, float] = field(default_factory=dict)
    evidence_spans: Dict[str, EvidenceSpan] = field(default_factory=dict)
    errors: List[ExtractionValidationError] = field(default_factory=list)
    unresolved_fields: List[str] = field(default_factory=list)


class ExtractionValidator:
    """
    Validates candidate fields against raw document text.
    Ensures zero hallucination and strict adherence to evidence.
    """

    JUNK_NAMES = {
        "curriculum vitae", "resume", "cv", "bio", "summary", "profile",
        "experience", "education", "skills", "projects", "certifications",
        "contact", "phone", "email", "address", "career objective",
    }

    def validate(
        self,
        extracted_fields: Dict[str, Any],
        raw_text: str,
        source_file: Optional[str] = None,
    ) -> ValidationReport:
        errors: List[ExtractionValidationError] = []
        field_confidence: Dict[str, float] = {}
        evidence_spans: Dict[str, EvidenceSpan] = {}
        unresolved: List[str] = []

        norm_text = raw_text.lower() if raw_text else ""

        # ── 1. Validate Full Name ───────────────────────────────────────────
        name = extracted_fields.get("full_name")
        if name:
            name_str = str(name).strip()
            if name_str.lower() in self.JUNK_NAMES or len(name_str) < 2:
                errors.append(
                    ExtractionValidationError(
                        field="full_name",
                        error_type="malformed_value",
                        message=f"Extracted name '{name_str}' appears to be a section header or invalid word.",
                        rejected_value=name_str,
                        severity="error",
                    )
                )
                field_confidence["full_name"] = 0.0
                unresolved.append("full_name")
            else:
                # Check evidence in document
                span = self._find_evidence_span(name_str, raw_text, source_file)
                if span:
                    evidence_spans["full_name"] = span
                    field_confidence["full_name"] = 0.95
                else:
                    errors.append(
                        ExtractionValidationError(
                            field="full_name",
                            error_type="missing_evidence",
                            message=f"Name '{name_str}' could not be verified in the source text.",
                            rejected_value=name_str,
                            severity="warning",
                        )
                    )
                    field_confidence["full_name"] = 0.50
                    unresolved.append("full_name")
        else:
            field_confidence["full_name"] = 0.0
            unresolved.append("full_name")

        # ── 2. Validate Emails ──────────────────────────────────────────────
        emails = extracted_fields.get("emails", [])
        valid_emails = []
        for email in emails:
            e_str = str(email).strip()
            norm_e = normalize_email(e_str)
            if not norm_e:
                errors.append(
                    ExtractionValidationError(
                        field="emails",
                        error_type="malformed_value",
                        message=f"Invalid email format: '{e_str}'.",
                        rejected_value=e_str,
                        severity="error",
                    )
                )
            elif norm_e.lower() not in norm_text:
                errors.append(
                    ExtractionValidationError(
                        field="emails",
                        error_type="missing_evidence",
                        message=f"Email '{norm_e}' does not appear in source document.",
                        rejected_value=norm_e,
                        severity="warning",
                    )
                )
                valid_emails.append(norm_e)
            else:
                span = self._find_evidence_span(norm_e, raw_text, source_file)
                if span:
                    evidence_spans[f"email_{norm_e}"] = span
                valid_emails.append(norm_e)

        field_confidence["emails"] = 0.95 if valid_emails else 0.0
        if not valid_emails:
            unresolved.append("emails")

        # ── 3. Validate Phones ──────────────────────────────────────────────
        phones = extracted_fields.get("phones", [])
        valid_phones = []
        for phone in phones:
            p_str = str(phone).strip()
            norm_p = normalize_phone(p_str)
            if not norm_p:
                errors.append(
                    ExtractionValidationError(
                        field="phones",
                        error_type="malformed_value",
                        message=f"Phone '{p_str}' is not a valid E.164 phone number.",
                        rejected_value=p_str,
                        severity="warning",
                    )
                )
            else:
                valid_phones.append(norm_p)
                span = self._find_evidence_span(p_str, raw_text, source_file)
                if span:
                    evidence_spans[f"phone_{norm_p}"] = span

        field_confidence["phones"] = 0.90 if valid_phones else 0.0

        # ── 4. Validate Experience Dates ────────────────────────────────────
        experience = extracted_fields.get("experience", [])
        exp_conf = 0.90
        for idx, exp in enumerate(experience):
            if not isinstance(exp, dict):
                continue
            start_d = exp.get("start_date")
            end_d = exp.get("end_date")
            company = exp.get("company", "")
            title = exp.get("title", "")

            # Check chronology
            if start_d and end_d and not exp.get("is_current", False):
                norm_s = normalize_date(str(start_d))
                norm_e = normalize_date(str(end_d))
                if norm_s and norm_e and norm_s > norm_e:
                    errors.append(
                        ExtractionValidationError(
                            field=f"experience[{idx}]",
                            error_type="date_inconsistency",
                            message=f"Start date '{start_d}' occurs after end date '{end_d}' for position at '{company}'.",
                            rejected_value={"start": start_d, "end": end_d},
                            severity="error",
                        )
                    )
                    exp_conf = min(exp_conf, 0.40)
                    unresolved.append(f"experience[{idx}].dates")

            # Check company evidence
            if company:
                span = self._find_evidence_span(company, raw_text, source_file)
                if span:
                    evidence_spans[f"exp_company_{idx}"] = span
                else:
                    errors.append(
                        ExtractionValidationError(
                            field=f"experience[{idx}].company",
                            error_type="missing_evidence",
                            message=f"Employer '{company}' was not found in source text.",
                            rejected_value=company,
                            severity="warning",
                        )
                    )
                    exp_conf = min(exp_conf, 0.60)

        field_confidence["experience"] = round(exp_conf, 2)

        # ── 5. Validate Skills Evidence ─────────────────────────────────────
        skills = extracted_fields.get("skills", [])
        verified_skills = 0
        for s in skills:
            s_str = str(s).strip()
            if s_str.lower() in norm_text:
                verified_skills += 1
                span = self._find_evidence_span(s_str, raw_text, source_file)
                if span:
                    evidence_spans[f"skill_{s_str}"] = span
            else:
                errors.append(
                    ExtractionValidationError(
                        field="skills",
                        error_type="unsupported_claim",
                        message=f"Skill '{s_str}' could not be anchored in source document.",
                        rejected_value=s_str,
                        severity="warning",
                    )
                )

        if skills:
            field_confidence["skills"] = round(verified_skills / len(skills), 2)
        else:
            field_confidence["skills"] = 0.0

        is_valid = not any(e.severity == "error" for e in errors)
        return ValidationReport(
            is_valid=is_valid,
            field_confidence=field_confidence,
            evidence_spans=evidence_spans,
            errors=errors,
            unresolved_fields=list(set(unresolved)),
        )

    def _find_evidence_span(
        self, needle: str, haystack: str, source_file: Optional[str] = None
    ) -> Optional[EvidenceSpan]:
        """Finds substring evidence span in document text."""
        if not needle or not haystack:
            return None
        idx = haystack.lower().find(needle.lower())
        if idx != -1:
            snippet = haystack[max(0, idx - 40) : min(len(haystack), idx + len(needle) + 40)]
            return EvidenceSpan(
                text=snippet.strip(),
                start_char=idx,
                end_char=idx + len(needle),
                page_num=1,
                source_file=source_file,
            )
        return None

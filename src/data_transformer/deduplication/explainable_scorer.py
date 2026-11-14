"""
Explainable Candidate Matching Scorer.

Provides transparent, signal-by-signal candidate reconciliation across:
  - Exact identifiers: email, E.164 phone, LinkedIn slug, GitHub username
  - Textual similarity: full name (token sort ratio), job titles
  - Career history: overlapping or shared employer names
  - Skill overlap: Jaccard similarity across normalized skill sets
  - Location compatibility: city and country agreement

Match classifications:
  - high_confidence (score >= 0.85): safe for automated grouping
  - uncertain (0.50 <= score < 0.85): queued for human reviewer triage
  - non_match (score < 0.50): distinct individuals

Crucial safeguard: Conflicting hard identifiers (e.g. distinct confirmed emails
or distinct phones without overlap) immediately force score = 0.0 to prevent
destructive false merges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from rapidfuzz import fuzz

from data_transformer.schema.canonical import RawRecord
from data_transformer.normalizers import (
    normalize_email,
    normalize_phone,
    normalize_name,
    normalize_company,
)
from data_transformer.normalizers.skills import normalize_skill
from data_transformer.deduplication.matcher import (
    _linkedin_ids_from_records,
    _github_usernames_from_records,
)


@dataclass
class MatchSignal:
    signal_type: str
    weight: float
    score_contribution: float
    matched: bool
    value_a: str
    value_b: str
    explanation: str


@dataclass
class ExplainedMatchResult:
    record_a_id: Optional[str]
    record_b_id: Optional[str]
    raw_score: float
    score: float
    classification: str  # high_confidence | uncertain | non_match
    requires_review: bool
    signals: List[MatchSignal] = field(default_factory=list)
    explanation_summary: str = ""
    conflicting_hard_identifiers: bool = False


class ExplainableScorer:
    """
    Computes explainable match scores between pairs of candidates.
    Supports customizable weights and calibration thresholds.
    """

    DEFAULT_SIGNAL_WEIGHTS: Dict[str, float] = {
        "exact_email": 1.00,
        "exact_phone": 1.00,
        "exact_linkedin": 0.95,
        "exact_github": 0.95,
        "fuzzy_name_high": 0.70,     # ratio >= 90
        "fuzzy_name_medium": 0.45,   # ratio 80-89
        "fuzzy_name_low": 0.20,      # ratio 65-79
        "shared_employer": 0.25,
        "shared_skills": 0.20,
        "shared_location": 0.10,
    }

    def __init__(
        self,
        high_confidence_threshold: float = 0.85,
        uncertain_threshold: float = 0.50,
        signal_weights: Optional[Dict[str, float]] = None,
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.uncertain_threshold = uncertain_threshold
        self.signal_weights = dict(self.DEFAULT_SIGNAL_WEIGHTS)
        if signal_weights:
            self.signal_weights.update(signal_weights)

    def score(self, record_a: RawRecord, record_b: RawRecord) -> ExplainedMatchResult:
        signals: List[MatchSignal] = []

        emails_a = self._get_emails(record_a)
        emails_b = self._get_emails(record_b)
        phones_a = self._get_phones(record_a)
        phones_b = self._get_phones(record_b)

        # ── Safeguard: Detect conflicting hard identifiers ──────────────────
        has_conflicting_email = bool(emails_a and emails_b and emails_a.isdisjoint(emails_b))
        has_conflicting_phone = bool(phones_a and phones_b and phones_a.isdisjoint(phones_b))

        # ── 1. Exact email match ────────────────────────────────────────────
        email_overlap = emails_a & emails_b
        if email_overlap:
            ex = sorted(email_overlap)[0]
            w = self.signal_weights["exact_email"]
            signals.append(
                MatchSignal(
                    signal_type="exact_email",
                    weight=w,
                    score_contribution=w,
                    matched=True,
                    value_a=ex,
                    value_b=ex,
                    explanation=f"Exact verified email match: {ex}",
                )
            )
        elif emails_a and emails_b:
            signals.append(
                MatchSignal(
                    signal_type="exact_email",
                    weight=self.signal_weights["exact_email"],
                    score_contribution=0.0,
                    matched=False,
                    value_a=", ".join(sorted(emails_a)),
                    value_b=", ".join(sorted(emails_b)),
                    explanation="Conflicting distinct email addresses",
                )
            )

        # ── 2. Exact phone match ────────────────────────────────────────────
        phone_overlap = phones_a & phones_b
        if phone_overlap:
            px = sorted(phone_overlap)[0]
            w = self.signal_weights["exact_phone"]
            signals.append(
                MatchSignal(
                    signal_type="exact_phone",
                    weight=w,
                    score_contribution=w,
                    matched=True,
                    value_a=px,
                    value_b=px,
                    explanation=f"Exact verified phone match (E.164): {px}",
                )
            )
        elif phones_a and phones_b:
            signals.append(
                MatchSignal(
                    signal_type="exact_phone",
                    weight=self.signal_weights["exact_phone"],
                    score_contribution=0.0,
                    matched=False,
                    value_a=", ".join(sorted(phones_a)),
                    value_b=", ".join(sorted(phones_b)),
                    explanation="Conflicting distinct phone numbers",
                )
            )

        # ── 3. Exact profile links ──────────────────────────────────────────
        li_a = _linkedin_ids_from_records([record_a])
        li_b = _linkedin_ids_from_records([record_b])
        li_overlap = li_a & li_b
        if li_overlap:
            lx = sorted(li_overlap)[0]
            w = self.signal_weights["exact_linkedin"]
            signals.append(
                MatchSignal(
                    signal_type="exact_linkedin",
                    weight=w,
                    score_contribution=w,
                    matched=True,
                    value_a=lx,
                    value_b=lx,
                    explanation=f"Shared LinkedIn profile slug: {lx}",
                )
            )

        gh_a = _github_usernames_from_records([record_a])
        gh_b = _github_usernames_from_records([record_b])
        gh_overlap = gh_a & gh_b
        if gh_overlap:
            gx = sorted(gh_overlap)[0]
            w = self.signal_weights["exact_github"]
            signals.append(
                MatchSignal(
                    signal_type="exact_github",
                    weight=w,
                    score_contribution=w,
                    matched=True,
                    value_a=gx,
                    value_b=gx,
                    explanation=f"Shared GitHub handle: {gx}",
                )
            )

        # ── 4. Name similarity ──────────────────────────────────────────────
        name_a = normalize_name(record_a.fields.get("full_name") or "") or ""
        name_b = normalize_name(record_b.fields.get("full_name") or "") or ""

        if name_a and name_b and len(name_a) >= 3 and len(name_b) >= 3:
            ratio = fuzz.token_sort_ratio(name_a.lower(), name_b.lower())
            if ratio >= 90:
                w = self.signal_weights["fuzzy_name_high"]
                signals.append(
                    MatchSignal(
                        signal_type="fuzzy_name_high",
                        weight=w,
                        score_contribution=w,
                        matched=True,
                        value_a=name_a,
                        value_b=name_b,
                        explanation=f"Strong name similarity ({ratio}% match): '{name_a}' ≈ '{name_b}'",
                    )
                )
            elif ratio >= 80:
                w = self.signal_weights["fuzzy_name_medium"]
                signals.append(
                    MatchSignal(
                        signal_type="fuzzy_name_medium",
                        weight=w,
                        score_contribution=w,
                        matched=True,
                        value_a=name_a,
                        value_b=name_b,
                        explanation=f"Moderate name similarity ({ratio}% match): '{name_a}' ≈ '{name_b}'",
                    )
                )
            elif ratio >= 65:
                w = self.signal_weights["fuzzy_name_low"]
                signals.append(
                    MatchSignal(
                        signal_type="fuzzy_name_low",
                        weight=w,
                        score_contribution=w,
                        matched=True,
                        value_a=name_a,
                        value_b=name_b,
                        explanation=f"Partial name similarity ({ratio}% match): '{name_a}' vs '{name_b}'",
                    )
                )
            else:
                signals.append(
                    MatchSignal(
                        signal_type="fuzzy_name_high",
                        weight=self.signal_weights["fuzzy_name_high"],
                        score_contribution=0.0,
                        matched=False,
                        value_a=name_a,
                        value_b=name_b,
                        explanation=f"Different names ({ratio}% match): '{name_a}' vs '{name_b}'",
                    )
                )

        # ── 5. Shared employment history ────────────────────────────────────
        companies_a = self._get_companies(record_a)
        companies_b = self._get_companies(record_b)
        company_overlap = companies_a & companies_b
        if company_overlap:
            cx = sorted(company_overlap)[0]
            w = self.signal_weights["shared_employer"]
            signals.append(
                MatchSignal(
                    signal_type="shared_employer",
                    weight=w,
                    score_contribution=w,
                    matched=True,
                    value_a=cx,
                    value_b=cx,
                    explanation=f"Shared employer in career history: {cx}",
                )
            )

        # ── 6. Skill overlap ────────────────────────────────────────────────
        skills_a = self._get_skills(record_a)
        skills_b = self._get_skills(record_b)
        if skills_a and skills_b:
            shared_skills = skills_a & skills_b
            union_skills = skills_a | skills_b
            jaccard = len(shared_skills) / len(union_skills) if union_skills else 0.0
            if jaccard >= 0.35:
                w = self.signal_weights["shared_skills"]
                contrib = round(w * min(1.0, jaccard * 1.5), 3)
                signals.append(
                    MatchSignal(
                        signal_type="shared_skills",
                        weight=w,
                        score_contribution=contrib,
                        matched=True,
                        value_a=f"{len(skills_a)} skills",
                        value_b=f"{len(skills_b)} skills",
                        explanation=f"High skill profile overlap ({len(shared_skills)} shared skills: {', '.join(sorted(list(shared_skills))[:5])})",
                    )
                )

        # ── 7. Location compatibility ───────────────────────────────────────
        loc_a = record_a.fields.get("location")
        loc_b = record_b.fields.get("location")
        if isinstance(loc_a, dict) and isinstance(loc_b, dict):
            city_a = (loc_a.get("city") or "").strip().lower()
            city_b = (loc_b.get("city") or "").strip().lower()
            country_a = (loc_a.get("country") or "").strip().lower()
            country_b = (loc_b.get("country") or "").strip().lower()
            w = self.signal_weights["shared_location"]
            if city_a and city_b and city_a == city_b:
                signals.append(
                    MatchSignal(
                        signal_type="shared_location",
                        weight=w,
                        score_contribution=w,
                        matched=True,
                        value_a=city_a.title(),
                        value_b=city_b.title(),
                        explanation=f"Identical location city: {city_a.title()}",
                    )
                )
            elif country_a and country_b and country_a == country_b:
                signals.append(
                    MatchSignal(
                        signal_type="shared_location",
                        weight=w,
                        score_contribution=w * 0.5,
                        matched=True,
                        value_a=country_a.upper(),
                        value_b=country_b.upper(),
                        explanation=f"Same country of residence: {country_a.upper()}",
                    )
                )

        # ── 8. Calculate Final Score & Apply Safeguards ──────────────────────
        has_exact = any(
            s.matched
            and s.signal_type in ("exact_email", "exact_phone", "exact_linkedin", "exact_github")
            for s in signals
        )

        conflicting = False
        if has_conflicting_email or has_conflicting_phone:
            if not has_exact:
                # Conflicting contact information without an exact match overrides any fuzzy similarity
                conflicting = True
                final_score = 0.0
                raw_score = 0.0
            else:
                # One identifier matched while another differed (e.g. personal email vs work email)
                raw_score = sum(s.score_contribution for s in signals)
                final_score = min(1.0, raw_score)
        elif has_exact:
            raw_score = 1.0
            final_score = 1.0
        else:
            raw_score = sum(s.score_contribution for s in signals)
            final_score = min(1.0, raw_score)

        # ── 9. Match Classification ─────────────────────────────────────────
        if final_score >= self.high_confidence_threshold:
            classification = "high_confidence"
            requires_review = False
        elif final_score >= self.uncertain_threshold:
            classification = "uncertain"
            requires_review = True
        else:
            classification = "non_match"
            requires_review = False

        # ── 10. Human-readable explanation summary ──────────────────────────
        if conflicting:
            summary = "Match rejected: conflicting email or phone identifiers detected."
        else:
            matched_signals = [s for s in signals if s.matched]
            if matched_signals:
                top_signals = sorted(matched_signals, key=lambda s: s.score_contribution, reverse=True)[:3]
                summary = "Contributing match signals: " + "; ".join(s.explanation for s in top_signals)
            else:
                summary = "No matching signals identified between records."

        return ExplainedMatchResult(
            record_a_id=record_a.source_id,
            record_b_id=record_b.source_id,
            raw_score=round(raw_score, 4),
            score=round(final_score, 4),
            classification=classification,
            requires_review=requires_review,
            signals=signals,
            explanation_summary=summary,
            conflicting_hard_identifiers=conflicting,
        )

    def _get_emails(self, record: RawRecord) -> Set[str]:
        out = set()
        for e in record.fields.get("emails", []):
            if e:
                n = normalize_email(str(e))
                if n:
                    out.add(n)
        return out

    def _get_phones(self, record: RawRecord) -> Set[str]:
        out = set()
        for p in record.fields.get("phones", []):
            if p:
                n = normalize_phone(str(p))
                if n:
                    out.add(n)
        return out

    def _get_companies(self, record: RawRecord) -> Set[str]:
        out = set()
        for exp in record.fields.get("experience", []):
            if isinstance(exp, dict):
                c = normalize_company(exp.get("company") or "")
                if c:
                    out.add(c.lower())
        return out

    def _get_skills(self, record: RawRecord) -> Set[str]:
        out = set()
        for s in record.fields.get("skills", []):
            if s:
                ns = normalize_skill(str(s))
                if ns:
                    out.add(ns.lower())
        return out

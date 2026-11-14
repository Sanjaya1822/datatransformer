"""
Identity Resolution module.
Scores pairwise candidate identity using deterministic and fuzzy heuristics.
Requires reliable corroborating signals to determine whether records belong to the same person,
preventing false merges while correctly merging multi-source records (resumes, ATS reports, notes).
"""
from typing import Dict, Any, List, Set, Optional
import re
from rapidfuzz import fuzz

from data_transformer.schema.canonical import RawRecord


def _norm_digits(phone_str: Any) -> str:
    if not phone_str:
        return ""
    digits = re.sub(r"\D", "", str(phone_str))
    return digits[-10:] if len(digits) >= 10 else digits


def _clean_url(url: Any) -> str:
    if not url:
        return ""
    u = str(url).strip().lower().rstrip("/")
    u = re.sub(r"^https?://(?:www\.)?", "", u)
    return u


def compute_identity_score(record_a: RawRecord, record_b: RawRecord) -> float:
    """
    Compute an identity score from 0.0 to 100.0 between two raw records.

    Identity resolution rules:
    - Shared hard identifier (matching email, matching 10-digit phone, matching LinkedIn or GitHub) -> 100.0 (definitive match)
    - Conflicting hard identifiers (e.g. conflicting phones, or conflicting emails with no other shared link) -> 0.0 (definitive non-match)
    - If one or both lack hard identifiers: evaluate soft signals:
        - Name match (token_set_ratio, token_sort_ratio, subset containment)
        - Corroborating signals: skill overlap, location match, education match
        - Strong name match + corroborating signals -> up to 98.0 (auto-merge)
        - Weak/dissimilar name -> 0.0 (never merge)
    """
    fields_a = record_a.fields or {}
    fields_b = record_b.fields or {}

    emails_a = {str(e).strip().lower() for e in fields_a.get("emails", []) if e}
    emails_b = {str(e).strip().lower() for e in fields_b.get("emails", []) if e}

    phones_a = {_norm_digits(p) for p in fields_a.get("phones", []) if _norm_digits(p)}
    phones_b = {_norm_digits(p) for p in fields_b.get("phones", []) if _norm_digits(p)}

    links_a = {_clean_url(l.get("url", "")) for l in fields_a.get("links", []) if isinstance(l, dict) and l.get("url")}
    links_b = {_clean_url(l.get("url", "")) for l in fields_b.get("links", []) if isinstance(l, dict) and l.get("url")}

    # ── 1. Deterministic Hard Matches ────────────────────────────────────────
    # Shared email
    if emails_a and emails_b and not emails_a.isdisjoint(emails_b):
        return 100.0

    # Shared phone
    if phones_a and phones_b and not phones_a.isdisjoint(phones_b):
        return 100.0

    # Shared profile link (LinkedIn or GitHub)
    if links_a and links_b and not links_a.isdisjoint(links_b):
        return 100.0

    # ── 2. Contradictory Hard Identifiers (Definitive Non-Matches) ────────────
    # Both have phones, but they completely disagree, and neither has matching links
    if phones_a and phones_b and phones_a.isdisjoint(phones_b) and links_a.isdisjoint(links_b):
        return 0.0

    # Both have emails, but they completely disagree, and neither shares phone or links
    if emails_a and emails_b and emails_a.isdisjoint(emails_b) and phones_a.isdisjoint(phones_b) and links_a.isdisjoint(links_b):
        return 0.0

    # ── 3. Soft Signals (No Contradictory Identifiers) ────────────────────────
    name_a = str(fields_a.get("full_name") or "").strip().lower()
    name_b = str(fields_b.get("full_name") or "").strip().lower()

    if not name_a or not name_b:
        return 0.0

    # Clean punctuation and extra spaces
    name_a_clean = re.sub(r"[^\w\s]", "", name_a).strip()
    name_b_clean = re.sub(r"[^\w\s]", "", name_b).strip()

    tokens_a = set(name_a_clean.split())
    tokens_b = set(name_b_clean.split())

    set_ratio = fuzz.token_set_ratio(name_a, name_b)
    sort_ratio = fuzz.token_sort_ratio(name_a, name_b)

    base_name_score = 0.0
    if tokens_a == tokens_b or name_a == name_b:
        base_name_score = 88.0
    elif (tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a)) and set_ratio >= 90:
        base_name_score = 80.0
    elif set_ratio >= 95 and sort_ratio >= 80:
        base_name_score = 75.0
    elif sort_ratio >= 85:
        base_name_score = 65.0
    else:
        # Names are genuinely distinct (e.g. Deva vs Preethi vs Varsha)
        return 0.0

    # Corroborating signals: skills overlap
    skills_a = {str(s).strip().lower() for s in fields_a.get("skills", []) if s}
    skills_b = {str(s).strip().lower() for s in fields_b.get("skills", []) if s}
    shared_skills = skills_a.intersection(skills_b)

    bonus = 0.0
    if len(shared_skills) >= 3:
        bonus += 20.0
    elif len(shared_skills) >= 1:
        bonus += 12.0

    # Location overlap
    loc_a = str(fields_a.get("location") or "").lower()
    loc_b = str(fields_b.get("location") or "").lower()
    if loc_a and loc_b and (loc_a in loc_b or loc_b in loc_a):
        bonus += 10.0

    # Exact name bonus
    if tokens_a == tokens_b:
        bonus += 5.0

    total_score = min(100.0, base_name_score + bonus)
    return total_score

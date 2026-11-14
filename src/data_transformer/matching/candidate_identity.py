"""
Candidate Identity Resolution, Accurate Attribute Comparison, and Profile Merging.

Provides robust, evidence-based candidate identity matching:
- Normalizes names, emails, phone numbers, locations, skills, and experience.
- Uses email and normalized phone number as strong identity signals.
- Uses name similarity and career/skill attributes as supporting signals.
- Handles missing values gracefully without penalizing them as mismatches or conflicts.
- Prevents false merges based only on similar names.
- Ignores internal UUIDs and metadata during comparison and similarity scoring.
- Computes weighted similarity score and granular field-by-field differences with the 6 required statuses:
  'Exact Match', 'Partial Match', 'Conflict', 'Missing in Candidate A', 'Missing in Candidate B', 'Insufficient Evidence'.
- Merges candidate profiles cleanly while preserving all contributing sources and attribute provenance.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from rapidfuzz import fuzz

from data_transformer.normalizers import (
    normalize_email,
    normalize_phone,
    normalize_name,
    normalize_company,
)
from data_transformer.normalizers.skills import normalize_skill


def _clean_phone_digits(val: Any) -> str:
    """Extract standard digits for phone comparison (e.g. 10 trailing digits)."""
    if not val:
        return ""
    digits = re.sub(r"\D", "", str(val))
    if len(digits) >= 10:
        return digits[-10:]
    return digits


def extract_candidate_attributes(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes and extracts structured, comparable attributes from a candidate profile dictionary.
    Handles various schemas (canonical, projected, raw) seamlessly.
    """
    if not profile or not isinstance(profile, dict):
        return {
            "full_name": "",
            "emails": [],
            "phones": [],
            "location": "",
            "skills": [],
            "experience": [],
            "education": [],
            "certifications": [],
            "links": [],
            "sources": [],
        }

    # 1. Full name
    raw_name = profile.get("full_name") or profile.get("name") or ""
    if isinstance(raw_name, dict):
        raw_name = raw_name.get("value") or ""
    full_name = normalize_name(str(raw_name or "").strip()) or str(raw_name or "").strip()

    # 2. Emails
    email_set = set()
    raw_emails = profile.get("emails") or []
    if isinstance(raw_emails, list):
        for e in raw_emails:
            val = e.get("value") if isinstance(e, dict) else str(e)
            if val:
                norm = normalize_email(str(val))
                if norm:
                    email_set.add(norm)
    # Also check single email / primary_email
    for k in ("primary_email", "email"):
        val = profile.get(k)
        if val:
            v_str = val.get("value") if isinstance(val, dict) else str(val)
            norm = normalize_email(v_str)
            if norm:
                email_set.add(norm)
    emails = sorted(list(email_set))

    # 3. Phones
    phone_set = set()
    raw_phones = profile.get("phones") or []
    if isinstance(raw_phones, list):
        for p in raw_phones:
            val = p.get("value") if isinstance(p, dict) else str(p)
            digits = _clean_phone_digits(val)
            if digits:
                phone_set.add(digits)
    for k in ("phone", "primary_phone"):
        val = profile.get(k)
        if val:
            v_str = val.get("value") if isinstance(val, dict) else str(val)
            digits = _clean_phone_digits(v_str)
            if digits:
                phone_set.add(digits)
    phones = sorted(list(phone_set))

    # 4. Location
    loc_val = profile.get("location")
    location_str = ""
    if isinstance(loc_val, dict):
        parts = [
            loc_val.get("city"),
            loc_val.get("state"),
            loc_val.get("country"),
        ]
        location_str = ", ".join([str(p).strip() for p in parts if p and str(p).strip()])
        if not location_str and loc_val.get("formatted"):
            location_str = str(loc_val.get("formatted")).strip()
    elif isinstance(loc_val, str):
        location_str = loc_val.strip()

    # 5. Skills
    raw_skills = profile.get("skills") or []
    skill_set = set()
    if isinstance(raw_skills, list):
        for s in raw_skills:
            sname = s.get("name") if isinstance(s, dict) else str(s)
            if sname:
                norm = normalize_skill(str(sname).strip())
                if norm:
                    skill_set.add(norm.title())
                else:
                    skill_set.add(str(sname).strip().title())
    skills = sorted(list(skill_set))

    # 6. Experience
    raw_exp = profile.get("experience") or []
    experience: List[Dict[str, Any]] = []
    if isinstance(raw_exp, list):
        for item in raw_exp:
            if isinstance(item, dict):
                company = normalize_company(item.get("company") or "") or item.get("company") or ""
                title = str(item.get("title") or "").strip()
                summary = str(item.get("summary") or item.get("description") or "").strip()
                experience.append({
                    "company": company,
                    "title": title,
                    "start": item.get("start") or item.get("start_date") or "",
                    "end": item.get("end") or item.get("end_date") or "",
                    "summary": summary,
                })

    # 7. Education
    raw_edu = profile.get("education") or []
    education: List[Dict[str, Any]] = []
    if isinstance(raw_edu, list):
        for item in raw_edu:
            if isinstance(item, dict):
                inst = str(item.get("institution") or item.get("school") or item.get("university") or "").strip()
                degree = str(item.get("degree") or "").strip()
                field_study = str(item.get("field") or item.get("field_of_study") or "").strip()
                education.append({
                    "institution": inst,
                    "degree": degree,
                    "field": field_study,
                })

    # 8. Certifications
    raw_certs = profile.get("certifications") or []
    certifications: List[str] = []
    if isinstance(raw_certs, list):
        for c in raw_certs:
            cname = c.get("name") if isinstance(c, dict) else str(c)
            if cname and str(cname).strip():
                certifications.append(str(cname).strip())

    # 9. Links
    raw_links = profile.get("links") or []
    links: List[str] = []
    if isinstance(raw_links, list):
        for lk in raw_links:
            url = lk.get("url") if isinstance(lk, dict) else str(lk)
            if url and str(url).strip():
                links.append(str(url).strip().lower())

    # 10. Contributing sources
    sources_merged = profile.get("sources_merged") or []
    if not sources_merged and isinstance(profile.get("merge_summary"), dict):
        sources_merged = profile.get("merge_summary", {}).get("sources_merged", [])
    if isinstance(sources_merged, str):
        sources_merged = [sources_merged]
    sources = [str(s).strip() for s in (sources_merged or []) if str(s).strip()]

    return {
        "full_name": full_name,
        "emails": emails,
        "phones": phones,
        "location": location_str,
        "skills": skills,
        "experience": experience,
        "education": education,
        "certifications": certifications,
        "links": links,
        "sources": sources,
    }


def evaluate_candidate_identity(canon_a: Dict[str, Any], canon_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates whether two records belong to the same real candidate using tiered evidence.
    - Uses normalized email and phone as strong identity signals.
    - Uses name similarity and career attributes as supporting signals.
    - Handles missing values without automatically treating them as mismatches.
    - Does NOT merge candidates based only on similar names.
    - Never uses internal UUIDs or metadata.
    """
    attr_a = extract_candidate_attributes(canon_a)
    attr_b = extract_candidate_attributes(canon_b)

    emails_a = set(attr_a["emails"])
    emails_b = set(attr_b["emails"])
    phones_a = set(attr_a["phones"])
    phones_b = set(attr_b["phones"])

    shared_emails = emails_a & emails_b
    shared_phones = phones_a & phones_b

    # Name similarity analysis
    name_a = attr_a["full_name"].strip().lower()
    name_b = attr_b["full_name"].strip().lower()

    name_ratio = 0
    if name_a and name_b:
        name_ratio = max(
            fuzz.token_set_ratio(name_a, name_b),
            fuzz.token_sort_ratio(name_a, name_b),
        )

    # 1. Hard identity overlap
    if shared_emails or shared_phones:
        # Verified contact match
        # Check if names are conflicting or compatible
        if name_a and name_b and name_ratio < 40:
            # Different names despite shared contact (e.g. shared company switchboard or family email)
            return {
                "is_same_candidate": False,
                "verdict": "Uncertain / Contradictory Evidence",
                "explanation": "Shared contact identifier detected, but candidate names strongly diverge.",
                "strong_signal": False,
            }
        matched_ids = []
        if shared_emails:
            matched_ids.append(f"email '{list(shared_emails)[0]}'")
        if shared_phones:
            matched_ids.append(f"phone '{list(shared_phones)[0]}'")
        return {
            "is_same_candidate": True,
            "verdict": "Same Candidate",
            "explanation": f"High-confidence match: shared verified {', '.join(matched_ids)}.",
            "strong_signal": True,
        }

    # 2. Hard identity conflict
    # Both records have emails, but they have zero overlap, AND both have phones that have zero overlap
    if emails_a and emails_b and emails_a.isdisjoint(emails_b) and phones_a and phones_b and phones_a.isdisjoint(phones_b):
        return {
            "is_same_candidate": False,
            "verdict": "Different Candidates",
            "explanation": "Distinct candidates: conflicting email addresses and conflicting phone numbers.",
            "strong_signal": True,
        }

    # 3. Soft corroborating signals (when hard IDs are missing in one or both records)
    # Check if names are genuinely distinct
    if name_a and name_b and name_ratio < 70:
        return {
            "is_same_candidate": False,
            "verdict": "Different Candidates",
            "explanation": f"Distinct candidate identities with different names ('{attr_a['full_name']}' vs '{attr_b['full_name']}').",
            "strong_signal": False,
        }

    # If name similarity is high (>=80), check for supporting signals
    if name_ratio >= 80:
        # Check skill overlap
        skills_a = set([s.lower() for s in attr_a["skills"]])
        skills_b = set([s.lower() for s in attr_b["skills"]])
        shared_skills = skills_a & skills_b

        # Check companies overlap
        companies_a = set([e["company"].lower() for e in attr_a["experience"] if e.get("company")])
        companies_b = set([e["company"].lower() for e in attr_b["experience"] if e.get("company")])
        shared_companies = companies_a & companies_b

        # Check education overlap
        edu_a = set([e["institution"].lower() for e in attr_a["education"] if e.get("institution")])
        edu_b = set([e["institution"].lower() for e in attr_b["education"] if e.get("institution")])
        shared_edu = edu_a & edu_b

        # If corroborated by employers, education, or significant skill overlap (>=3)
        if shared_companies or shared_edu or len(shared_skills) >= 3:
            reasons = []
            if shared_companies:
                reasons.append(f"shared employer ({', '.join(list(shared_companies)[:2])})")
            if shared_edu:
                reasons.append(f"same institution ({', '.join(list(shared_edu)[:1])})")
            if len(shared_skills) >= 3:
                reasons.append(f"{len(shared_skills)} overlapping skills")
            return {
                "is_same_candidate": True,
                "verdict": "Same Candidate (Corroborated)",
                "explanation": f"Strong name alignment corroborated by {', and '.join(reasons)}.",
                "strong_signal": False,
            }
        else:
            # Rule: Do not merge candidates based ONLY on similar names without corroboration
            return {
                "is_same_candidate": False,
                "verdict": "Insufficient Evidence",
                "explanation": "Name similarity observed, but insufficient corroborating identity evidence to confirm same candidate.",
                "strong_signal": False,
            }

    return {
        "is_same_candidate": False,
        "verdict": "Different Candidates",
        "explanation": "Independent candidate records with no shared identity markers.",
        "strong_signal": False,
    }


def compare_candidates_detailed(canon_a: Dict[str, Any], canon_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs field-by-field comparison and computes a transparent weighted similarity score.
    Returns:
      - similarity_score (0.0 to 1.0)
      - similarity_percentage (0.0 to 100.0)
      - classification ('high_confidence', 'uncertain', 'non_match')
      - identity_evaluation
      - score_breakdown (list of attribute weights and contributions)
      - field_diffs (dict of meaningful fields with the 6 required statuses and explanations)
    """
    attr_a = extract_candidate_attributes(canon_a)
    attr_b = extract_candidate_attributes(canon_b)
    ident_eval = evaluate_candidate_identity(canon_a, canon_b)

    field_diffs: Dict[str, Dict[str, Any]] = {}
    score_components: List[Dict[str, Any]] = []

    # ── 1. Full Name ────────────────────────────────────────────────────────
    val_a_name = attr_a["full_name"]
    val_b_name = attr_b["full_name"]
    name_score = 0.0
    if not val_a_name and not val_b_name:
        status = "Insufficient Evidence"
        expl = "Neither profile has a name recorded."
        name_score = 0.5
    elif not val_a_name:
        status = "Missing in Candidate A"
        expl = f"Candidate A has no name; Candidate B is '{val_b_name}'."
        name_score = 0.0
    elif not val_b_name:
        status = "Missing in Candidate B"
        expl = f"Candidate A is '{val_a_name}'; Candidate B has no name."
        name_score = 0.0
    else:
        norm_a = val_a_name.lower().strip()
        norm_b = val_b_name.lower().strip()
        ratio = max(
            fuzz.token_set_ratio(norm_a, norm_b),
            fuzz.token_sort_ratio(norm_a, norm_b),
        )
        if norm_a == norm_b or ratio >= 98:
            status = "Exact Match"
            expl = f"Identical normalized name: '{val_a_name}'."
            name_score = 1.0
        elif ratio >= 75:
            status = "Partial Match"
            expl = f"Close name variant ({ratio}% similarity): '{val_a_name}' vs '{val_b_name}'."
            name_score = ratio / 100.0
        else:
            status = "Conflict"
            expl = f"Different names: '{val_a_name}' vs '{val_b_name}'."
            name_score = ratio / 100.0

    field_diffs["full_name"] = {
        "field_label": "Full Name",
        "value_a": val_a_name or "--",
        "value_b": val_b_name or "--",
        "status": status,
        "differs": status in ("Conflict", "Partial Match", "Missing in Candidate A", "Missing in Candidate B"),
        "explanation": expl,
    }
    score_components.append({
        "attribute": "Full Name",
        "weight": 0.20,
        "score": round(name_score, 3),
        "contribution": round(0.20 * name_score, 3),
        "status": status,
        "explanation": expl,
    })

    # ── 2. Email ────────────────────────────────────────────────────────────
    emails_a = set(attr_a["emails"])
    emails_b = set(attr_b["emails"])
    email_score = 0.0
    if not emails_a and not emails_b:
        status = "Insufficient Evidence"
        expl = "No email addresses recorded for either candidate."
        email_score = 0.5  # Neutral when both missing
    elif not emails_a:
        status = "Missing in Candidate A"
        expl = f"Candidate A has no email; Candidate B has {', '.join(emails_b)}."
        email_score = 0.0
    elif not emails_b:
        status = "Missing in Candidate B"
        expl = f"Candidate A has {', '.join(emails_a)}; Candidate B has no email."
        email_score = 0.0
    else:
        shared_e = emails_a & emails_b
        if emails_a == emails_b:
            status = "Exact Match"
            expl = f"Exact verified email match: {', '.join(shared_e)}."
            email_score = 1.0
        elif shared_e:
            status = "Partial Match"
            expl = f"Shared verified email ({', '.join(shared_e)}), with additional secondary addresses."
            email_score = 0.95
        else:
            status = "Conflict"
            expl = f"Distinct email addresses ({', '.join(emails_a)} vs {', '.join(emails_b)})."
            email_score = 0.0

    field_diffs["email"] = {
        "field_label": "Email",
        "value_a": ", ".join(attr_a["emails"]) if attr_a["emails"] else "--",
        "value_b": ", ".join(attr_b["emails"]) if attr_b["emails"] else "--",
        "status": status,
        "differs": status in ("Conflict", "Partial Match", "Missing in Candidate A", "Missing in Candidate B"),
        "explanation": expl,
    }
    score_components.append({
        "attribute": "Email Address",
        "weight": 0.25,
        "score": round(email_score, 3),
        "contribution": round(0.25 * email_score, 3),
        "status": status,
        "explanation": expl,
    })

    # ── 3. Phone Number ─────────────────────────────────────────────────────
    phones_a = set(attr_a["phones"])
    phones_b = set(attr_b["phones"])
    phone_score = 0.0
    if not phones_a and not phones_b:
        status = "Insufficient Evidence"
        expl = "No phone numbers recorded for either candidate."
        phone_score = 0.5  # Neutral when both missing
    elif not phones_a:
        status = "Missing in Candidate A"
        expl = f"Candidate A has no phone; Candidate B has {', '.join(phones_b)}."
        phone_score = 0.0
    elif not phones_b:
        status = "Missing in Candidate B"
        expl = f"Candidate A has {', '.join(phones_a)}; Candidate B has no phone."
        phone_score = 0.0
    else:
        shared_p = phones_a & phones_b
        if phones_a == phones_b:
            status = "Exact Match"
            expl = f"Exact verified phone match: {', '.join(shared_p)}."
            phone_score = 1.0
        elif shared_p:
            status = "Partial Match"
            expl = f"Shared phone ({', '.join(shared_p)}) with alternative numbers."
            phone_score = 0.95
        else:
            status = "Conflict"
            expl = f"Distinct phone numbers ({', '.join(phones_a)} vs {', '.join(phones_b)})."
            phone_score = 0.0

    field_diffs["phone"] = {
        "field_label": "Phone Number",
        "value_a": ", ".join(attr_a["phones"]) if attr_a["phones"] else "--",
        "value_b": ", ".join(attr_b["phones"]) if attr_b["phones"] else "--",
        "status": status,
        "differs": status in ("Conflict", "Partial Match", "Missing in Candidate A", "Missing in Candidate B"),
        "explanation": expl,
    }
    score_components.append({
        "attribute": "Phone Number",
        "weight": 0.20,
        "score": round(phone_score, 3),
        "contribution": round(0.20 * phone_score, 3),
        "status": status,
        "explanation": expl,
    })

    # ── 4. Location ─────────────────────────────────────────────────────────
    loc_a = attr_a["location"].strip()
    loc_b = attr_b["location"].strip()
    loc_score = 0.0
    if not loc_a and not loc_b:
        status = "Insufficient Evidence"
        expl = "No location recorded for either candidate."
        loc_score = 0.5
    elif not loc_a:
        status = "Missing in Candidate A"
        expl = f"Candidate A missing location; Candidate B located in '{loc_b}'."
        loc_score = 0.0
    elif not loc_b:
        status = "Missing in Candidate B"
        expl = f"Candidate A located in '{loc_a}'; Candidate B missing location."
        loc_score = 0.0
    else:
        norm_loc_a = loc_a.lower()
        norm_loc_b = loc_b.lower()
        if norm_loc_a == norm_loc_b:
            status = "Exact Match"
            expl = f"Identical location: '{loc_a}'."
            loc_score = 1.0
        elif norm_loc_a in norm_loc_b or norm_loc_b in norm_loc_a or fuzz.token_set_ratio(norm_loc_a, norm_loc_b) >= 80:
            status = "Partial Match"
            expl = f"Compatible regional location: '{loc_a}' ≈ '{loc_b}'."
            loc_score = 0.8
        else:
            status = "Conflict"
            expl = f"Different locations: '{loc_a}' vs '{loc_b}'."
            loc_score = 0.0

    field_diffs["location"] = {
        "field_label": "Location",
        "value_a": loc_a or "--",
        "value_b": loc_b or "--",
        "status": status,
        "differs": status in ("Conflict", "Partial Match", "Missing in Candidate A", "Missing in Candidate B"),
        "explanation": expl,
    }

    # ── 5. Skills ───────────────────────────────────────────────────────────
    skills_a = set([s.lower() for s in attr_a["skills"]])
    skills_b = set([s.lower() for s in attr_b["skills"]])
    skills_score = 0.0
    if not skills_a and not skills_b:
        status = "Insufficient Evidence"
        expl = "No skills listed for either candidate."
        skills_score = 0.5
    elif not skills_a:
        status = "Missing in Candidate A"
        expl = f"Candidate A has no recorded skills; Candidate B has {len(skills_b)} skills."
        skills_score = 0.0
    elif not skills_b:
        status = "Missing in Candidate B"
        expl = f"Candidate A has {len(skills_a)} skills; Candidate B has no recorded skills."
        skills_score = 0.0
    else:
        shared_s = skills_a & skills_b
        union_s = skills_a | skills_b
        jaccard = len(shared_s) / len(union_s) if union_s else 0.0
        overlap_coeff = len(shared_s) / min(len(skills_a), len(skills_b)) if min(len(skills_a), len(skills_b)) > 0 else 0.0

        if skills_a == skills_b:
            status = "Exact Match"
            expl = f"Identical set of {len(skills_a)} skills."
            skills_score = 1.0
        elif len(shared_s) > 0:
            status = "Partial Match"
            sample_shared = ", ".join(sorted([s.title() for s in list(shared_s)[:5]]))
            expl = f"{len(shared_s)} shared skills ({sample_shared}); {len(skills_a - shared_s)} unique to A, {len(skills_b - shared_s)} unique to B."
            skills_score = round(max(jaccard, overlap_coeff * 0.85), 3)
        else:
            status = "Conflict"
            expl = f"Completely distinct skill profiles with 0 common skills ({len(skills_a)} vs {len(skills_b)} skills)."
            skills_score = 0.0

    field_diffs["skills"] = {
        "field_label": "Skills",
        "value_a": f"{len(attr_a['skills'])} skills ({', '.join(attr_a['skills'][:6])}{'...' if len(attr_a['skills']) > 6 else ''})" if attr_a["skills"] else "--",
        "value_b": f"{len(attr_b['skills'])} skills ({', '.join(attr_b['skills'][:6])}{'...' if len(attr_b['skills']) > 6 else ''})" if attr_b["skills"] else "--",
        "status": status,
        "differs": status in ("Conflict", "Partial Match", "Missing in Candidate A", "Missing in Candidate B"),
        "explanation": expl,
    }
    score_components.append({
        "attribute": "Skills Overlap",
        "weight": 0.15,
        "score": round(skills_score, 3),
        "contribution": round(0.15 * skills_score, 3),
        "status": status,
        "explanation": expl,
    })

    # ── 6. Work Experience ──────────────────────────────────────────────────
    exp_a = attr_a["experience"]
    exp_b = attr_b["experience"]
    exp_score = 0.0
    if not exp_a and not exp_b:
        status = "Insufficient Evidence"
        expl = "No employment history records present in either candidate profile."
        exp_score = 0.5
    elif not exp_a:
        status = "Missing in Candidate A"
        expl = f"Candidate A has no employment records; Candidate B has {len(exp_b)} role(s)."
        exp_score = 0.0
    elif not exp_b:
        status = "Missing in Candidate B"
        expl = f"Candidate A has {len(exp_a)} role(s); Candidate B has no employment records."
        exp_score = 0.0
    else:
        comps_a = set([e["company"].lower().strip() for e in exp_a if e.get("company")])
        comps_b = set([e["company"].lower().strip() for e in exp_b if e.get("company")])
        shared_c = comps_a & comps_b
        if comps_a and comps_b and comps_a == comps_b:
            status = "Exact Match"
            expl = f"Identical employer history: {', '.join([c.title() for c in comps_a])}."
            exp_score = 1.0
        elif shared_c:
            status = "Partial Match"
            expl = f"Shared career experience at: {', '.join([c.title() for c in shared_c])}."
            exp_score = 0.8
        else:
            status = "Conflict"
            expl = f"Different employers recorded ({', '.join([c.title() for c in comps_a]) or 'None'} vs {', '.join([c.title() for c in comps_b]) or 'None'})."
            exp_score = 0.0

    field_diffs["experience"] = {
        "field_label": "Work Experience",
        "value_a": f"{len(exp_a)} role(s) ({', '.join([e['company'] for e in exp_a if e.get('company')])})" if exp_a else "--",
        "value_b": f"{len(exp_b)} role(s) ({', '.join([e['company'] for e in exp_b if e.get('company')])})" if exp_b else "--",
        "status": status,
        "differs": status in ("Conflict", "Partial Match", "Missing in Candidate A", "Missing in Candidate B"),
        "explanation": expl,
    }
    score_components.append({
        "attribute": "Work Experience",
        "weight": 0.10,
        "score": round(exp_score, 3),
        "contribution": round(0.10 * exp_score, 3),
        "status": status,
        "explanation": expl,
    })

    # ── 7. Education ────────────────────────────────────────────────────────
    edu_a = attr_a["education"]
    edu_b = attr_b["education"]
    edu_score = 0.0
    if not edu_a and not edu_b:
        status = "Insufficient Evidence"
        expl = "No educational records present in either candidate profile."
        edu_score = 0.5
    elif not edu_a:
        status = "Missing in Candidate A"
        expl = "Candidate A has no education records."
        edu_score = 0.0
    elif not edu_b:
        status = "Missing in Candidate B"
        expl = "Candidate B has no education records."
        edu_score = 0.0
    else:
        inst_a = set([e["institution"].lower().strip() for e in edu_a if e.get("institution")])
        inst_b = set([e["institution"].lower().strip() for e in edu_b if e.get("institution")])
        shared_inst = inst_a & inst_b
        if inst_a and inst_b and inst_a == inst_b:
            status = "Exact Match"
            expl = f"Identical institution(s): {', '.join([i.title() for i in inst_a])}."
            edu_score = 1.0
        elif shared_inst:
            status = "Partial Match"
            expl = f"Shared educational background at {', '.join([i.title() for i in shared_inst])}."
            edu_score = 0.85
        else:
            status = "Conflict"
            expl = "Different educational institutions recorded."
            edu_score = 0.0

    field_diffs["education"] = {
        "field_label": "Education",
        "value_a": "; ".join([f"{e.get('degree','')} {e.get('institution','')}".strip() for e in edu_a]) if edu_a else "--",
        "value_b": "; ".join([f"{e.get('degree','')} {e.get('institution','')}".strip() for e in edu_b]) if edu_b else "--",
        "status": status,
        "differs": status in ("Conflict", "Partial Match", "Missing in Candidate A", "Missing in Candidate B"),
        "explanation": expl,
    }
    score_components.append({
        "attribute": "Education & Location",
        "weight": 0.10,
        "score": round(max(edu_score, loc_score), 3),
        "contribution": round(0.10 * max(edu_score, loc_score), 3),
        "status": status,
        "explanation": expl,
    })

    # ── 8. Certifications ───────────────────────────────────────────────────
    certs_a = set([c.lower() for c in attr_a["certifications"]])
    certs_b = set([c.lower() for c in attr_b["certifications"]])
    if not certs_a and not certs_b:
        status = "Insufficient Evidence"
        expl = "No certifications recorded for either candidate."
    elif not certs_a:
        status = "Missing in Candidate A"
        expl = f"Candidate A has no certifications; Candidate B has {len(certs_b)}."
    elif not certs_b:
        status = "Missing in Candidate B"
        expl = f"Candidate A has {len(certs_a)} certification(s); Candidate B has none."
    else:
        shared_c = certs_a & certs_b
        if certs_a == certs_b:
            status = "Exact Match"
            expl = f"Matching certifications: {', '.join(attr_a['certifications'])}."
        elif shared_c:
            status = "Partial Match"
            expl = f"{len(shared_c)} shared certification(s)."
        else:
            status = "Conflict"
            expl = "Different certifications held by each candidate."

    field_diffs["certifications"] = {
        "field_label": "Certifications",
        "value_a": ", ".join(attr_a["certifications"]) if attr_a["certifications"] else "--",
        "value_b": ", ".join(attr_b["certifications"]) if attr_b["certifications"] else "--",
        "status": status,
        "differs": status in ("Conflict", "Partial Match", "Missing in Candidate A", "Missing in Candidate B"),
        "explanation": expl,
    }

    # ── 9. Source Records ───────────────────────────────────────────────────
    srcs_a = set(attr_a["sources"])
    srcs_b = set(attr_b["sources"])
    if not srcs_a and not srcs_b:
        status = "Insufficient Evidence"
        expl = "No source document provenance tracked."
    elif srcs_a == srcs_b:
        status = "Exact Match"
        expl = f"Both candidates derived from identical source documents ({', '.join(srcs_a)})."
    elif srcs_a & srcs_b:
        status = "Partial Match"
        expl = f"Partially overlapping source documents ({', '.join(srcs_a & srcs_b)})."
    else:
        status = "Conflict"
        expl = f"Candidate A sourced from {', '.join(srcs_a) or '--'}; Candidate B sourced from {', '.join(srcs_b) or '--'}."

    field_diffs["source_records"] = {
        "field_label": "Source Records",
        "value_a": ", ".join(attr_a["sources"]) if attr_a["sources"] else "--",
        "value_b": ", ".join(attr_b["sources"]) if attr_b["sources"] else "--",
        "status": status,
        "differs": status != "Exact Match",
        "explanation": expl,
    }

    # ── Compute Weighted Final Score ────────────────────────────────────────
    total_weighted = sum(c["contribution"] for c in score_components)
    total_weights = sum(c["weight"] for c in score_components)

    raw_similarity = total_weighted / total_weights if total_weights > 0 else 0.0

    # If identity evaluation confirms they are the same candidate via verified hard identifiers,
    # ensure similarity reflects that strong identity alignment.
    if ident_eval["is_same_candidate"] and ident_eval.get("strong_signal"):
        similarity = max(0.85, raw_similarity)
    else:
        similarity = raw_similarity

    similarity_pct = round(similarity * 100.0, 1)

    if similarity >= 0.85 or (ident_eval["is_same_candidate"] and ident_eval.get("strong_signal")):
        classification = "high_confidence"
    elif similarity >= 0.50:
        classification = "uncertain"
    else:
        classification = "non_match"

    overall_explanation = (
        f"{ident_eval['verdict']}: {ident_eval['explanation']} "
        f"Overall attribute similarity: {similarity_pct}%."
    )

    return {
        "match_score": round(similarity, 4),
        "similarity_percentage": similarity_pct,
        "classification": classification,
        "identity_verdict": ident_eval["verdict"],
        "is_same_candidate": ident_eval["is_same_candidate"],
        "identity_explanation": ident_eval["explanation"],
        "explanation": overall_explanation,
        "score_breakdown": score_components,
        "field_diffs": field_diffs,
    }


def merge_canonical_profiles(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges valid attributes of an incoming candidate profile into an existing canonical profile.
    - Preserves all original source records.
    - Tracks source of merged attributes.
    - Resolves conflicts using completeness & priority.
    - Never silently discards conflicting information.
    """
    merged = dict(existing)

    # 1. Full name: prefer longer / title-cased name
    name_ex = existing.get("full_name") or ""
    if isinstance(name_ex, dict):
        name_ex = name_ex.get("value") or ""
    name_in = incoming.get("full_name") or ""
    if isinstance(name_in, dict):
        name_in = name_in.get("value") or ""

    if not name_ex and name_in:
        merged["full_name"] = name_in
    elif name_ex and name_in:
        # Choose the more complete name (more tokens, longer length, title-cased)
        if len(str(name_in).split()) > len(str(name_ex).split()) or (len(str(name_in)) > len(str(name_ex)) and str(name_in).istitle()):
            merged["full_name"] = name_in
        else:
            merged["full_name"] = name_ex

    # 2. Emails: deduplicate by normalized value, preserve is_primary
    existing_emails = existing.get("emails") or []
    if not existing_emails and existing.get("primary_email"):
        pe = existing.get("primary_email")
        existing_emails = [pe if isinstance(pe, dict) else {"value": str(pe), "is_primary": True}]

    incoming_emails = incoming.get("emails") or []
    if not incoming_emails and incoming.get("primary_email"):
        pe = incoming.get("primary_email")
        incoming_emails = [pe if isinstance(pe, dict) else {"value": str(pe), "is_primary": True}]

    combined_emails: List[Dict[str, Any]] = []
    seen_emails: Set[str] = set()

    for e in existing_emails + incoming_emails:
        val = e.get("value") if isinstance(e, dict) else str(e)
        norm = normalize_email(str(val or ""))
        if norm and norm not in seen_emails:
            seen_emails.add(norm)
            is_pri = e.get("is_primary", False) if isinstance(e, dict) else False
            combined_emails.append({"value": norm, "is_primary": is_pri or len(combined_emails) == 0})

    if combined_emails:
        merged["emails"] = combined_emails
        merged["primary_email"] = combined_emails[0]["value"]

    # 3. Phones: deduplicate by normalized 10-digits
    existing_phones = existing.get("phones") or []
    if not existing_phones and existing.get("phone"):
        pp = existing.get("phone")
        existing_phones = [pp if isinstance(pp, dict) else {"value": str(pp), "is_primary": True}]

    incoming_phones = incoming.get("phones") or []
    if not incoming_phones and incoming.get("phone"):
        pp = incoming.get("phone")
        incoming_phones = [pp if isinstance(pp, dict) else {"value": str(pp), "is_primary": True}]

    combined_phones: List[Dict[str, Any]] = []
    seen_phones: Set[str] = set()

    for p in existing_phones + incoming_phones:
        val = p.get("value") if isinstance(p, dict) else str(p)
        digits = _clean_phone_digits(val)
        if digits and digits not in seen_phones:
            seen_phones.add(digits)
            is_pri = p.get("is_primary", False) if isinstance(p, dict) else False
            combined_phones.append({"value": str(val), "is_primary": is_pri or len(combined_phones) == 0})
        elif digits and str(val).startswith("+91"):
            # Upgrade any previously recorded +1 variant to verified +91
            for cp in combined_phones:
                if _clean_phone_digits(cp["value"]) == digits and cp["value"].startswith("+1"):
                    cp["value"] = str(val)

    if combined_phones:
        merged["phones"] = combined_phones
        merged["phone"] = combined_phones[0]["value"]

    # 4. Location: prefer more detailed
    loc_ex = existing.get("location")
    loc_in = incoming.get("location")
    if not loc_ex and loc_in:
        merged["location"] = loc_in
    elif isinstance(loc_ex, dict) and isinstance(loc_in, dict):
        merged["location"] = {**loc_ex, **{k: v for k, v in loc_in.items() if v}}

    # 5. Skills: union all skills, preserving sources
    skills_map: Dict[str, Dict[str, Any]] = {}
    for s in (existing.get("skills") or []):
        sname = s.get("name") if isinstance(s, dict) else str(s)
        if sname:
            norm = normalize_skill(sname) or sname
            key = norm.lower()
            srcs = s.get("sources", ["existing"]) if isinstance(s, dict) else ["existing"]
            skills_map[key] = {"name": norm.title() if norm else sname, "sources": list(srcs)}

    for s in (incoming.get("skills") or []):
        sname = s.get("name") if isinstance(s, dict) else str(s)
        if sname:
            norm = normalize_skill(sname) or sname
            key = norm.lower()
            srcs = s.get("sources", ["incoming"]) if isinstance(s, dict) else ["incoming"]
            if key in skills_map:
                skills_map[key]["sources"] = sorted(list(set(skills_map[key]["sources"] + srcs)))
            else:
                skills_map[key] = {"name": norm.title() if norm else sname, "sources": list(srcs)}

    merged["skills"] = list(skills_map.values())

    # 6. Experience: union without duplicates
    exp_ex = existing.get("experience") or []
    exp_in = incoming.get("experience") or []
    merged_exp = list(exp_ex)
    for ei in exp_in:
        if isinstance(ei, dict):
            comp = (ei.get("company") or "").lower().strip()
            title = (ei.get("title") or "").lower().strip()
            # Check if this role is already represented
            is_dup = any(
                comp and comp == (e.get("company") or "").lower().strip()
                and (not title or title in (e.get("title") or "").lower().strip() or (e.get("title") or "").lower().strip() in title)
                for e in merged_exp if isinstance(e, dict)
            )
            if not is_dup:
                merged_exp.append(ei)
    merged["experience"] = merged_exp

    # 7. Education: union without duplicates
    edu_ex = existing.get("education") or []
    edu_in = incoming.get("education") or []
    merged_edu = list(edu_ex)
    for ed in edu_in:
        if isinstance(ed, dict):
            inst = (ed.get("institution") or "").lower().strip()
            deg = (ed.get("degree") or "").lower().strip()
            is_dup = any(
                inst and inst == (e.get("institution") or "").lower().strip()
                for e in merged_edu if isinstance(e, dict)
            )
            if not is_dup:
                merged_edu.append(ed)
    merged["education"] = merged_edu

    # 8. Sources merged: union of all contributing filenames
    src_ex = existing.get("sources_merged") or []
    src_in = incoming.get("sources_merged") or []
    combined_sources = sorted(list(set(src_ex + src_in)))
    merged["sources_merged"] = combined_sources

    # Merge summary
    ms_ex = existing.get("merge_summary") or {}
    merged["merge_summary"] = {
        **ms_ex,
        "sources_merged": combined_sources,
        "last_merged_at": "now",
    }

    # Confidence: max or recalculated
    conf_ex = float(existing.get("overall_confidence") or 0.5)
    conf_in = float(incoming.get("overall_confidence") or 0.5)
    merged["overall_confidence"] = max(conf_ex, conf_in)

    return merged

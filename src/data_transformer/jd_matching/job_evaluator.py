"""
Job-Specific Candidate Evaluation & Recommendation Engine.

Evaluates two candidates independently against specific job requirements:
  - Technical skills match (30%)
  - Relevant work experience (25%)
  - Experience duration (15%)
  - Educational qualifications (10%)
  - Project relevance (10%)
  - Certifications and preferred skills (10%)

Provides transparent weighted scoring, configurable weights, detailed side-by-side
comparisons, strengths/gaps breakdowns, and explainable AI recommendations answering
all key hiring questions based on real extracted profile evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from rapidfuzz import fuzz

from data_transformer.normalizers.skills import normalize_skill


# Skill equivalencies and related clusters
SKILL_EQUIVALENCE_CLUSTERS: List[Set[str]] = [
    # Relational databases
    {"sql", "postgresql", "mysql", "oracle", "sqlite", "relational database", "rdbms"},
    # Frontend fundamentals
    {"html", "html5", "css", "css3", "javascript", "js", "web technologies", "frontend"},
    # Modern frontend frameworks
    {"react", "react.js", "reactjs", "vue", "vue.js", "angular", "next.js"},
    # Java ecosystem
    {"java", "spring", "spring boot", "servlets", "jsp", "jdbc", "hibernate"},
    # Python & Data / ML
    {"python", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "machine learning", "data analysis", "eda"},
    # Cloud & DevOps
    {"aws", "gcp", "azure", "docker", "kubernetes", "cloud", "devops", "ci/cd"},
    # Backend / Node
    {"node.js", "nodejs", "express", "express.js", "rest api", "restful apis"},
    # Core CS
    {"dsa", "data structures", "algorithms", "data structures & algorithms", "oop", "oops"},
]

DEFAULT_WEIGHTS = {
    "technical_skills": 0.30,
    "relevant_experience": 0.25,
    "experience_duration": 0.15,
    "education": 0.10,
    "project_relevance": 0.10,
    "certifications_preferred": 0.10,
}


@dataclass
class JobCriteria:
    job_title: str
    job_description: str = ""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    required_education: str = "Bachelor's Degree"
    additional_requirements: str = ""
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def __post_init__(self):
        if not self.weights:
            self.weights = dict(DEFAULT_WEIGHTS)


def _extract_candidate_skill_names(canon: Dict[str, Any]) -> List[str]:
    skills = []
    raw_skills = canon.get("skills") or []
    if isinstance(raw_skills, list):
        for s in raw_skills:
            if isinstance(s, dict):
                val = s.get("name") or s.get("value") or ""
            else:
                val = str(s)
            if val and str(val).strip():
                skills.append(str(val).strip())

    # Also harvest technologies mentioned in projects
    for proj in canon.get("projects") or []:
        if isinstance(proj, dict):
            for t in proj.get("technologies") or []:
                if t and str(t).strip() not in skills:
                    skills.append(str(t).strip())
    return skills


def _extract_candidate_experience_years(canon: Dict[str, Any]) -> float:
    # 1. Stored years_experience
    stored_exp = canon.get("years_experience")
    if isinstance(stored_exp, (int, float)) and stored_exp > 0:
        return float(stored_exp)
    if isinstance(stored_exp, dict):
        v = stored_exp.get("value")
        if isinstance(v, (int, float)) and v > 0:
            return float(v)

    # 2. Derive from experience entries
    exp_entries = canon.get("experience") or []
    if not isinstance(exp_entries, list) or not exp_entries:
        return 0.0

    total_months = 0
    for exp in exp_entries:
        if not isinstance(exp, dict):
            continue
        start_d = exp.get("start_date") or exp.get("start")
        end_d = exp.get("end_date") or exp.get("end")
        is_curr = exp.get("is_current", False)

        try:
            if start_d:
                sy, sm = int(start_d[:4]), int(start_d[5:7]) if len(start_d) >= 7 else 1
                if is_curr or not end_d:
                    ey, em = 2026, 3
                else:
                    ey, em = int(end_d[:4]), int(end_d[5:7]) if len(end_d) >= 7 else 1
                m = (ey - sy) * 12 + (em - sm)
                total_months += max(m, 1)  # At least 1 month per role
            else:
                total_months += 3  # Internship default
        except Exception:
            total_months += 3

    return round(total_months / 12.0, 1)


def _check_skill_match(required_skill: str, candidate_skills: List[str]) -> Tuple[float, str, str]:
    """
    Returns (score, status, explanation):
      - 1.0, 'Exact Match', matched skill name
      - 0.75, 'Equivalent / Related Skill', related skill name
      - 0.50, 'Partial Knowledge', note
      - 0.0, 'Skill Not Found', ''
    """
    req_norm = (normalize_skill(required_skill) or required_skill).lower().strip()

    cand_norms = {}
    for cs in candidate_skills:
        n = (normalize_skill(cs) or cs).lower().strip()
        cand_norms[n] = cs

    # 1. Exact match
    if req_norm in cand_norms:
        return 1.0, "Exact Match", cand_norms[req_norm]

    # Fuzzy match on normalized string
    for cn, orig in cand_norms.items():
        if fuzz.ratio(req_norm, cn) >= 85:
            return 1.0, "Exact Match", orig

    # 2. Equivalent / Related match
    for cluster in SKILL_EQUIVALENCE_CLUSTERS:
        if req_norm in cluster:
            overlap = cluster & set(cand_norms.keys())
            if overlap:
                matched_orig = cand_norms[list(overlap)[0]]
                return 0.75, "Equivalent / Related Skill", f"{matched_orig} (related to {required_skill})"

    # 3. Substring / Partial knowledge
    for cn, orig in cand_norms.items():
        if req_norm in cn or cn in req_norm:
            return 0.50, "Partial Knowledge", orig

    return 0.0, "Skill Not Found", ""


def evaluate_single_candidate(
    candidate: Dict[str, Any],
    criteria: JobCriteria,
) -> Dict[str, Any]:
    """
    Evaluates one candidate against the specified JobCriteria across the 6 dimensions.
    """
    name_val = candidate.get("full_name") or candidate.get("name") or "Anonymous"
    candidate_name = name_val.get("value") if isinstance(name_val, dict) else str(name_val)
    cid = candidate.get("candidate_id") or candidate.get("id") or "unknown"

    cand_skills = _extract_candidate_skill_names(candidate)
    cand_experience = candidate.get("experience") or []
    cand_education = candidate.get("education") or []
    cand_projects = candidate.get("projects") or []
    cand_certifications = candidate.get("certifications") or []
    cand_years = _extract_candidate_experience_years(candidate)

    # ─── Dimension 1: Technical Skills Match (30%) ─────────────────────────
    req_skills = criteria.required_skills
    if not req_skills and criteria.job_description:
        # Extract skills from job description if not explicitly passed
        for word in re.findall(r"\b[A-Za-z0-9\+#\.]+\b", criteria.job_description):
            ns = normalize_skill(word)
            if ns and ns not in req_skills:
                req_skills.append(ns.title())

    matching_skills: List[str] = []
    missing_required_skills: List[str] = []
    skill_evaluations: List[Dict[str, Any]] = []

    if req_skills:
        skill_scores = []
        for rs in req_skills:
            score, status, matched_evidence = _check_skill_match(rs, cand_skills)
            skill_scores.append(score)
            if score >= 0.75:
                matching_skills.append(rs if score == 1.0 else f"{rs} (via {matched_evidence})")
            elif score == 0.50:
                matching_skills.append(f"{rs} (partial: {matched_evidence})")
            else:
                missing_required_skills.append(rs)

            skill_evaluations.append({
                "skill": rs,
                "score": score,
                "status": status,
                "evidence": matched_evidence,
            })
        tech_score = (sum(skill_scores) / len(skill_scores)) * 100.0
        tech_explanation = (
            f"Satisfies {len(matching_skills)} of {len(req_skills)} required technical skills. "
            f"Verified: {', '.join(matching_skills[:5]) if matching_skills else 'None'}. "
            f"Missing: {', '.join(missing_required_skills) if missing_required_skills else 'None'}."
        )
    else:
        tech_score = 75.0 if cand_skills else 20.0
        tech_explanation = f"Evaluated across {len(cand_skills)} verified profile skills."

    # ─── Dimension 2: Relevant Work Experience (25%) ──────────────────────
    role_keywords = set(re.findall(r"\b[a-z]{3,}\b", criteria.job_title.lower()))
    relevant_roles = []
    exp_quality_points = 0.0

    if cand_experience:
        for exp in cand_experience:
            if not isinstance(exp, dict):
                continue
            title = str(exp.get("title") or "")
            company = str(exp.get("company") or "")
            desc = str(exp.get("description") or "")
            combined = f"{title} {company} {desc}".lower()

            matches = sum(1 for kw in role_keywords if kw in combined)
            if matches >= 1:
                relevant_roles.append(f"{title} at {company}" if company else title)
                exp_quality_points += 40.0
            else:
                exp_quality_points += 20.0

        exp_score = min(exp_quality_points, 100.0)
        exp_explanation = (
            f"Demonstrates relevant background in {len(relevant_roles)} role(s): "
            f"{', '.join(relevant_roles) if relevant_roles else 'General software exposure'}."
        )
    else:
        exp_score = 15.0 if cand_projects else 5.0
        exp_explanation = "No formal employment or internship history on record."

    # ─── Dimension 3: Experience Duration (15%) ───────────────────────────
    min_years = criteria.min_experience_years
    if min_years <= 0:
        dur_score = 100.0 if cand_years >= 0.5 else (85.0 if cand_years > 0 else 70.0)
        dur_explanation = f"Candidate has {cand_years} years experience; meets entry-level / open experience criteria."
    else:
        ratio = cand_years / min_years
        dur_score = min(ratio * 100.0, 100.0)
        if cand_years >= min_years:
            dur_explanation = f"Meets experience criteria: {cand_years} years (required: {min_years}+ years)."
        else:
            dur_explanation = f"Below minimum experience: has {cand_years} years vs {min_years} years required."

    # ─── Dimension 4: Educational Qualifications (10%) ─────────────────────
    req_edu = criteria.required_education.lower()
    edu_score = 20.0
    edu_details = []

    for edu in cand_education:
        if not isinstance(edu, dict):
            continue
        deg = str(edu.get("degree") or "")
        inst = str(edu.get("institution") or "")
        field_of_s = str(edu.get("field_of_study") or "")
        acad = str(edu.get("academic_qualifications") or edu.get("grade") or "")

        full_edu_str = f"{deg} in {field_of_s} from {inst} ({acad})".strip()
        edu_details.append(full_edu_str)

        # Tech degree match
        if any(term in deg.lower() or term in field_of_s.lower() for term in ["computer", "technology", "information", "b.tech", "b.e.", "b.sc", "software"]):
            edu_score = max(edu_score, 95.0)
        elif any(term in deg.lower() for term in ["engineering", "m.tech", "master"]):
            edu_score = max(edu_score, 85.0)
        elif any(term in deg.lower() for term in ["bachelor", "degree"]):
            edu_score = max(edu_score, 75.0)
        elif any(term in deg.lower() for term in ["hsc", "school"]):
            edu_score = max(edu_score, 50.0)

    if not cand_education:
        edu_score = 15.0
        edu_explanation = "No formal educational credentials recorded on file."
    else:
        edu_explanation = f"Credentials: {'; '.join(edu_details[:2])}."

    # ─── Dimension 5: Project Relevance (10%) ─────────────────────────────
    proj_score = 0.0
    proj_highlights = []
    if cand_projects:
        for proj in cand_projects:
            if not isinstance(proj, dict):
                continue
            pname = proj.get("name") or "Project"
            ptech = proj.get("technologies") or []
            pdesc = proj.get("description") or ""

            # Check overlap with required skills or role
            tech_overlap = [t for t in ptech if any(_check_skill_match(t, [rs])[0] >= 0.75 for rs in req_skills)]
            if tech_overlap or any(kw in f"{pname} {pdesc}".lower() for kw in role_keywords):
                proj_score += 45.0
                proj_highlights.append(f"{pname} ({', '.join(tech_overlap) if tech_overlap else 'Relevant Domain'})")
            else:
                proj_score += 25.0
                proj_highlights.append(pname)

        proj_score = min(proj_score, 100.0)
        proj_explanation = f"Developed {len(cand_projects)} project(s): {', '.join(proj_highlights[:2])}."
    else:
        proj_score = 10.0
        proj_explanation = "No project portfolio records found in candidate profile."

    # ─── Dimension 6: Certifications & Preferred Skills (10%) ─────────────
    pref_skills = criteria.preferred_skills
    matched_preferred = []
    for ps in pref_skills:
        s_score, _, p_ev = _check_skill_match(ps, cand_skills)
        if s_score >= 0.75:
            matched_preferred.append(ps)

    cert_points = len(cand_certifications) * 30.0
    pref_points = (len(matched_preferred) / max(len(pref_skills), 1)) * 50.0 if pref_skills else 25.0
    cert_score = min(cert_points + pref_points, 100.0)

    cert_names = [c if isinstance(c, str) else c.get("name", "") for c in cand_certifications]
    cert_explanation = (
        f"{len(cand_certifications)} certification(s) ({', '.join(cert_names[:2]) if cert_names else 'None'}). "
        f"Preferred skills matched: {len(matched_preferred)}/{len(pref_skills)}."
    )

    # ─── Weighted Score Calculation ───────────────────────────────────────
    weights = dict(criteria.weights)
    total_weight = sum(weights.values()) or 1.0

    scores = {
        "technical_skills": tech_score,
        "relevant_experience": exp_score,
        "experience_duration": dur_score,
        "education": edu_score,
        "project_relevance": proj_score,
        "certifications_preferred": cert_score,
    }

    weighted_score = sum(scores[k] * (weights.get(k, 0.0) / total_weight) for k in scores)
    overall_score = round(weighted_score, 1)

    # Match Category
    if not cand_skills and not cand_experience and not cand_education:
        match_category = "Insufficient Evidence"
    elif overall_score >= 80.0:
        match_category = "Excellent Match"
    elif overall_score >= 65.0:
        match_category = "Strong Match"
    elif overall_score >= 50.0:
        match_category = "Moderate Match"
    elif overall_score >= 20.0:
        match_category = "Low Match"
    else:
        match_category = "Insufficient Evidence"

    # Strengths and Gaps
    strengths = []
    gaps = []

    if tech_score >= 70.0:
        strengths.append(f"Strong skill alignment with {len(matching_skills)} verified core competencies")
    else:
        gaps.append(f"Missing {len(missing_required_skills)} required technical skills ({', '.join(missing_required_skills[:3])})")

    if exp_score >= 70.0:
        strengths.append(f"Demonstrated hands-on experience in relevant role(s) ({', '.join(relevant_roles[:2])})")
    elif not cand_experience:
        gaps.append("Lacks documented professional or internship work experience")

    if dur_score >= 90.0:
        strengths.append(f"Meets or exceeds required tenure ({cand_years} years)")
    elif min_years > 0 and cand_years < min_years:
        gaps.append(f"Tenure deficit: {cand_years} years vs {min_years} years requested")

    if edu_score >= 80.0:
        strengths.append("Strong technical educational background (Computer Science / IT)")
    elif not cand_education:
        gaps.append("No verified degree credentials on file")

    if proj_score >= 70.0:
        strengths.append(f"Demonstrated project portfolio ({', '.join(proj_highlights[:2])})")

    if cert_score >= 70.0:
        strengths.append(f"Verified certifications: {', '.join(cert_names[:2])}")

    return {
        "candidate_id": cid,
        "name": candidate_name,
        "overall_score": overall_score,
        "match_category": match_category,
        "dimension_scores": scores,
        "explanations": {
            "technical_skills": tech_explanation,
            "relevant_experience": exp_explanation,
            "experience_duration": dur_explanation,
            "education": edu_explanation,
            "project_relevance": proj_explanation,
            "certifications_preferred": cert_explanation,
        },
        "matching_skills": matching_skills,
        "missing_required_skills": missing_required_skills,
        "relevant_experience": exp_explanation,
        "strengths": strengths if strengths else ["Profile on file with basic domain exposure"],
        "gaps": gaps if gaps else ["No major qualification deficits identified"],
        "evidence": {
            "skills_count": len(cand_skills),
            "experience_years": cand_years,
            "education_entries": len(cand_education),
            "projects_count": len(cand_projects),
            "certifications_count": len(cand_certifications),
        },
    }


def compare_and_recommend_candidates(
    candidate_a: Dict[str, Any],
    candidate_b: Dict[str, Any],
    criteria: JobCriteria,
) -> Dict[str, Any]:
    """
    Evaluates Candidate A and Candidate B independently against criteria,
    produces a side-by-side comparison, and generates explainable AI hiring recommendations.
    """
    eval_a = evaluate_single_candidate(candidate_a, criteria)
    eval_b = evaluate_single_candidate(candidate_b, criteria)

    score_a = eval_a["overall_score"]
    score_b = eval_b["overall_score"]
    name_a = eval_a["name"]
    name_b = eval_b["name"]
    diff = round(score_a - score_b, 1)
    abs_diff = abs(diff)

    # Determine Winner / Tie / Insufficient Evidence
    both_insufficient = (eval_a["match_category"] == "Insufficient Evidence" and eval_b["match_category"] == "Insufficient Evidence")

    if both_insufficient:
        recommended_candidate = "Insufficient Evidence"
        recommendation_confidence = "Low"
        winner_id = None
        synthesis = (
            f"Both {name_a} and {name_b} have insufficient evidence or incomplete profiles "
            f"for the role of '{criteria.job_title}'. Neither candidate can be recommended without additional data."
        )
    elif abs_diff < 2.5:
        recommended_candidate = "Tie / Equally Qualified"
        recommendation_confidence = "Moderate"
        winner_id = None
        synthesis = (
            f"{name_a} (Score: {score_a}%) and {name_b} (Score: {score_b}%) are equally qualified "
            f"for the role of '{criteria.job_title}'. Both candidates present comparable technical alignments."
        )
    elif score_a > score_b:
        recommended_candidate = f"Candidate A ({name_a})"
        recommendation_confidence = "High" if abs_diff >= 12.0 else "Moderate"
        winner_id = eval_a["candidate_id"]
        synthesis = (
            f"{name_a} is recommended over {name_b} for '{criteria.job_title}' with an alignment margin of +{diff}%. "
            f"{name_a} demonstrates superior performance in {', '.join([k.replace('_', ' ').title() for k, v in eval_a['dimension_scores'].items() if v > eval_b['dimension_scores'][k]])}."
        )
    else:
        recommended_candidate = f"Candidate B ({name_b})"
        recommendation_confidence = "High" if abs_diff >= 12.0 else "Moderate"
        winner_id = eval_b["candidate_id"]
        synthesis = (
            f"{name_b} is recommended over {name_a} for '{criteria.job_title}' with an alignment margin of +{abs_diff}%. "
            f"{name_b} demonstrates superior performance in {', '.join([k.replace('_', ' ').title() for k, v in eval_b['dimension_scores'].items() if v > eval_a['dimension_scores'][k]])}."
        )

    # Detailed Side-by-Side Comparison Rows
    criteria_labels = {
        "technical_skills": "Technical Skills Match",
        "relevant_experience": "Relevant Experience",
        "experience_duration": "Experience Duration",
        "education": "Education Qualifications",
        "project_relevance": "Project Relevance",
        "certifications_preferred": "Certifications & Preferred Skills",
    }

    detailed_comparison = []
    for crit_key, label in criteria_labels.items():
        w = criteria.weights.get(crit_key, 0.10)
        detailed_comparison.append({
            "criterion": label,
            "weight": w,
            "weight_percentage": f"{int(w * 100)}%",
            "candidate_a": {
                "score": round(eval_a["dimension_scores"][crit_key], 1),
                "explanation": eval_a["explanations"][crit_key],
            },
            "candidate_b": {
                "score": round(eval_b["dimension_scores"][crit_key], 1),
                "explanation": eval_b["explanations"][crit_key],
            },
        })

    # AI Recommendation: Explicitly answers the 7 required questions
    winner_name = name_a if score_a > score_b else (name_b if score_b > score_a else "Neither (Tie)")
    loser_name = name_b if score_a > score_b else (name_a if score_b > score_a else "Neither (Tie)")
    winner_eval = eval_a if score_a >= score_b else eval_b
    loser_eval = eval_b if score_a >= score_b else eval_a

    where_other_excels = []
    for k, label in criteria_labels.items():
        if loser_eval["dimension_scores"][k] > winner_eval["dimension_scores"][k]:
            where_other_excels.append(f"{label} ({loser_eval['dimension_scores'][k]}% vs {winner_eval['dimension_scores'][k]}%)")

    ai_recommendation = {
        "which_candidate_more_suitable": winner_name if abs_diff >= 2.5 else "Both candidates are equally suitable.",
        "why_more_suitable": (
            f"{winner_name} achieved an overall match of {winner_eval['overall_score']}% compared to {loser_name}'s {loser_eval['overall_score']}%. "
            f"Key driving factors: {', '.join(winner_eval['strengths'][:2])}."
        ) if abs_diff >= 2.5 else f"Both candidates achieved equivalent alignment within a {abs_diff}% margin.",
        "requirements_satisfied_by_winner": winner_eval["matching_skills"],
        "important_requirements_missing": winner_eval["missing_required_skills"],
        "where_other_candidate_excels": (
            f"{loser_name} demonstrated higher evaluation in: {', '.join(where_other_excels)}."
        ) if where_other_excels else f"{loser_name} did not outscore {winner_name} in any primary dimension.",
        "are_both_equally_suitable": abs_diff < 2.5,
        "is_either_unsuitable": (
            f"Yes: {eval_a['name'] if score_a < 35.0 else eval_b['name']} has low alignment (<35%) for this role."
        ) if (score_a < 35.0 or score_b < 35.0) else "No: both candidates meet the minimum threshold of qualifications for consideration.",
        "synthesis": synthesis,
    }

    return {
        "job_details": {
            "title": criteria.job_title,
            "description": criteria.job_description,
            "required_skills": criteria.required_skills,
            "preferred_skills": criteria.preferred_skills,
            "min_experience_years": criteria.min_experience_years,
            "required_education": criteria.required_education,
            "additional_requirements": criteria.additional_requirements,
        },
        "overall_recommendation": {
            "candidate_a_score": score_a,
            "candidate_b_score": score_b,
            "recommended_candidate": recommended_candidate,
            "recommendation_confidence": recommendation_confidence,
            "score_margin": abs_diff,
            "winner_id": winner_id,
            "candidate_a_category": eval_a["match_category"],
            "candidate_b_category": eval_b["match_category"],
            "summary": synthesis,
        },
        "candidate_a": eval_a,
        "candidate_b": eval_b,
        "detailed_comparison": detailed_comparison,
        "ai_recommendation": ai_recommendation,
    }

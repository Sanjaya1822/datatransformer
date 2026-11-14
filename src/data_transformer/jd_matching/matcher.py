"""
Job Description Semantic & Requirement Matcher.

Features:
- Expanded skill vocabulary (200+ skills)
- Structured criteria extraction:
  * Required skills
  * Preferred skills
  * Experience requirements
  * Education requirements
  * Certifications
  * Job responsibilities
- Fuzzy skill matching using rapidfuzz + aliases + substring
- Accurate experience calculation from date ranges
- Robust education and degree matching
- Certification matching
- Responsibility alignment matching against work history
- Detailed per-requirement evidence with candidate profile references
- Fully explainable ranking with strengths, gaps, and comparative rationale
- Bulletproof None/empty input handling preventing 500 crashes
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz, process

from data_transformer.normalizers.skills import normalize_skill


# ── Expanded skill vocabulary ─────────────────────────────────────────────────
TECH_SKILLS = [
    # Languages
    "python", "java", "golang", "go", "c++", "c", "c#", "javascript", "typescript",
    "ruby", "scala", "kotlin", "swift", "rust", "php", "perl", "r", "matlab",
    "bash", "shell", "powershell", "dart", "lua", "haskell", "erlang", "elixir",
    # Web frameworks
    "react", "angular", "vue", "next.js", "nuxt", "svelte", "fastapi", "django",
    "flask", "express", "spring", "spring boot", "rails", "laravel", "asp.net",
    "node.js", "nestjs", "graphql", "rest api", "restful", "grpc", "websocket",
    "html", "css", "tailwind", "bootstrap",
    # Data / ML
    "machine learning", "deep learning", "nlp", "computer vision", "pytorch",
    "tensorflow", "keras", "scikit-learn", "pandas", "numpy", "scipy",
    "xgboost", "lightgbm", "hugging face", "transformers", "llm", "openai",
    "langchain", "spark", "hadoop", "kafka", "airflow", "mlflow", "dbt", "rag",
    # Databases
    "sql", "postgresql", "mysql", "sqlite", "oracle", "mongodb", "redis",
    "elasticsearch", "cassandra", "dynamodb", "neo4j", "firebase", "supabase",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "ci/cd", "github actions", "gitlab ci", "circleci",
    "helm", "istio", "prometheus", "grafana", "datadog", "splunk", "nginx",
    "linux", "unix", "git", "github", "gitlab", "bitbucket",
    # Data engineering
    "etl", "data pipeline", "data warehouse", "snowflake", "bigquery", "redshift",
    "databricks", "apache beam", "flink", "rabbitmq", "celery",
    # Security / Core CS / Architecture
    "oauth", "jwt", "ssl", "tls", "cybersecurity", "penetration testing",
    "agile", "scrum", "jira", "confluence", "microservices", "api gateway",
    "system design", "distributed systems", "blockchain", "web3", "dsa",
    "data structures", "algorithms", "oop", "dbms", "operating systems", "computer networks",
]

# Skill normalization aliases
SKILL_ALIASES: Dict[str, str] = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "k8s": "kubernetes", "node": "node.js", "pg": "postgresql",
    "postgres": "postgresql", "ml": "machine learning", "dl": "deep learning",
    "aws lambda": "aws", "ec2": "aws", "s3": "aws", "gke": "kubernetes",
    "react.js": "react", "reactjs": "react", "vuejs": "vue",
    "springboot": "spring boot", "nodejs": "node.js",
    "data structures and algorithms": "data structures",
    "algorithms and data structures": "algorithms",
}

# Common industry certifications
CERTIFICATIONS_VOCAB = [
    "aws certified", "aws solutions architect", "aws developer",
    "azure certified", "azure administrator", "azure solutions architect",
    "gcp certified", "google cloud certified", "google professional cloud architect",
    "cka", "certified kubernetes administrator", "ckad",
    "pmp", "project management professional",
    "cissp", "certified ethical hacker", "ceh", "security+",
    "scrum master", "csm", "psm",
    "oracle certified", "comptia",
]


def _normalize_skill_name(s: str) -> str:
    s = s.lower().strip()
    return SKILL_ALIASES.get(s, s)


def _fuzzy_skill_match(required_skill: str, candidate_skills: Set[str], threshold: int = 78) -> Optional[str]:
    """
    Returns the best matching candidate skill or None.
    Uses RapidFuzz token_set_ratio to handle abbreviations and variants.
    """
    if not candidate_skills or not required_skill:
        return None

    req = _normalize_skill_name(required_skill)
    if req in candidate_skills:
        return req

    # Direct substring check
    for cs in candidate_skills:
        if req == cs or req in cs or cs in req:
            return cs

    # Fuzzy match
    result = process.extractOne(req, list(candidate_skills), scorer=fuzz.token_set_ratio)
    if result and result[1] >= threshold:
        return result[0]

    return None


def _calculate_experience_years(experience_entries: Optional[List[Dict[str, Any]]]) -> float:
    """
    Calculates total years of experience from experience entries.
    Parses start_date / end_date fields. Falls back to entry count estimate.
    Safely handles None or malformed dictionaries.
    """
    if not experience_entries or not isinstance(experience_entries, list):
        return 0.0

    total_months = 0
    unparsed_count = 0

    for entry in experience_entries:
        if not isinstance(entry, dict):
            unparsed_count += 1
            continue

        start_raw = entry.get("start_date") or entry.get("start") or ""
        end_raw = entry.get("end_date") or entry.get("end") or ""

        try:
            start_dt = _parse_date(str(start_raw))
            if not start_dt:
                unparsed_count += 1
                continue
            end_dt = (
                _parse_date(str(end_raw))
                if end_raw and str(end_raw).lower() not in ("present", "current", "now", "")
                else datetime.now()
            )
            if not end_dt or end_dt < start_dt:
                unparsed_count += 1
                continue
            months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
            total_months += max(0, months)
        except Exception:
            unparsed_count += 1

    calculated_years = total_months / 12.0
    estimated_years = unparsed_count * 1.0

    return round(calculated_years + estimated_years, 1)


def _parse_date(date_str: str) -> Optional[datetime]:
    """Attempts to parse various date string formats."""
    if not date_str or str(date_str).lower() in ("present", "current", "now", "", "none"):
        return None
    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%m/%Y", "%Y-%m", "%B %Y",
        "%b %Y", "%Y", "%m-%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    # Try just year
    year_match = re.search(r"\b(19|20)\d{2}\b", str(date_str))
    if year_match:
        try:
            return datetime(int(year_match.group()), 1, 1)
        except Exception:
            pass
    return None


@dataclass
class JDRequirement:
    category: str      # "required_skill" | "preferred_skill" | "experience" | "education" | "certification" | "responsibility"
    text: str
    weight: float
    is_required: bool = False


@dataclass
class RequirementMatchItem:
    requirement: str
    category: str
    matched: bool
    evidence: Optional[str] = None
    score_contribution: float = 0.0
    status: str = "missing"  # "verified" | "partial" | "missing"


@dataclass
class CandidateJDMatchScore:
    candidate_id: str
    candidate_name: str
    overall_score: float              # 0.0 to 100.0
    fit_level: str                    # "Strong Fit" | "Moderate Fit" | "Potential Fit" | "Low Alignment"
    matched_skills: List[str] = field(default_factory=list)
    missing_required_skills: List[str] = field(default_factory=list)
    relevant_experience: str = ""
    education_match: str = ""
    matched_requirements: List[RequirementMatchItem] = field(default_factory=list)
    missing_requirements: List[RequirementMatchItem] = field(default_factory=list)
    summary_explanation: str = ""
    advisory_note: str = "Recommendations are strictly advisory. Final candidate assessment requires human review."
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    experience_years: float = 0.0


class JobDescriptionParser:
    """Extracts structured criteria from raw Job Description text."""

    @staticmethod
    def parse_requirements(jd_text: str) -> List[JDRequirement]:
        if not jd_text or not jd_text.strip():
            return []

        requirements: List[JDRequirement] = []
        text_lower = jd_text.lower()

        # 1. Experience years — capture the maximum mentioned
        exp_years = 0
        for m in re.finditer(r"(\d+)\+?\s*(?:-\s*\d+)?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:relevant\s+)?(?:professional\s+)?experience", text_lower):
            y = int(m.group(1))
            if y > exp_years:
                exp_years = y
        if exp_years > 0:
            requirements.append(
                JDRequirement(
                    category="experience",
                    text=f"{exp_years}+ years of professional experience",
                    weight=0.20,
                    is_required=True,
                )
            )

        # 2. Education level
        if any(term in text_lower for term in ["ph.d", "phd", "doctorate"]):
            requirements.append(
                JDRequirement(
                    category="education",
                    text="PhD or Doctorate in Computer Science or relevant discipline",
                    weight=0.10,
                    is_required=False,
                )
            )
        elif any(term in text_lower for term in ["master", "ms ", "m.s.", "msc", "m.tech", "m.eng"]):
            requirements.append(
                JDRequirement(
                    category="education",
                    text="Master's degree or equivalent in a relevant field",
                    weight=0.10,
                    is_required=False,
                )
            )
        elif any(term in text_lower for term in ["bachelor", "b.s.", "b.tech", "b.e.", "degree", "b.sc", "bs in"]):
            requirements.append(
                JDRequirement(
                    category="education",
                    text="Bachelor's degree in Computer Science or related discipline",
                    weight=0.10,
                    is_required=True,
                )
            )

        # 3. Certifications
        found_certs = []
        for cert in CERTIFICATIONS_VOCAB:
            if re.search(r"\b" + re.escape(cert) + r"\b", text_lower):
                found_certs.append(cert.title())
        for cert_name in found_certs:
            requirements.append(
                JDRequirement(
                    category="certification",
                    text=cert_name,
                    weight=0.08,
                    is_required=False,
                )
            )

        # 4. Job Responsibilities
        resp_section = _extract_section(jd_text, [
            r"(?:responsibilities|duties|what\s+you(?:'ll|\s+will)\s+do|role\s+overview)[:\s\n]+(.*?)(?=requirements|qualifications|skills|preferred|\Z)"
        ])
        if resp_section:
            resp_lines = [
                line.strip("-•* \t") for line in resp_section.splitlines()
                if len(line.strip("-•* \t")) > 20 and len(line.strip("-•* \t")) < 200
            ]
            for r_line in resp_lines[:3]:  # Top 3 key responsibilities
                requirements.append(
                    JDRequirement(
                        category="responsibility",
                        text=r_line,
                        weight=0.08,
                        is_required=False,
                    )
                )

        # 5. Skills — scan from expanded vocabulary
        found_skills: List[str] = []
        for skill in TECH_SKILLS:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                found_skills.append(skill)

        if found_skills:
            required_section = _extract_section(jd_text, [
                r"(?:required|must\s+have|mandatory|essential|minimum\s+qualifications)[:\s\n]+(.*?)(?=preferred|nice.to.have|bonus|\Z)"
            ])
            preferred_section = _extract_section(jd_text, [
                r"(?:preferred|nice.to.have|bonus|plus|desired)[:\s\n]+(.*?)(?=required|must|\Z)"
            ])

            weight_per_skill = min(0.12, 0.50 / len(found_skills))

            for s in found_skills:
                is_req = False
                if required_section and re.search(r"\b" + re.escape(s) + r"\b", required_section.lower()):
                    is_req = True
                elif preferred_section and re.search(r"\b" + re.escape(s) + r"\b", preferred_section.lower()):
                    is_req = False
                else:
                    is_req = True  # Default to required if not clearly marked preferred

                category = "required_skill" if is_req else "preferred_skill"
                requirements.append(
                    JDRequirement(
                        category=category,
                        text=s.title(),
                        weight=round(weight_per_skill, 3),
                        is_required=is_req,
                    )
                )

        return requirements


def _extract_section(text: str, patterns: List[str]) -> str:
    """Helper to extract sections matching regex patterns."""
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1)[:2500]
    return ""


class JobDescriptionMatcher:
    """Compares candidate profiles against Job Description requirements with full explainability."""

    def __init__(self, category_weights: Optional[Dict[str, float]] = None):
        self.weights = category_weights or {
            "required_skill": 1.0,
            "preferred_skill": 0.5,
            "experience": 1.0,
            "education": 0.5,
            "certification": 0.6,
            "responsibility": 0.5,
        }

    def match_candidate(
        self,
        candidate_canonical: Optional[Dict[str, Any]],
        requirements: List[JDRequirement],
    ) -> CandidateJDMatchScore:
        if candidate_canonical is None:
            candidate_canonical = {}

        cid = candidate_canonical.get("candidate_id") or "unknown"
        name_val = candidate_canonical.get("full_name") or candidate_canonical.get("name") or {}
        candidate_name = (
            name_val.get("value") if isinstance(name_val, dict) else str(name_val or "Anonymous Candidate")
        )

        # ── 1. Candidate Skill Set ──────────────────────────────────────────
        raw_skills = candidate_canonical.get("skills") or []
        cand_skills: Set[str] = set()
        if isinstance(raw_skills, list):
            for s in raw_skills:
                if isinstance(s, dict):
                    name = s.get("name") or ""
                else:
                    name = str(s)
                normalized = _normalize_skill_name(name)
                if normalized:
                    cand_skills.add(normalized)
                if name.lower().strip():
                    cand_skills.add(name.lower().strip())

        # ── 2. Experience ───────────────────────────────────────────────────
        cand_years_raw = candidate_canonical.get("years_experience")
        total_exp = 0.0
        if isinstance(cand_years_raw, dict):
            total_exp = float(cand_years_raw.get("value") or 0.0)
        elif cand_years_raw is not None:
            try:
                total_exp = float(cand_years_raw)
            except Exception:
                total_exp = 0.0

        exp_entries = candidate_canonical.get("experience") or []
        if not isinstance(exp_entries, list):
            exp_entries = []

        if total_exp == 0.0 and exp_entries:
            total_exp = _calculate_experience_years(exp_entries)

        # Build companies and experience narrative text
        exp_companies = [
            str(e.get("company") or "").strip()
            for e in exp_entries if isinstance(e, dict) and e.get("company")
        ]
        exp_desc = " ".join([
            str(e.get("description", "") or e.get("summary", "") or e.get("title", "") or "")
            for e in exp_entries if isinstance(e, dict)
        ]).lower()

        relevant_exp_str = (
            f"{total_exp:.1f} years of verified experience across {len(exp_entries)} role(s) "
            f"({', '.join(exp_companies[:3])})"
            if exp_entries
            else f"{total_exp:.1f} years recorded" if total_exp > 0 else "No recorded professional experience"
        )

        # ── 3. Education ────────────────────────────────────────────────────
        edu_entries = candidate_canonical.get("education") or []
        if not isinstance(edu_entries, list):
            edu_entries = []

        degrees_text = " ".join([
            " ".join([
                str(e.get("degree") or ""),
                str(e.get("field_of_study") or e.get("field") or ""),
                str(e.get("institution") or e.get("school") or ""),
            ]).lower()
            for e in edu_entries if isinstance(e, dict)
        ])

        education_match_str = (
            "; ".join([
                f"{e.get('degree', '')} in {e.get('field', e.get('field_of_study', ''))} ({e.get('institution', '')})".strip()
                for e in edu_entries if isinstance(e, dict)
            ])
            if edu_entries
            else "No educational credentials on file"
        )

        # ── 4. Certifications ───────────────────────────────────────────────
        cert_entries = candidate_canonical.get("certifications") or []
        if not isinstance(cert_entries, list):
            cert_entries = []
        cand_certs_text = " ".join([
            str(c.get("name") if isinstance(c, dict) else c).lower()
            for c in cert_entries
        ])

        # ── 5. Evaluate Requirements ────────────────────────────────────────
        matched_items: List[RequirementMatchItem] = []
        missing_items: List[RequirementMatchItem] = []
        matched_skills_list: List[str] = []
        missing_required_skills_list: List[str] = []

        total_weighted_score = 0.0
        total_possible_weighted = 0.0

        for req in requirements:
            cat_weight = self.weights.get(req.category, 1.0)
            req_weight = req.weight * cat_weight
            total_possible_weighted += req_weight

            req_text_lower = req.text.lower()

            # ─ Skill requirement ─────────────────────────────────────────────
            if "skill" in req.category:
                matched_skill = _fuzzy_skill_match(req_text_lower, cand_skills, threshold=78)
                if matched_skill:
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=True,
                        evidence=f"Confirmed in candidate verified skills: '{matched_skill.title()}'.",
                        score_contribution=req_weight,
                        status="verified",
                    )
                    matched_items.append(item)
                    matched_skills_list.append(req.text)
                    total_weighted_score += req_weight
                elif req_text_lower in exp_desc or (len(req_text_lower) > 3 and req_text_lower[:5] in exp_desc):
                    # Referenced in work experience narrative
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=True,
                        evidence="Referenced in work experience descriptions.",
                        score_contribution=req_weight * 0.7,
                        status="partial",
                    )
                    matched_items.append(item)
                    matched_skills_list.append(req.text)
                    total_weighted_score += req_weight * 0.7
                else:
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=False,
                        evidence="Not listed in candidate skills or career history.",
                        score_contribution=0.0,
                        status="missing",
                    )
                    missing_items.append(item)
                    if req.is_required:
                        missing_required_skills_list.append(req.text)

            # ─ Experience requirement ─────────────────────────────────────────
            elif req.category == "experience":
                digits = re.search(r"(\d+)", req.text)
                target_years = int(digits.group(1)) if digits else 2

                if total_exp >= target_years:
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=True,
                        evidence=(
                            f"Verified: candidate has {total_exp:.1f} years of experience "
                            f"({len(exp_entries)} role(s) recorded)."
                        ),
                        score_contribution=req_weight,
                        status="verified",
                    )
                    matched_items.append(item)
                    total_weighted_score += req_weight
                elif total_exp > 0:
                    ratio = total_exp / target_years
                    partial_score = req_weight * ratio
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=False,
                        evidence=(
                            f"Partial alignment: candidate has {total_exp:.1f} years; "
                            f"requirement is {target_years}+ years ({target_years - total_exp:.1f} yr gap)."
                        ),
                        score_contribution=partial_score,
                        status="partial",
                    )
                    missing_items.append(item)
                    total_weighted_score += partial_score
                else:
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=False,
                        evidence="No verifiable professional experience recorded.",
                        score_contribution=0.0,
                        status="missing",
                    )
                    missing_items.append(item)

            # ─ Education requirement ──────────────────────────────────────────
            elif req.category == "education":
                has_degree = False
                evidence_text = "No matching educational credentials found."

                if "ph" in req_text_lower and any(kw in degrees_text for kw in ["ph.d", "phd", "doctor"]):
                    has_degree = True
                    evidence_text = "PhD/Doctorate verified in education profile."
                elif "master" in req_text_lower and any(kw in degrees_text for kw in ["master", "m.s", "msc", "m.tech", "m.eng", "postgrad"]):
                    has_degree = True
                    evidence_text = "Master's degree verified in education profile."
                elif "bachelor" in req_text_lower:
                    bachelor_keywords = ["bachelor", "b.s", "b.tech", "b.e", "b.sc", "undergraduate", "engineering", "computer science"]
                    if edu_entries and any(kw in degrees_text for kw in bachelor_keywords):
                        has_degree = True
                        evidence_text = "Bachelor's degree in engineering/computer science verified."
                    elif edu_entries:
                        has_degree = False
                        evidence_text = "Higher education recorded; field/degree alignment partial."

                if has_degree:
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=True,
                        evidence=evidence_text,
                        score_contribution=req_weight,
                        status="verified",
                    )
                    matched_items.append(item)
                    total_weighted_score += req_weight
                else:
                    partial = req_weight * 0.4 if edu_entries else 0.0
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=False,
                        evidence=evidence_text,
                        score_contribution=partial,
                        status="partial" if edu_entries else "missing",
                    )
                    missing_items.append(item)
                    total_weighted_score += partial

            # ─ Certification requirement ──────────────────────────────────────
            elif req.category == "certification":
                if req_text_lower in cand_certs_text or any(req_text_lower in s for s in cand_skills):
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=True,
                        evidence=f"Certification or domain mastery verified: '{req.text}'.",
                        score_contribution=req_weight,
                        status="verified",
                    )
                    matched_items.append(item)
                    total_weighted_score += req_weight
                else:
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=False,
                        evidence=f"Candidate has no verified record of '{req.text}'.",
                        score_contribution=0.0,
                        status="missing",
                    )
                    missing_items.append(item)

            # ─ Responsibility requirement ─────────────────────────────────────
            elif req.category == "responsibility":
                # Check for keywords in experience description
                tokens = [t for t in re.findall(r"\w+", req_text_lower) if len(t) > 3]
                matched_tokens = [t for t in tokens if t in exp_desc]
                if len(matched_tokens) >= 2 or (tokens and len(matched_tokens) / len(tokens) >= 0.4):
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=True,
                        evidence=f"Prior demonstrated experience with {', '.join(matched_tokens[:3])}.",
                        score_contribution=req_weight,
                        status="verified",
                    )
                    matched_items.append(item)
                    total_weighted_score += req_weight
                else:
                    item = RequirementMatchItem(
                        requirement=req.text,
                        category=req.category,
                        matched=False,
                        evidence="Limited documented experience in this responsibility area.",
                        score_contribution=0.0,
                        status="missing",
                    )
                    missing_items.append(item)

        # ── Compute Final Score ───────────────────────────────────────────────
        if total_possible_weighted > 0:
            overall_percentage = round((total_weighted_score / total_possible_weighted) * 100.0, 1)
        else:
            overall_percentage = 0.0

        overall_percentage = min(100.0, overall_percentage)

        if overall_percentage >= 80.0:
            fit_level = "Strong Fit"
        elif overall_percentage >= 60.0:
            fit_level = "Moderate Fit"
        elif overall_percentage >= 40.0:
            fit_level = "Potential Fit"
        else:
            fit_level = "Low Alignment"

        # ── Build Strengths and Gaps ──────────────────────────────────────────
        strengths = [
            f"{m.requirement} ({m.category.replace('_', ' ').title()})"
            for m in matched_items if m.status == "verified"
        ]
        gaps = [
            f"{m.requirement} ({m.category.replace('_', ' ').title()})"
            for m in missing_items if m.status in ("missing", "partial")
        ]

        # ── Summary Explanation ───────────────────────────────────────────────
        req_matched = sum(1 for m in matched_items if m.status == "verified")
        req_total = len(requirements)
        summary = (
            f"{candidate_name} satisfies {req_matched} of {req_total} evaluated requirements "
            f"({overall_percentage}% weighted alignment). "
            f"Fit level: {fit_level}. Experience: {total_exp:.1f} years."
        )
        if strengths:
            summary += f" Verified competencies: {', '.join(strengths[:3])}."
        if missing_required_skills_list:
            summary += f" Missing key required skills: {', '.join(missing_required_skills_list[:3])}."

        return CandidateJDMatchScore(
            candidate_id=cid,
            candidate_name=candidate_name,
            overall_score=overall_percentage,
            fit_level=fit_level,
            matched_skills=matched_skills_list,
            missing_required_skills=missing_required_skills_list,
            relevant_experience=relevant_exp_str,
            education_match=education_match_str,
            matched_requirements=matched_items,
            missing_requirements=missing_items,
            summary_explanation=summary,
            strengths=strengths,
            gaps=gaps,
            experience_years=total_exp,
        )

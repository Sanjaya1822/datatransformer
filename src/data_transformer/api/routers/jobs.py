"""
Job Description Semantic Matching & Candidate Comparison API Router.

Enables recruiters to:
  - Enter or upload Job Descriptions (.txt, .pdf, .docx, .md)
  - Automatically parse criteria (required skills, preferred skills, experience, education)
  - Rank candidates using explainable evidence-based scoring
  - Compare candidates side-by-side against job requirements
  - Explain why one candidate is a better match than another
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from data_transformer.auth.dependencies import get_current_user
from data_transformer.db.database import get_db
from data_transformer.db.models import CandidateProfile, JobDescription, User
from data_transformer.db.repositories import JobDescriptionRepository
from data_transformer.jd_matching.matcher import (
    CandidateJDMatchScore,
    JDRequirement,
    JobDescriptionMatcher,
    JobDescriptionParser,
)

router = APIRouter(prefix="/jobs", tags=["job_matching"])


class JobDescriptionCreateRequest(BaseModel):
    title: str
    content: str


class MatchCandidatesRequest(BaseModel):
    candidate_ids: Optional[List[str]] = None
    min_score: float = 0.0


class CompareCandidatesJDRequest(BaseModel):
    candidate_id_a: str
    candidate_id_b: str


def _extract_text_from_file(filename: str, content: bytes) -> str:
    """Extract plain text from uploaded JD file (.txt, .md, .pdf, .docx)."""
    ext = Path(filename).suffix.lower()
    if ext in [".txt", ".md"]:
        return content.decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            pages_text = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages_text)
        except Exception:
            return content.decode("utf-8", errors="ignore")
    elif ext in [".docx", ".doc"]:
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([p.text for p in doc.paragraphs if p.text])
        except Exception:
            return content.decode("utf-8", errors="ignore")
    return content.decode("utf-8", errors="ignore")


@router.get("/", response_model=list)
def list_job_descriptions(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """Lists saved job descriptions."""
    repo = JobDescriptionRepository(db)
    items = repo.list(skip=skip, limit=limit)
    return [
        {
            "id": jd.id,
            "title": jd.title,
            "requirements_count": len(jd.extracted_requirements_json or []),
            "created_at": jd.created_at.isoformat() if jd.created_at else None,
        }
        for jd in items
    ]


@router.post("/", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_job_description(
    req: JobDescriptionCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Creates a job description from text and automatically extracts requirements."""
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(400, "Title and content cannot be empty.")

    requirements = JobDescriptionParser.parse_requirements(req.content)
    reqs_json = [
        {
            "category": r.category,
            "text": r.text,
            "weight": r.weight,
            "is_required": r.is_required,
        }
        for r in requirements
    ]

    repo = JobDescriptionRepository(db)
    jd = repo.create(
        title=req.title,
        content=req.content,
        requirements=reqs_json,
        created_by=current_user.id,
    )

    return {
        "id": jd.id,
        "title": jd.title,
        "extracted_requirements": reqs_json,
        "created_at": jd.created_at.isoformat(),
    }


@router.post("/upload", response_model=dict, status_code=status.HTTP_201_CREATED)
async def upload_job_description(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    db: Annotated[Session, Depends(get_db)] = None,
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    """Uploads a job description file (.pdf, .docx, .txt, .md) and parses requirements."""
    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(400, "Uploaded file is empty.")

    text = _extract_text_from_file(file.filename, content_bytes)
    if not text.strip():
        raise HTTPException(400, f"Could not extract text from '{file.filename}'.")

    # Determine job title
    jd_title = (title or "").strip()
    if not jd_title:
        # Use first non-empty line or filename
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and len(lines[0]) < 80:
            jd_title = lines[0]
        else:
            jd_title = Path(file.filename).stem.replace("_", " ").replace("-", " ").title()

    requirements = JobDescriptionParser.parse_requirements(text)
    reqs_json = [
        {
            "category": r.category,
            "text": r.text,
            "weight": r.weight,
            "is_required": r.is_required,
        }
        for r in requirements
    ]

    repo = JobDescriptionRepository(db)
    jd = repo.create(
        title=jd_title,
        content=text,
        requirements=reqs_json,
        created_by=current_user.id,
    )

    return {
        "id": jd.id,
        "title": jd.title,
        "filename": file.filename,
        "extracted_requirements": reqs_json,
        "created_at": jd.created_at.isoformat(),
    }


@router.get("/{jd_id}", response_model=dict)
def get_job_description(
    jd_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Retrieves job description details and requirement schema."""
    repo = JobDescriptionRepository(db)
    jd = repo.get(jd_id)
    if not jd:
        raise HTTPException(404, f"Job Description '{jd_id}' not found.")

    return {
        "id": jd.id,
        "title": jd.title,
        "content": jd.content,
        "extracted_requirements": jd.extracted_requirements_json,
        "created_at": jd.created_at.isoformat() if jd.created_at else None,
    }


@router.post("/{jd_id}/match", response_model=dict)
def match_candidates_against_job(
    jd_id: str,
    req: MatchCandidatesRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Evaluates candidate profiles against job requirements.
    Produces transparent match scores, strengths, missing skills, experience gaps,
    verified profile evidence, and comparative ranking explanations.
    """
    try:
        repo = JobDescriptionRepository(db)
        jd = repo.get(jd_id)
        if not jd:
            raise HTTPException(404, f"Job Description '{jd_id}' not found.")

        requirements = [
            JDRequirement(
                category=r.get("category", "skill"),
                text=r.get("text", ""),
                weight=float(r.get("weight", 0.1)),
                is_required=r.get("is_required", False),
            )
            for r in (jd.extracted_requirements_json or [])
        ]
        if not requirements:
            requirements = JobDescriptionParser.parse_requirements(jd.content)

        q = db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False)
        if req.candidate_ids:
            q = q.filter(CandidateProfile.candidate_id.in_(req.candidate_ids))
        profiles = q.all()

        matcher = JobDescriptionMatcher()
        match_results = []

        for p in profiles:
            canon = p.canonical_json or {}
            score_res: CandidateJDMatchScore = matcher.match_candidate(canon, requirements)

            if score_res.overall_score >= req.min_score:
                matched_dicts = [
                    {
                        "requirement": m.requirement,
                        "category": m.category,
                        "evidence": m.evidence,
                        "score_contribution": m.score_contribution,
                        "status": m.status,
                    }
                    for m in score_res.matched_requirements
                ]
                missing_dicts = [
                    {
                        "requirement": m.requirement,
                        "category": m.category,
                        "evidence": m.evidence,
                        "score_contribution": m.score_contribution,
                        "status": m.status,
                    }
                    for m in score_res.missing_requirements
                ]

                match_results.append({
                    "candidate_id": p.candidate_id,
                    "name": score_res.candidate_name,
                    "overall_score": score_res.overall_score,
                    "fit_level": score_res.fit_level,
                    "experience_years": score_res.experience_years,
                    "matched_skills": score_res.matched_skills,
                    "missing_required_skills": score_res.missing_required_skills,
                    "relevant_experience": score_res.relevant_experience,
                    "education_match": score_res.education_match,
                    "strengths": score_res.strengths,
                    "gaps": score_res.gaps,
                    "summary": score_res.summary_explanation,
                    "advisory_note": score_res.advisory_note,
                    "matched_requirements": matched_dicts,
                    "missing_requirements": missing_dicts,
                })

                repo.save_match_result(
                    jd_id=jd.id,
                    profile_id=p.id,
                    score=score_res.overall_score,
                    explanation={"summary": score_res.summary_explanation, "fit_level": score_res.fit_level},
                    matched=matched_dicts,
                    missing=missing_dicts,
                )

        # Sort descending by match score
        match_results.sort(key=lambda r: r["overall_score"], reverse=True)

        # Compute comparative ranking explanations
        for idx, r in enumerate(match_results):
            r["rank"] = idx + 1
            if idx == 0:
                if len(match_results) > 1:
                    next_c = match_results[1]
                    delta = round(r["overall_score"] - next_c["overall_score"], 1)
                    reasons = []
                    unique_strengths = set(r["strengths"]) - set(next_c["strengths"])
                    if unique_strengths:
                        reasons.append(f"offers additional verified skills: {', '.join(list(unique_strengths)[:3])}")
                    if r["experience_years"] > next_c["experience_years"]:
                        exp_diff = round(r["experience_years"] - next_c["experience_years"], 1)
                        reasons.append(f"has {exp_diff} more years of recorded experience")
                    reason_str = "; ".join(reasons) if reasons else f"higher overall requirement alignment (+{delta}%)"
                    r["why_better_match"] = f"Top-ranked candidate. Outperforms #{2} ({next_c['name']}) by +{delta}%: {reason_str}."
                else:
                    r["why_better_match"] = "Sole candidate evaluated; matches key core requirements."
            else:
                prev_c = match_results[idx - 1]
                delta = round(prev_c["overall_score"] - r["overall_score"], 1)
                missing_rel = set(prev_c["strengths"]) - set(r["strengths"])
                gap_str = f"Lacks skills present in #{idx} ({prev_c['name']}): {', '.join(list(missing_rel)[:3])}." if missing_rel else f"Scores {delta}% lower on requirement depth."
                r["why_better_match"] = f"Ranked below #{idx} ({prev_c['name']}) by {delta}%. {gap_str}"

        return {
            "job_id": jd.id,
            "job_title": jd.title,
            "total_evaluated": len(profiles),
            "total_returned": len(match_results),
            "rankings": match_results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, detail=f"Candidate matching error: {str(exc)}")


@router.post("/{jd_id}/compare", response_model=dict)
def compare_two_candidates_against_jd(
    jd_id: str,
    req: CompareCandidatesJDRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Direct head-to-head comparison between two candidates for a specific Job Description.
    Explains which candidate is a stronger fit and why, breaking down skills, experience,
    and missing qualifications.
    """
    repo = JobDescriptionRepository(db)
    jd = repo.get(jd_id)
    if not jd:
        raise HTTPException(404, f"Job Description '{jd_id}' not found.")

    requirements = [
        JDRequirement(
            category=r.get("category", "skill"),
            text=r.get("text", ""),
            weight=float(r.get("weight", 0.1)),
            is_required=r.get("is_required", False),
        )
        for r in (jd.extracted_requirements_json or [])
    ]
    if not requirements:
        requirements = JobDescriptionParser.parse_requirements(jd.content)

    cand_repo_a = db.query(CandidateProfile).filter(CandidateProfile.candidate_id == req.candidate_id_a, CandidateProfile.is_deleted == False).first()
    cand_repo_b = db.query(CandidateProfile).filter(CandidateProfile.candidate_id == req.candidate_id_b, CandidateProfile.is_deleted == False).first()

    if not cand_repo_a or not cand_repo_b:
        raise HTTPException(404, "One or both candidate profiles were not found.")

    matcher = JobDescriptionMatcher()
    score_a = matcher.match_candidate(cand_repo_a.canonical_json or {}, requirements)
    score_b = matcher.match_candidate(cand_repo_b.canonical_json or {}, requirements)

    # Determine better candidate
    if score_a.overall_score > score_b.overall_score:
        winner = score_a.candidate_name
        winner_id = req.candidate_id_a
        delta = round(score_a.overall_score - score_b.overall_score, 1)
        unique_skills = set(score_a.strengths) - set(score_b.strengths)
        exp_diff = round(score_a.experience_years - score_b.experience_years, 1)
        reasons = []
        if unique_skills:
            reasons.append(f"has verified competencies in {', '.join(list(unique_skills)[:3])} that {score_b.candidate_name} lacks")
        if exp_diff > 0:
            reasons.append(f"brings {exp_diff} additional years of verified industry experience")
        comp_summary = f"{winner} is a stronger match (+{delta}%) because they {', and '.join(reasons) if reasons else 'have broader requirement alignment'}."
    elif score_b.overall_score > score_a.overall_score:
        winner = score_b.candidate_name
        winner_id = req.candidate_id_b
        delta = round(score_b.overall_score - score_a.overall_score, 1)
        unique_skills = set(score_b.strengths) - set(score_a.strengths)
        exp_diff = round(score_b.experience_years - score_a.experience_years, 1)
        reasons = []
        if unique_skills:
            reasons.append(f"has verified competencies in {', '.join(list(unique_skills)[:3])} that {score_a.candidate_name} lacks")
        if exp_diff > 0:
            reasons.append(f"brings {exp_diff} additional years of verified industry experience")
        comp_summary = f"{winner} is a stronger match (+{delta}%) because they {', and '.join(reasons) if reasons else 'have broader requirement alignment'}."
    else:
        winner = "Tie"
        winner_id = None
        comp_summary = f"Both candidates show equivalent alignment ({score_a.overall_score}%); further manual assessment recommended."

    # Requirement-by-requirement comparison table
    comparison_table = []
    for req_item in requirements:
        m_a = next((m for m in score_a.matched_requirements if m.requirement == req_item.text), None)
        m_b = next((m for m in score_b.matched_requirements if m.requirement == req_item.text), None)
        
        status_a = m_a.status if m_a else "missing"
        status_b = m_b.status if m_b else "missing"
        evidence_a = m_a.evidence if m_a else "Not matched"
        evidence_b = m_b.evidence if m_b else "Not matched"

        comparison_table.append({
            "requirement": req_item.text,
            "category": req_item.category,
            "is_required": req_item.is_required,
            "candidate_a": {
                "status": status_a,
                "evidence": evidence_a,
            },
            "candidate_b": {
                "status": status_b,
                "evidence": evidence_b,
            },
            "advantage": req.candidate_id_a if (status_a == "verified" and status_b != "verified") else (req.candidate_id_b if (status_b == "verified" and status_a != "verified") else "neutral")
        })

    return {
        "job_id": jd.id,
        "job_title": jd.title,
        "candidate_a": {
            "id": req.candidate_id_a,
            "name": score_a.candidate_name,
            "score": score_a.overall_score,
            "fit_level": score_a.fit_level,
            "experience_years": score_a.experience_years,
            "strengths": score_a.strengths,
            "gaps": score_a.gaps,
        },
        "candidate_b": {
            "id": req.candidate_id_b,
            "name": score_b.candidate_name,
            "score": score_b.overall_score,
            "fit_level": score_b.fit_level,
            "experience_years": score_b.experience_years,
            "strengths": score_b.strengths,
            "gaps": score_b.gaps,
        },
        "stronger_candidate": winner,
        "stronger_candidate_id": winner_id,
        "comparison_explanation": comp_summary,
        "matrix": comparison_table,
    }

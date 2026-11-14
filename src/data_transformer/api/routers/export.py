"""
Candidate Data Export API Router.

Supports:
  - JSON format export
  - CSV flat table export
  - Role-based PII redaction (mask_pii parameter)
"""
from __future__ import annotations

import csv
import io
import json
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from data_transformer.auth.dependencies import get_current_user
from data_transformer.db.database import get_db
from data_transformer.db.models import CandidateProfile, User
from data_transformer.security.pii_masker import PIIMasker

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/json")
def export_candidates_json(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    mask_pii: bool = Query(False),
):
    """Exports canonical candidate profiles as JSON."""
    profiles = db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False).all()
    results = []

    for p in profiles:
        canon = p.canonical_json or {}
        if mask_pii or current_user.role.value == "recruiter":
            canon = PIIMasker.mask_profile(canon, "recruiter")
        results.append(canon)

    payload = json.dumps({"count": len(results), "candidates": results}, indent=2)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=candidates_export.json"},
    )


@router.get("/csv")
def export_candidates_csv(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    mask_pii: bool = Query(False),
):
    """Exports candidate records as a flat CSV table."""
    profiles = db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False).all()

    output = io.StringIO()
    writer = csv.writer(output)

    # Headers
    writer.writerow([
        "candidate_id",
        "full_name",
        "primary_email",
        "primary_phone",
        "location",
        "skills",
        "experience_count",
        "education_count",
        "overall_confidence",
        "sources_merged",
    ])

    for p in profiles:
        canon = p.canonical_json or {}
        if mask_pii or current_user.role.value == "recruiter":
            canon = PIIMasker.mask_profile(canon, "recruiter")

        name_f = canon.get("full_name", {})
        name = name_f.get("value") if isinstance(name_f, dict) else str(name_f or "")

        emails = canon.get("emails") or []
        email = emails[0].get("value") if emails and isinstance(emails[0], dict) else (emails[0] if emails else "")

        phones = canon.get("phones") or []
        phone = phones[0].get("value") if phones and isinstance(phones[0], dict) else (phones[0] if phones else "")

        loc = canon.get("location") or {}
        loc_str = loc.get("formatted") or loc.get("city") or ""

        skills = [s.get("name", s) if isinstance(s, dict) else str(s) for s in (canon.get("skills") or [])]
        skills_str = ", ".join(skills[:8])

        exp_count = len(canon.get("experience") or [])
        edu_count = len(canon.get("education") or [])

        writer.writerow([
            p.candidate_id,
            name,
            email,
            phone,
            loc_str,
            skills_str,
            exp_count,
            edu_count,
            p.overall_confidence,
            ", ".join(p.sources_merged or []),
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=candidates_export.csv"},
    )

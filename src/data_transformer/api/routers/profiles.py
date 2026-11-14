"""
Candidate Profiles API Router.
CRUD for canonical profiles, version history, rollback, field overrides, and comparisons.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from data_transformer.auth.dependencies import get_current_user
from data_transformer.db.database import get_db
from data_transformer.db.models import ConflictDecision, User, UserRole
from data_transformer.db.repositories import AuditRepository, CandidateRepository
from data_transformer.deduplication.explainable_scorer import ExplainableScorer
from data_transformer.schema.canonical import RawRecord
from data_transformer.security.pii_masker import PIIMasker

router = APIRouter(prefix="/candidates", tags=["candidates"])


class ProfileSummary(BaseModel):
    id: str
    candidate_id: str
    overall_confidence: float
    sources_merged: Optional[List[str]] = None
    current_version: int
    created_at: str
    updated_at: str
    full_name: Optional[str] = None
    headline: Optional[str] = None
    primary_email: Optional[str] = None
    skills_count: int = 0


class FieldOverrideRequest(BaseModel):
    field_name: str
    new_value: Any
    reason: str


@router.get("/", response_model=dict)
def list_profiles(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
):
    """Lists candidates with search, filtering, and pagination. Redacts PII for recruiters."""
    repo = CandidateRepository(db)
    items, total = repo.list(skip=skip, limit=limit, search=search, min_confidence=min_confidence)

    summaries = []
    for p in items:
        canon = p.canonical_json or {}
        masked = PIIMasker.mask_profile(canon, current_user.role.value)

        name_f = masked.get("full_name", {})
        name = name_f.get("value") if isinstance(name_f, dict) else str(name_f or "Unknown")

        headline_f = masked.get("headline", {})
        headline = headline_f.get("value") if isinstance(headline_f, dict) else str(headline_f or "")

        emails = masked.get("emails") or []
        primary_email = emails[0].get("value") if emails and isinstance(emails[0], dict) else (emails[0] if emails else None)
        skills = masked.get("skills") or []

        summaries.append({
            "id": p.id,
            "candidate_id": p.candidate_id,
            "overall_confidence": p.overall_confidence,
            "sources_merged": p.sources_merged,
            "current_version": p.current_version,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "updated_at": p.updated_at.isoformat() if p.updated_at else None,
            "full_name": name,
            "headline": headline,
            "primary_email": primary_email,
            "skills_count": len(skills),
        })

    return {"total": total, "skip": skip, "limit": limit, "items": summaries}


@router.get("/{candidate_id}", response_model=dict)
def get_profile(
    candidate_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Retrieves full canonical details of a single candidate."""
    repo = CandidateRepository(db)
    profile = repo.get_by_candidate_id(candidate_id)
    if not profile:
        raise HTTPException(404, f"Candidate '{candidate_id}' not found.")

    canon = profile.canonical_json or {}
    masked = PIIMasker.mask_profile(canon, current_user.role.value)

    return {
        "id": profile.id,
        "candidate_id": profile.candidate_id,
        "canonical": masked,
        "projected": profile.projected_json,
        "overall_confidence": profile.overall_confidence,
        "sources_merged": profile.sources_merged,
        "current_version": profile.current_version,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


@router.get("/{candidate_id}/versions", response_model=list)
def get_version_history(
    candidate_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Returns version history and audit changelog for a profile."""
    repo = CandidateRepository(db)
    profile = repo.get_by_candidate_id(candidate_id)
    if not profile:
        raise HTTPException(404, f"Candidate '{candidate_id}' not found.")

    versions = repo.get_versions(profile.id)
    return [
        {
            "version": v.version,
            "changed_by": v.changed_by,
            "change_reason": v.change_reason,
            "changed_at": v.changed_at.isoformat() if v.changed_at else None,
            "field_diffs": v.field_diffs_json,
        }
        for v in versions
    ]


@router.post("/{candidate_id}/rollback/{target_version}", response_model=dict)
def rollback_profile(
    candidate_id: str,
    target_version: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Rolls back profile to a previous version snapshot with full audit log."""
    repo = CandidateRepository(db)
    audit = AuditRepository(db)

    profile = repo.get_by_candidate_id(candidate_id)
    if not profile:
        raise HTTPException(404, f"Candidate '{candidate_id}' not found.")

    ver = repo.get_version(profile.id, target_version)
    if not ver:
        raise HTTPException(404, f"Version {target_version} not found for candidate '{candidate_id}'.")

    old_version = profile.current_version
    diff = [{
        "field": "*",
        "old_value": f"version_{old_version}",
        "new_value": f"version_{target_version}",
        "change_type": "rollback",
    }]

    updated = repo.update(
        profile=profile,
        canonical=ver.snapshot_json,
        changed_by=current_user.id,
        reason=f"Rollback to snapshot v{target_version}",
        field_diffs=diff,
    )

    audit.log(
        action="profile_rollback",
        resource_type="candidate_profile",
        resource_id=candidate_id,
        user_id=current_user.id,
        old_value={"version": old_version},
        new_value={"version": updated.current_version, "restored_from": target_version},
    )

    return {
        "success": True,
        "candidate_id": candidate_id,
        "restored_version": target_version,
        "current_version": updated.current_version,
    }


@router.post("/{candidate_id}/override", response_model=dict)
def override_field_value(
    candidate_id: str,
    req: FieldOverrideRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Allows reviewers to manually override conflicting or incorrect field values."""
    repo = CandidateRepository(db)
    audit = AuditRepository(db)

    profile = repo.get_by_candidate_id(candidate_id)
    if not profile:
        raise HTTPException(404, f"Candidate '{candidate_id}' not found.")

    canonical = dict(profile.canonical_json)
    old_val = canonical.get(req.field_name)

    # Apply override (wrap in FieldValue structure if scalar or update dict)
    if isinstance(old_val, dict) and "value" in old_val:
        new_field_val = dict(old_val)
        new_field_val["value"] = req.new_value
        new_field_val["merge_reason"] = f"Manual override by {current_user.username}: {req.reason}"
        canonical[req.field_name] = new_field_val
    else:
        canonical[req.field_name] = req.new_value

    diff = [{
        "field": req.field_name,
        "old_value": old_val,
        "new_value": req.new_value,
        "change_type": "manual_override",
    }]

    updated = repo.update(
        profile=profile,
        canonical=canonical,
        changed_by=current_user.id,
        reason=req.reason,
        field_diffs=diff,
    )

    decision = ConflictDecision(
        profile_id=profile.id,
        field_name=req.field_name,
        selected_value_json=req.new_value,
        rejected_values_json=[{"value": old_val, "reason": "Manually superseded"}],
        resolution_method="manual_override",
        explanation=req.reason,
        is_manual_override=True,
        reviewer_id=current_user.id,
    )
    db.add(decision)
    db.commit()

    audit.log(
        action="field_override",
        resource_type="candidate_profile",
        resource_id=candidate_id,
        user_id=current_user.id,
        old_value={req.field_name: old_val},
        new_value={req.field_name: req.new_value},
    )

    return {
        "success": True,
        "candidate_id": candidate_id,
        "field": req.field_name,
        "new_value": req.new_value,
        "new_version": updated.current_version,
    }


@router.delete("/{candidate_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_profile(
    candidate_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Admin-only: Soft-deletes a candidate profile."""
    repo = CandidateRepository(db)
    audit = AuditRepository(db)

    profile = repo.get_by_candidate_id(candidate_id)
    if not profile:
        raise HTTPException(404, f"Candidate '{candidate_id}' not found.")

    repo.soft_delete(profile, deleted_by=current_user.id)
    audit.log(
        action="candidate_soft_delete",
        resource_type="candidate_profile",
        resource_id=candidate_id,
        user_id=current_user.id,
    )

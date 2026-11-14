"""
Conflict Resolution Workspace API Router.

Enables reviewers to inspect field conflicts, compare conflicting values
side-by-side with source reliability and extraction confidence, and override decisions.
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

router = APIRouter(prefix="/conflict", tags=["conflict_resolution"])


class ManualConflictResolutionRequest(BaseModel):
    field_name: str
    selected_value: Any
    reason: str


@router.get("/decisions", response_model=dict)
def list_conflict_decisions(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    candidate_id: Optional[str] = Query(None),
):
    """Lists historical conflict resolution decisions across canonical profiles."""
    q = db.query(ConflictDecision)
    if candidate_id:
        cand_repo = CandidateRepository(db)
        p = cand_repo.get_by_candidate_id(candidate_id)
        if p:
            q = q.filter(ConflictDecision.profile_id == p.id)

    total = q.count()
    items = q.order_by(ConflictDecision.decided_at.desc()).offset(skip).limit(limit).all()

    return {
        "total": total,
        "items": [
            {
                "id": d.id,
                "profile_id": d.profile_id,
                "field_name": d.field_name,
                "selected_value": d.selected_value_json,
                "rejected_values": d.rejected_values_json,
                "resolution_method": d.resolution_method,
                "explanation": d.explanation,
                "is_manual_override": d.is_manual_override,
                "reviewer_id": d.reviewer_id,
                "decided_at": d.decided_at.isoformat() if d.decided_at else None,
            }
            for d in items
        ],
    }


@router.get("/candidates/{candidate_id}/conflicts", response_model=dict)
def get_candidate_conflicts(
    candidate_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Retrieves field-level conflicts, alternative rejected values, and evidence for a candidate."""
    cand_repo = CandidateRepository(db)
    profile = cand_repo.get_by_candidate_id(candidate_id)
    if not profile:
        raise HTTPException(404, f"Candidate '{candidate_id}' not found.")

    canon = profile.canonical_json or {}
    merge_summary = canon.get("merge_summary", {})
    conflicted_fields = merge_summary.get("fields_conflicted", [])
    merge_decisions = merge_summary.get("merge_decisions", [])

    # Map decisions by field
    decision_map = {d.get("field"): d for d in merge_decisions if isinstance(d, dict)}

    conflicts = []
    for f in conflicted_fields:
        field_val = canon.get(f, {})
        val = field_val.get("value") if isinstance(field_val, dict) else field_val
        conf = field_val.get("confidence", 0.0) if isinstance(field_val, dict) else 0.0
        prov = field_val.get("provenance", []) if isinstance(field_val, dict) else []

        decision = decision_map.get(f, {})
        strategy = decision.get("resolver_strategy", "default")
        reason = decision.get("winning_reason", "Resolved via strategy")

        alternatives = []
        for p in prov:
            if isinstance(p, dict) and p.get("raw_value") != val:
                alternatives.append({
                    "source": p.get("source"),
                    "raw_value": p.get("raw_value"),
                    "confidence": p.get("confidence"),
                    "reason": p.get("reason"),
                })

        conflicts.append({
            "field": f,
            "selected_value": val,
            "confidence": conf,
            "resolution_strategy": strategy,
            "reasoning": reason,
            "supporting_provenance": prov,
            "alternative_values": alternatives,
        })

    return {
        "candidate_id": candidate_id,
        "total_conflicts": len(conflicts),
        "conflicts": conflicts,
    }


@router.post("/candidates/{candidate_id}/resolve", response_model=dict)
def resolve_candidate_conflict(
    candidate_id: str,
    req: ManualConflictResolutionRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Applies a reviewer's conflict resolution override to a candidate profile."""
    cand_repo = CandidateRepository(db)
    audit = AuditRepository(db)

    profile = cand_repo.get_by_candidate_id(candidate_id)
    if not profile:
        raise HTTPException(404, f"Candidate '{candidate_id}' not found.")

    canon = dict(profile.canonical_json)
    old_val = canon.get(req.field_name)

    if isinstance(old_val, dict) and "value" in old_val:
        updated_field = dict(old_val)
        updated_field["value"] = req.selected_value
        updated_field["merge_reason"] = f"Reviewer Conflict Override: {req.reason}"
        canon[req.field_name] = updated_field
    else:
        canon[req.field_name] = req.selected_value

    # Update summary to mark resolved
    ms = canon.setdefault("merge_summary", {})
    cf = ms.setdefault("fields_conflicted", [])
    if req.field_name in cf:
        cf.remove(req.field_name)

    diff = [{
        "field": req.field_name,
        "old_value": old_val,
        "new_value": req.selected_value,
        "change_type": "conflict_resolution_override",
    }]

    cand_repo.update(
        profile=profile,
        canonical=canon,
        changed_by=current_user.id,
        reason=req.reason,
        field_diffs=diff,
    )

    dec = ConflictDecision(
        profile_id=profile.id,
        field_name=req.field_name,
        selected_value_json=req.selected_value,
        rejected_values_json=[{"value": old_val, "reason": "Manually overridden by reviewer"}],
        resolution_method="manual_reviewer_override",
        explanation=req.reason,
        is_manual_override=True,
        reviewer_id=current_user.id,
    )
    db.add(dec)
    db.commit()

    audit.log(
        action="resolve_conflict",
        resource_type="candidate_profile",
        resource_id=candidate_id,
        user_id=current_user.id,
        old_value={req.field_name: old_val},
        new_value={req.field_name: req.selected_value, "reason": req.reason},
    )

    return {
        "success": True,
        "candidate_id": candidate_id,
        "field_name": req.field_name,
        "resolved_value": req.selected_value,
    }

"""
Matching, Entity Resolution & Active Learning API Router.

Endpoints:
  - GET /matching/queue: Prioritized review queue for uncertain matches
  - GET /matching/compare/{id_a}/{id_b}: Side-by-side candidate comparison
  - POST /matching/pairs/{pair_id}/review: Approve/reject proposed matches or merge
  - POST /matching/merge-manual: Manually merge two candidate clusters
  - POST /matching/split-manual: Manually split merged records into separate candidates
  - POST /matching/recalibrate: Recalibrate thresholds based on accumulated feedback
  - GET /matching/stats: Review queue and feedback statistics
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from data_transformer.auth.dependencies import get_current_user
from data_transformer.db.database import get_db
from data_transformer.db.models import (
    CandidateProfile,
    FeedbackLabel,
    MatchClassification,
    MatchingConfig,
    MatchPair,
    User,
    UserRole,
)
from data_transformer.db.repositories import (
    AuditRepository,
    CandidateRepository,
    MatchPairRepository,
)
from data_transformer.deduplication.explainable_scorer import ExplainableScorer
from data_transformer.matching.active_learning import (
    ActiveLearner,
    EvaluationMetrics,
    LabeledPair,
    MatchingConfigVersion,
)
from data_transformer.schema.canonical import RawRecord
from data_transformer.security.pii_masker import PIIMasker
from data_transformer.jd_matching.job_evaluator import JobCriteria, compare_and_recommend_candidates, DEFAULT_WEIGHTS

router = APIRouter(prefix="/matching", tags=["matching"])



class CandidateEvaluationRequest(BaseModel):
    candidate_id_a: str
    candidate_id_b: str
    job_title: str = "Software Engineer"
    job_description: Optional[str] = ""
    required_skills: Optional[List[str]] = None
    preferred_skills: Optional[List[str]] = None
    min_experience_years: Optional[float] = 0.0
    required_education: Optional[str] = "Bachelor's Degree"
    additional_requirements: Optional[str] = ""
    weights: Optional[Dict[str, float]] = None

class ReviewDecisionRequest(BaseModel):
    decision: str  # "approve_merge" | "reject_match" | "defer"
    notes: Optional[str] = None


class ManualMergeRequest(BaseModel):
    candidate_id_a: str
    candidate_id_b: str
    reason: str


class ManualSplitRequest(BaseModel):
    candidate_id: str
    source_to_detach: str
    new_candidate_id: Optional[str] = None
    reason: str


@router.get("/queue", response_model=dict)
def get_review_queue(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = Query(50, ge=1, le=200),
):
    """Retrieves uncertain matches queued for active learning review."""
    repo = MatchPairRepository(db)
    cand_repo = CandidateRepository(db)
    pairs = repo.get_review_queue(limit=limit)

    items = []
    for pair in pairs:
        pa = cand_repo.get_by_id(pair.profile_a_id)
        pb = cand_repo.get_by_id(pair.profile_b_id)
        if not pa or not pb:
            continue

        name_a = (pa.canonical_json.get("full_name", {}) or {}).get("value") if isinstance(pa.canonical_json.get("full_name"), dict) else str(pa.canonical_json.get("full_name") or "")
        name_b = (pb.canonical_json.get("full_name", {}) or {}).get("value") if isinstance(pb.canonical_json.get("full_name"), dict) else str(pb.canonical_json.get("full_name") or "")

        items.append({
            "pair_id": pair.id,
            "score": round(pair.score, 4),
            "classification": pair.classification.value,
            "signals": pair.signals_json,
            "candidate_a": {
                "id": pa.id,
                "candidate_id": pa.candidate_id,
                "name": name_a,
                "confidence": pa.overall_confidence,
                "sources": pa.sources_merged,
            },
            "candidate_b": {
                "id": pb.id,
                "candidate_id": pb.candidate_id,
                "name": name_b,
                "confidence": pb.overall_confidence,
                "sources": pb.sources_merged,
            },
            "created_at": pair.created_at.isoformat() if pair.created_at else None,
        })

    return {
        "total_uncertain": repo.count_uncertain(),
        "items": items,
    }



@router.post("/evaluate-candidates", response_model=dict)
@router.post("/compare", response_model=dict)
def evaluate_candidates_for_job(
    req: CandidateEvaluationRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Job-Specific Candidate Evaluation & Recommendation Engine.
    Evaluates Candidate A and Candidate B independently against job requirements across:
    technical skills, experience relevance, duration, education, projects, and certifications.
    Produces an explainable recommendation answering all core hiring questions.
    """
    cand_repo = CandidateRepository(db)
    pa = cand_repo.get_by_candidate_id(req.candidate_id_a)
    pb = cand_repo.get_by_candidate_id(req.candidate_id_b)

    if not pa or not pb:
        raise HTTPException(404, "One or both candidate profiles not found.")

    cand_a_dict = dict(pa.canonical_json or {})
    cand_a_dict.setdefault('candidate_id', pa.candidate_id)
    cand_b_dict = dict(pb.canonical_json or {})
    cand_b_dict.setdefault('candidate_id', pb.candidate_id)

    criteria = JobCriteria(
        job_title=req.job_title,
        job_description=req.job_description or '',
        required_skills=req.required_skills or [],
        preferred_skills=req.preferred_skills or [],
        min_experience_years=req.min_experience_years or 0.0,
        required_education=req.required_education or "Bachelor's Degree",
        additional_requirements=req.additional_requirements or '',
        weights=req.weights or dict(DEFAULT_WEIGHTS),
    )

    result = compare_and_recommend_candidates(
        cand_a_dict,
        cand_b_dict,
        criteria,
    )

    # Attach masked profiles for secure UI display
    result["candidate_a_profile"] = PIIMasker.mask_profile(pa.canonical_json or {}, current_user.role.value)
    result["candidate_b_profile"] = PIIMasker.mask_profile(pb.canonical_json or {}, current_user.role.value)

    return result

@router.get("/compare/{candidate_id_a}/{candidate_id_b}", response_model=dict)
def compare_candidates_side_by_side(
    candidate_id_a: str,
    candidate_id_b: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Side-by-side comparison of two candidates with accurate field diffs,
    transparent weighted similarity score, and identity evidence evaluation.
    """
    cand_repo = CandidateRepository(db)
    pa = cand_repo.get_by_candidate_id(candidate_id_a)
    pb = cand_repo.get_by_candidate_id(candidate_id_b)

    if not pa or not pb:
        raise HTTPException(404, "One or both candidate profiles not found.")

    from data_transformer.matching.candidate_identity import compare_candidates_detailed
    comp = compare_candidates_detailed(pa.canonical_json or {}, pb.canonical_json or {})

    # Role-based PII masking for candidate display
    canon_a_masked = PIIMasker.mask_profile(pa.canonical_json or {}, current_user.role.value)
    canon_b_masked = PIIMasker.mask_profile(pb.canonical_json or {}, current_user.role.value)

    # Also compute job suitability evaluation
    crit = JobCriteria(job_title="Software Engineer")
    job_eval = compare_and_recommend_candidates(pa.canonical_json or {}, pb.canonical_json or {}, crit)

    return {
        "candidate_a": canon_a_masked,
        "candidate_b": canon_b_masked,
        "match_score": comp["match_score"],
        "similarity_percentage": comp["similarity_percentage"],
        "classification": comp["classification"],
        "identity_verdict": comp["identity_verdict"],
        "is_same_candidate": comp["is_same_candidate"],
        "requires_review": comp["classification"] == "uncertain",
        "explanation": comp["explanation"],
        "score_breakdown": comp["score_breakdown"],
        "signals": [
            {
                "type": s["attribute"],
                "matched": s["status"] in ("Exact Match", "Partial Match"),
                "weight": s["weight"],
                "contribution": s["contribution"],
                "explanation": s["explanation"],
            }
            for s in comp["score_breakdown"]
        ],
        "field_diffs": comp["field_diffs"],
        "job_evaluation": job_eval,
    }


@router.post("/pairs/{pair_id}/review", response_model=dict)
def review_proposed_match(
    pair_id: str,
    req: ReviewDecisionRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Approve or reject a proposed match pair in the review queue."""
    repo = MatchPairRepository(db)
    audit = AuditRepository(db)

    pair = db.query(MatchPair).filter(MatchPair.id == pair_id).first()
    if not pair:
        raise HTTPException(404, f"Match pair '{pair_id}' not found.")

    is_merge = req.decision == "approve_merge"
    new_class = MatchClassification.high_confidence if is_merge else MatchClassification.non_match
    feedback_label = FeedbackLabel.match if is_merge else FeedbackLabel.non_match

    repo.review(pair=pair, classification=new_class, reviewer_id=current_user.id, merged=is_merge)
    repo.add_feedback(pair_id=pair.id, label=feedback_label, reviewer_id=current_user.id, notes=req.notes)

    audit.log(
        action="review_match_pair",
        resource_type="match_pair",
        resource_id=pair_id,
        user_id=current_user.id,
        new_value={"decision": req.decision, "merged": is_merge, "notes": req.notes},
    )

    return {
        "success": True,
        "pair_id": pair_id,
        "decision": req.decision,
        "classification": new_class.value,
    }


@router.post("/merge-manual", response_model=dict)
def manual_merge_candidates(
    req: ManualMergeRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Manually merge two candidate clusters into a single canonical record."""
    cand_repo = CandidateRepository(db)
    audit = AuditRepository(db)

    pa = cand_repo.get_by_candidate_id(req.candidate_id_a)
    pb = cand_repo.get_by_candidate_id(req.candidate_id_b)

    if not pa or not pb:
        raise HTTPException(404, "Both candidates must exist to perform a manual merge.")

    from data_transformer.matching.candidate_identity import merge_canonical_profiles
    canon_a = dict(pa.canonical_json or {})
    canon_b = dict(pb.canonical_json or {})
    merged_canonical = merge_canonical_profiles(canon_a, canon_b)

    # Update primary record A
    updated_a = cand_repo.update(
        profile=pa,
        canonical=merged_canonical,
        changed_by=current_user.id,
        reason=f"Manually merged with candidate '{pb.candidate_id}': {req.reason}",
    )
    # Re-link raw records of B to A for complete audit provenance
    from data_transformer.db.models import RawRecordDB
    db.query(RawRecordDB).filter(RawRecordDB.profile_id == pb.id).update({RawRecordDB.profile_id: pa.id})

    # Soft-delete merged record B
    cand_repo.soft_delete(pb, deleted_by=current_user.id)

    audit.log(
        action="manual_cluster_merge",
        resource_type="candidate_profile",
        resource_id=pa.candidate_id,
        user_id=current_user.id,
        new_value={"merged_with": pb.candidate_id, "reason": req.reason},
    )

    return {
        "success": True,
        "canonical_candidate_id": pa.candidate_id,
        "merged_candidate_id": pb.candidate_id,
        "new_version": updated_a.current_version,
    }


@router.post("/recalibrate", response_model=dict)
def recalibrate_matching_engine(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Recalibrates matching thresholds based on accumulated human reviewer feedback."""
    repo = MatchPairRepository(db)
    all_fb = repo.get_all_feedback()

    if not all_fb:
        raise HTTPException(400, "No feedback labels exist yet to recalibrate matching thresholds.")

    # Convert DB feedback to LabeledPairs
    labeled_pairs = []
    for fb in all_fb:
        pair = fb.match_pair
        if pair:
            labeled_pairs.append(
                LabeledPair(
                    pair_id=pair.id,
                    label=fb.label.value,
                    score=pair.score,
                    signals={},
                    timestamp=fb.timestamp.isoformat(),
                )
            )

    latest_config = db.query(MatchingConfig).order_by(MatchingConfig.version.desc()).first()
    cur_v = latest_config.version if latest_config else 1

    learner = ActiveLearner()
    new_config_v = learner.recalibrate(labeled_pairs, new_version=cur_v + 1)

    new_db_config = MatchingConfig(
        version=new_config_v.version,
        thresholds_json={
            "high_confidence_threshold": new_config_v.high_confidence_threshold,
            "uncertain_threshold": new_config_v.uncertain_threshold,
        },
        signal_weights_json={},
        is_active=True,
        created_by=current_user.id,
        notes=new_config_v.notes,
    )
    if latest_config:
        latest_config.is_active = False
    db.add(new_db_config)
    db.commit()

    return {
        "success": True,
        "previous_version": cur_v,
        "new_version": new_config_v.version,
        "high_confidence_threshold": new_config_v.high_confidence_threshold,
        "uncertain_threshold": new_config_v.uncertain_threshold,
        "validation_f1": new_config_v.validation_f1,
        "feedback_count": len(labeled_pairs),
        "notes": new_config_v.notes,
    }


@router.get("/stats", response_model=dict)
def matching_statistics(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    repo = MatchPairRepository(db)
    total_pairs = db.query(MatchPair).count()
    uncertain = repo.count_uncertain()
    merged = repo.count_merged()
    feedback = repo.get_all_feedback()

    matches = sum(1 for f in feedback if f.label == FeedbackLabel.match)
    non_matches = sum(1 for f in feedback if f.label == FeedbackLabel.non_match)

    active_cfg = db.query(MatchingConfig).filter(MatchingConfig.is_active == True).first()

    return {
        "total_pairs_evaluated": total_pairs,
        "uncertain_pending_review": uncertain,
        "reviewer_confirmed_merges": merged,
        "total_feedback_labels": len(feedback),
        "confirmed_matches": matches,
        "confirmed_non_matches": non_matches,
        "active_config": {
            "version": active_cfg.version if active_cfg else 1,
            "thresholds": active_cfg.thresholds_json if active_cfg else {},
        },
    }


def _canonical_to_flat_fields(canon: dict) -> dict:
    fields = {}
    name_f = canon.get("full_name", {})
    fields["full_name"] = name_f.get("value") if isinstance(name_f, dict) else str(name_f or "")

    emails = canon.get("emails") or []
    fields["emails"] = [e.get("value", e) if isinstance(e, dict) else e for e in emails]

    phones = canon.get("phones") or []
    fields["phones"] = [p.get("value", p) if isinstance(p, dict) else p for p in phones]

    fields["location"] = canon.get("location") or {}
    fields["skills"] = [s.get("name", s) if isinstance(s, dict) else s for s in (canon.get("skills") or [])]
    fields["experience"] = canon.get("experience") or []
    return fields

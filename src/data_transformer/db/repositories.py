"""
Database Repositories for DataTransformer.
Encapsulates all database query and persistence logic.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc, func, String
from sqlalchemy.orm import Session

from .models import (
    AuditLog,
    CandidateProfile,
    ConflictDecision,
    FeedbackLabel,
    JobDescription,
    JDMatchResult,
    MatchClassification,
    MatchFeedback,
    MatchingConfig,
    MatchPair,
    PipelineRun,
    ProfileVersion,
    RawRecordDB,
    RunStatus,
    UploadedFile,
    User,
    UserRole,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CandidateRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        candidate_id: str,
        canonical: dict,
        projected: Optional[dict] = None,
        confidence: float = 0.0,
        sources: Optional[list] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> CandidateProfile:
        profile = CandidateProfile(
            candidate_id=candidate_id,
            canonical_json=canonical,
            projected_json=projected,
            overall_confidence=confidence,
            sources_merged=sources or [],
            pipeline_run_id=pipeline_run_id,
            current_version=1,
        )
        self.db.add(profile)
        self.db.flush()

        # Create initial snapshot version 1
        initial_version = ProfileVersion(
            profile_id=profile.id,
            version=1,
            snapshot_json=canonical,
            field_diffs_json=[],
            change_reason="Initial canonical profile ingestion",
            changed_at=_now(),
        )
        self.db.add(initial_version)
        self.db.commit()
        self.db.refresh(profile)
        return profile

    def get_by_candidate_id(self, candidate_id: str) -> Optional[CandidateProfile]:
        return (
            self.db.query(CandidateProfile)
            .filter(
                CandidateProfile.candidate_id == candidate_id,
                CandidateProfile.is_deleted == False,
            )
            .first()
        )

    def get_by_id(self, profile_id: str) -> Optional[CandidateProfile]:
        return (
            self.db.query(CandidateProfile)
            .filter(
                CandidateProfile.id == profile_id,
                CandidateProfile.is_deleted == False,
            )
            .first()
        )

    def get_by_email(self, email: str) -> Optional[CandidateProfile]:
        if not email:
            return None
        norm_e = str(email).strip().lower()
        profiles = self.db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False).all()
        for p in profiles:
            c = p.canonical_json or {}
            for e in c.get("emails", []):
                val = (e.get("value") if isinstance(e, dict) else str(e)).strip().lower()
                if val == norm_e:
                    return p
        return None

    def get_by_phone(self, phone: str) -> Optional[CandidateProfile]:
        if not phone:
            return None
        import re
        norm_p = re.sub(r"\D", "", str(phone))[-10:]
        if not norm_p:
            return None
        profiles = self.db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False).all()
        for p in profiles:
            c = p.canonical_json or {}
            for ph in c.get("phones", []):
                val = re.sub(r"\D", "", ph.get("value") if isinstance(ph, dict) else str(ph))[-10:]
                if val == norm_p:
                    return p
        return None

    def get_by_name(self, name: str) -> Optional[CandidateProfile]:
        if not name:
            return None
        from rapidfuzz import fuzz
        norm_n = str(name).strip().lower()
        profiles = self.db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False).all()
        for p in profiles:
            c = p.canonical_json or {}
            pname = (c.get("full_name") or c.get("name") or "").strip().lower()
            if pname:
                tokens_p = set(pname.split())
                tokens_n = set(norm_n.split())
                if pname == norm_n or tokens_p == tokens_n or (tokens_n.issubset(tokens_p) and fuzz.token_set_ratio(pname, norm_n) >= 90):
                    return p
        return None

    def find_matching_candidate(self, p_dict: dict) -> Optional[CandidateProfile]:
        """
        Find an existing active candidate matching by identity evidence.
        - Uses email and normalized phone number as strong identity signals.
        - Uses name similarity and career/skill attributes as supporting signals.
        - Handles missing values without automatically treating them as mismatches.
        - Does NOT merge candidates based only on similar names.
        - Does NOT treat generated UUIDs or internal candidate IDs as identity evidence.
        """
        if not p_dict or not isinstance(p_dict, dict):
            return None

        from data_transformer.matching.candidate_identity import evaluate_candidate_identity

        profiles = self.db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False).all()
        for p in profiles:
            canon = p.canonical_json or {}
            eval_res = evaluate_candidate_identity(canon, p_dict)
            if eval_res["is_same_candidate"]:
                return p

        return None

    def list(
        self,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> Tuple[List[CandidateProfile], int]:
        q = self.db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False)
        if min_confidence is not None:
            q = q.filter(CandidateProfile.overall_confidence >= min_confidence)
        if search:
            # Flexible search over canonical JSON
            q = q.filter(CandidateProfile.canonical_json.cast(String).ilike(f"%{search}%"))

        total = q.count()
        items = q.order_by(desc(CandidateProfile.created_at)).offset(skip).limit(limit).all()
        return items, total

    def update(
        self,
        profile: CandidateProfile,
        canonical: dict,
        projected: Optional[dict] = None,
        changed_by: Optional[str] = None,
        reason: Optional[str] = None,
        field_diffs: Optional[list] = None,
    ) -> CandidateProfile:
        profile.canonical_json = canonical
        if projected is not None:
            profile.projected_json = projected
        profile.current_version += 1
        profile.updated_at = _now()

        version_entry = ProfileVersion(
            profile_id=profile.id,
            version=profile.current_version,
            snapshot_json=canonical,
            field_diffs_json=field_diffs or [],
            changed_by=changed_by,
            change_reason=reason or "Manual update / merge correction",
            changed_at=_now(),
        )
        self.db.add(version_entry)
        self.db.commit()
        self.db.refresh(profile)
        return profile

    def soft_delete(self, profile: CandidateProfile, deleted_by: Optional[str] = None) -> None:
        profile.is_deleted = True
        profile.updated_at = _now()
        self.db.commit()

    def get_versions(self, profile_id: str) -> List[ProfileVersion]:
        return (
            self.db.query(ProfileVersion)
            .filter(ProfileVersion.profile_id == profile_id)
            .order_by(ProfileVersion.version.asc())
            .all()
        )

    def get_version(self, profile_id: str, version: int) -> Optional[ProfileVersion]:
        return (
            self.db.query(ProfileVersion)
            .filter(
                ProfileVersion.profile_id == profile_id,
                ProfileVersion.version == version,
            )
            .first()
        )

    def count_active(self) -> int:
        return self.db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False).count()


class PipelineRunRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, run_id: str, config: Optional[dict] = None, triggered_by: Optional[str] = None) -> PipelineRun:
        run = PipelineRun(
            run_id=run_id,
            config_json=config or {},
            status=RunStatus.running,
            triggered_by=triggered_by,
            started_at=_now(),
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def complete(
        self,
        run: PipelineRun,
        summary: dict,
        report: dict,
        files: int,
        candidates: int,
        duplicates: int,
        failed: int,
        duration: float,
    ) -> PipelineRun:
        run.status = RunStatus.completed
        run.summary_json = summary
        run.quality_report_json = report
        run.files_processed = files
        run.candidates_detected = candidates
        run.duplicates_merged = duplicates
        run.failed_files = failed
        run.duration_seconds = duration
        run.finished_at = _now()
        self.db.commit()
        self.db.refresh(run)
        return run

    def fail(self, run: PipelineRun, error_message: str) -> PipelineRun:
        run.status = RunStatus.failed
        run.summary_json = {"error": error_message}
        run.finished_at = _now()
        self.db.commit()
        self.db.refresh(run)
        return run

    def get_by_run_id(self, run_id: str) -> Optional[PipelineRun]:
        return self.db.query(PipelineRun).filter(PipelineRun.run_id == run_id).first()

    def list(self, skip: int = 0, limit: int = 20) -> List[PipelineRun]:
        return (
            self.db.query(PipelineRun)
            .order_by(desc(PipelineRun.started_at))
            .offset(skip)
            .limit(limit)
            .all()
        )


class MatchPairRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        profile_a_id: str,
        profile_b_id: str,
        score: float,
        signals: list,
        classification: MatchClassification,
    ) -> MatchPair:
        pair = MatchPair(
            profile_a_id=profile_a_id,
            profile_b_id=profile_b_id,
            score=score,
            signals_json=signals,
            classification=classification,
            created_at=_now(),
        )
        self.db.add(pair)
        self.db.commit()
        self.db.refresh(pair)
        return pair

    def get_review_queue(self, limit: int = 50) -> List[MatchPair]:
        """Returns uncertain candidate matches prioritized for active learning."""
        return (
            self.db.query(MatchPair)
            .filter(
                MatchPair.classification == MatchClassification.uncertain,
                MatchPair.reviewed_at == None,
            )
            .order_by(desc(MatchPair.score))
            .limit(limit)
            .all()
        )

    def review(
        self,
        pair: MatchPair,
        classification: MatchClassification,
        reviewer_id: str,
        merged: bool = False,
    ) -> MatchPair:
        pair.classification = classification
        pair.reviewed_by = reviewer_id
        pair.reviewed_at = _now()
        pair.was_merged = merged
        self.db.commit()
        self.db.refresh(pair)
        return pair

    def add_feedback(
        self,
        pair_id: str,
        label: FeedbackLabel,
        reviewer_id: str,
        notes: Optional[str] = None,
    ) -> MatchFeedback:
        fb = MatchFeedback(
            pair_id=pair_id,
            label=label,
            reviewer_id=reviewer_id,
            notes=notes,
            timestamp=_now(),
        )
        self.db.add(fb)
        self.db.commit()
        self.db.refresh(fb)
        return fb

    def get_all_feedback(self) -> List[MatchFeedback]:
        return self.db.query(MatchFeedback).order_by(MatchFeedback.timestamp.asc()).all()

    def count_uncertain(self) -> int:
        return (
            self.db.query(MatchPair)
            .filter(
                MatchPair.classification == MatchClassification.uncertain,
                MatchPair.reviewed_at == None,
            )
            .count()
        )

    def count_merged(self) -> int:
        return self.db.query(MatchPair).filter(MatchPair.was_merged == True).count()


class AuditRepository:
    def __init__(self, db: Session):
        self.db = db

    def log(
        self,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        old_value: Any = None,
        new_value: Any = None,
        ip: Optional[str] = None,
    ) -> AuditLog:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_value_json=old_value,
            new_value_json=new_value,
            ip_address=ip,
            timestamp=_now(),
        )
        self.db.add(entry)
        self.db.commit()
        return entry

    def list(
        self,
        skip: int = 0,
        limit: int = 100,
        resource_type: Optional[str] = None,
    ) -> List[AuditLog]:
        q = self.db.query(AuditLog)
        if resource_type:
            q = q.filter(AuditLog.resource_type == resource_type)
        return q.order_by(desc(AuditLog.timestamp)).offset(skip).limit(limit).all()


class JobDescriptionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        title: str,
        content: str,
        requirements: list,
        created_by: Optional[str] = None,
    ) -> JobDescription:
        jd = JobDescription(
            title=title,
            content=content,
            extracted_requirements_json=requirements,
            created_by=created_by,
            created_at=_now(),
        )
        self.db.add(jd)
        self.db.commit()
        self.db.refresh(jd)
        return jd

    def list(self, skip: int = 0, limit: int = 20) -> List[JobDescription]:
        return (
            self.db.query(JobDescription)
            .order_by(desc(JobDescription.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get(self, jd_id: str) -> Optional[JobDescription]:
        return self.db.query(JobDescription).filter(JobDescription.id == jd_id).first()

    def save_match_result(
        self,
        jd_id: str,
        profile_id: str,
        score: float,
        explanation: dict,
        matched: list,
        missing: list,
    ) -> JDMatchResult:
        result = JDMatchResult(
            jd_id=jd_id,
            profile_id=profile_id,
            overall_score=score,
            explanation_json=explanation,
            matched_requirements_json=matched,
            missing_requirements_json=missing,
            created_at=_now(),
        )
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        return result

    def get_results_for_jd(self, jd_id: str) -> List[JDMatchResult]:
        return (
            self.db.query(JDMatchResult)
            .filter(JDMatchResult.jd_id == jd_id)
            .order_by(desc(JDMatchResult.overall_score))
            .all()
        )


class FileRepository:
    """Repository for managing uploaded source files, storage cleanup, and dependent records."""

    def __init__(self, db: Session):
        self.db = db

    def list(self, include_deleted: bool = False) -> List[UploadedFile]:
        query = self.db.query(UploadedFile)
        if not include_deleted:
            query = query.filter(UploadedFile.is_deleted == False)
        return query.order_by(desc(UploadedFile.uploaded_at)).all()

    def get(self, file_id: str) -> Optional[UploadedFile]:
        return self.db.query(UploadedFile).filter(UploadedFile.id == file_id).first()

    def get_by_filename(self, filename: str) -> Optional[UploadedFile]:
        return (
            self.db.query(UploadedFile)
            .filter(UploadedFile.filename == filename, UploadedFile.is_deleted == False)
            .first()
        )

    def create(
        self,
        filename: str,
        file_path: str,
        file_type: str = "unknown",
        file_size: int = 0,
        status: str = "processed",
        pipeline_run_id: Optional[str] = None,
    ) -> UploadedFile:
        f = UploadedFile(
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
            status=status,
            pipeline_run_id=pipeline_run_id,
            is_deleted=False,
            uploaded_at=_now(),
        )
        self.db.add(f)
        self.db.commit()
        self.db.refresh(f)
        return f

    def delete_file_and_cleanup(self, file_id: str) -> dict:
        """
        Permanently removes file from disk, removes its RawRecordDB entries,
        and carefully updates candidate profiles without destroying consolidated candidates
        that contain data from other sources.
        """
        f = self.get(file_id)
        if not f or f.is_deleted:
            return {"success": False, "error": f"File with id '{file_id}' not found."}

        # 1. Permanently remove file on disk
        if f.file_path:
            try:
                p = Path(f.file_path)
                if p.exists() and p.is_file():
                    p.unlink()
            except Exception:
                pass

        # 2. Find associated RawRecordDB entries
        raw_recs = self.db.query(RawRecordDB).filter(
            (RawRecordDB.file_id == f.id) |
            (RawRecordDB.source_path == f.file_path)
        ).all()

        affected_profile_ids = set()
        for r in raw_recs:
            if r.profile_id:
                affected_profile_ids.add(r.profile_id)
            self.db.delete(r)

        # Also find candidate profiles referencing this filename or file_type in sources_merged
        all_profiles = self.db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False).all()
        for p in all_profiles:
            sources = p.sources_merged or []
            if f.filename in sources or any(f.filename in str(s) for s in sources):
                affected_profile_ids.add(p.id)

        profiles_updated = 0
        profiles_deleted = 0

        # 3. Handle dependent CandidateProfile records
        for pid in affected_profile_ids:
            cand = self.db.query(CandidateProfile).filter(CandidateProfile.id == pid).first()
            if not cand:
                continue

            # Remaining raw records for this candidate
            remaining_raw = self.db.query(RawRecordDB).filter(
                RawRecordDB.profile_id == pid,
                RawRecordDB.file_id != f.id,
            ).all()

            current_sources = [
                s for s in (cand.sources_merged or [])
                if s != f.filename and f.filename not in str(s)
            ]

            # If there are other sources or raw records remaining: KEEP the candidate!
            if current_sources or remaining_raw:
                cand.sources_merged = current_sources
                
                # Clean up provenance in canonical_json
                canon = dict(cand.canonical_json or {})
                for field_key, field_val in list(canon.items()):
                    if isinstance(field_val, dict) and "provenance" in field_val:
                        prov_list = field_val.get("provenance", [])
                        filtered_prov = [
                            entry for entry in prov_list
                            if isinstance(entry, dict)
                            and entry.get("source_id") != f.filename
                            and entry.get("source") != f.filename
                            and f.filename not in str(entry.get("source", ""))
                        ]
                        field_val["provenance"] = filtered_prov
                        canon[field_key] = field_val

                cand.canonical_json = canon
                cand.updated_at = _now()
                profiles_updated += 1
            else:
                # Only this source existed for this candidate: soft-delete candidate
                cand.is_deleted = True
                cand.updated_at = _now()
                profiles_deleted += 1

                # Clean up match pairs involving this deleted profile
                pairs = self.db.query(MatchPair).filter(
                    (MatchPair.profile_a_id == cand.id) | (MatchPair.profile_b_id == cand.id)
                ).all()
                for pair in pairs:
                    self.db.delete(pair)

        # 4. Mark or delete the uploaded file
        f.is_deleted = True
        self.db.commit()

        return {
            "success": True,
            "deleted_file_id": file_id,
            "filename": f.filename,
            "profiles_updated": profiles_updated,
            "profiles_deleted": profiles_deleted,
            "message": f"File '{f.filename}' permanently removed.",
        }

    def batch_delete(self, file_ids: List[str]) -> dict:
        """Permanently deletes multiple files and aggregates dependent profile updates."""
        results = []
        total_profiles_updated = 0
        total_profiles_deleted = 0
        deleted_count = 0

        for fid in file_ids:
            res = self.delete_file_and_cleanup(fid)
            if res.get("success"):
                deleted_count += 1
                total_profiles_updated += res.get("profiles_updated", 0)
                total_profiles_deleted += res.get("profiles_deleted", 0)
            results.append(res)

        return {
            "success": True,
            "deleted_count": deleted_count,
            "profiles_updated": total_profiles_updated,
            "profiles_deleted": total_profiles_deleted,
            "details": results,
            "message": f"Successfully deleted {deleted_count} file(s).",
        }

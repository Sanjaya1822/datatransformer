"""
Uploaded Files Management API Router.
Handles listing, deleting, and batch deletion of uploaded source files and their dependent records.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from data_transformer.auth.dependencies import get_current_user
from data_transformer.db.database import get_db
from data_transformer.db.models import CandidateProfile, RawRecordDB, UploadedFile, User
from data_transformer.db.repositories import FileRepository, CandidateRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


class BatchDeleteRequest(BaseModel):
    file_ids: List[str]


class FileItemResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    uploaded_at: Optional[str] = None
    candidates_linked: int = 0


@router.get("/", response_model=dict)
def list_uploaded_files(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Lists all uploaded source files with processing status and linked candidate counts."""
    try:
        repo = FileRepository(db)
        files = repo.list(include_deleted=False)

        items = []
        for f in files:
            # Count how many active profiles reference this file via RawRecordDB
            linked_count = (
                db.query(RawRecordDB)
                .filter(
                    (RawRecordDB.file_id == f.id) | (RawRecordDB.source_path == f.file_path),
                    RawRecordDB.profile_id != None,
                )
                .distinct(RawRecordDB.profile_id)
                .count()
            )
            if linked_count == 0 and f.filename:
                # Fallback: search sources_merged JSON column
                from sqlalchemy import String
                linked_count = (
                    db.query(CandidateProfile)
                    .filter(
                        CandidateProfile.is_deleted == False,
                        CandidateProfile.sources_merged.cast(String).like(f"%{f.filename}%"),
                    )
                    .count()
                )

            items.append({
                "id": f.id,
                "filename": f.filename,
                "file_type": f.file_type or "unknown",
                "file_size": f.file_size or 0,
                "status": f.status or "processed",
                "uploaded_at": f.uploaded_at.isoformat() if f.uploaded_at else None,
                "candidates_linked": linked_count,
            })

        return {"total": len(items), "items": items}

    except Exception as exc:
        logger.exception("Error listing uploaded files: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list uploaded files: {str(exc)}",
        )


@router.get("/{file_id}", response_model=dict)
def get_uploaded_file(
    file_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Retrieve details for a single uploaded file."""
    f = db.query(UploadedFile).filter(UploadedFile.id == file_id, UploadedFile.is_deleted == False).first()
    if not f:
        raise HTTPException(status_code=404, detail=f"File with id '{file_id}' not found.")
    return {
        "id": f.id,
        "filename": f.filename,
        "file_type": f.file_type or "unknown",
        "file_size": f.file_size or 0,
        "file_path": f.file_path,
        "status": f.status or "processed",
        "uploaded_at": f.uploaded_at.isoformat() if f.uploaded_at else None,
    }


@router.post("/{file_id}/reprocess", response_model=dict)
def reprocess_uploaded_file(
    file_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Reprocess an individual uploaded file through the transformation pipeline."""
    f = db.query(UploadedFile).filter(UploadedFile.id == file_id, UploadedFile.is_deleted == False).first()
    if not f:
        raise HTTPException(status_code=404, detail=f"File with id '{file_id}' not found.")

    from pathlib import Path
    p = Path(f.file_path) if f.file_path else None
    if not p or not p.exists():
        raise HTTPException(
            status_code=400,
            detail=f"File '{f.filename}' does not exist on disk at path '{f.file_path}'. Cannot reprocess.",
        )

    try:
        from data_transformer.pipeline.runner import PipelineRunner
        from data_transformer.schema.canonical import Source

        config = {"clustering": {"auto_merge_threshold": 90.0, "manual_review_threshold": 70.0}}
        schema = {
            "fields": [
                {"path": "candidate_id"},
                {"path": "full_name"},
                {"path": "emails"},
                {"path": "phones"},
                {"path": "location"},
                {"path": "skills"},
                {"path": "experience"},
                {"path": "education"},
                {"path": "projects"},
                {"path": "certifications"},
            ]
        }
        runner = PipelineRunner(config, schema)
        source = Source(type=f.file_type or "resume", path=str(p))
        result = runner.run([source])

        import uuid
        cand_repo = CandidateRepository(db)
        for profile_data in result.profiles:
            cid = profile_data.get("candidate_id", str(uuid.uuid4()))
            conf = float(profile_data.get("overall_confidence", 0.85))
            sources_merged = [f.filename]
            existing = cand_repo.find_matching_candidate(profile_data)
            if existing:
                from data_transformer.matching.candidate_identity import merge_canonical_profiles
                merged_canonical = merge_canonical_profiles(existing.canonical_json or {}, profile_data)
                combined_sources = sorted(list(set((existing.sources_merged or []) + sources_merged)))
                existing.sources_merged = combined_sources
                existing.overall_confidence = float(merged_canonical.get("overall_confidence", conf))
                cand_repo.update(profile=existing, canonical=merged_canonical, projected=merged_canonical, reason=f"Reprocessed {f.filename}")
                prof_id = existing.id
            else:
                created = cand_repo.create(
                    candidate_id=cid,
                    canonical=profile_data,
                    projected=profile_data,
                    confidence=conf,
                    sources=sources_merged,
                )
                prof_id = created.id

            # Ensure RawRecordDB link exists for this file & profile
            existing_rec = db.query(RawRecordDB).filter(RawRecordDB.file_id == f.id, RawRecordDB.profile_id == prof_id).first()
            if not existing_rec:
                raw_rec = RawRecordDB(
                    profile_id=prof_id,
                    file_id=f.id,
                    source_type=f.file_type or "resume",
                    source_path=f.file_path,
                    raw_fields_json=profile_data,
                )
                db.add(raw_rec)

        f.status = "processed"
        db.commit()

        return {
            "success": True,
            "file_id": f.id,
            "filename": f.filename,
            "candidates_extracted": len(result.profiles),
            "status": "processed",
            "message": f"Successfully reprocessed file '{f.filename}'.",
        }
    except Exception as exc:
        f.status = "error"
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Reprocessing '{f.filename}' failed: {str(exc)}",
        )


@router.delete("/{file_id}", response_model=dict)
def delete_uploaded_file(
    file_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Permanently removes the selected file from storage, removes associated raw processing data,
    and updates dependent candidate records (preserving profiles that have other sources).
    """
    repo = FileRepository(db)
    result = repo.delete_file_and_cleanup(file_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "File not found."))
    return result


@router.post("/batch-delete", response_model=dict)
def batch_delete_uploaded_files(
    payload: BatchDeleteRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Deletes multiple selected uploaded files and aggregates dependent profile updates."""
    if not payload.file_ids:
        raise HTTPException(status_code=400, detail="No file_ids provided for batch deletion.")

    repo = FileRepository(db)
    result = repo.batch_delete(payload.file_ids)
    return result

"""
FastAPI application for the Candidate Data Transformer & Reconciliation Platform.
Integrates ingestion pipeline, intelligent matching, active learning, provenance,
knowledge graph, job description matching, and data quality analytics.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_transformer.auth.router import router as auth_router
from data_transformer.api.routers import (
    analytics_router,
    conflict_router,
    export_router,
    extraction_router,
    files_router,
    jobs_router,
    matching_router,
    profiles_router,
)
from data_transformer.auth.service import hash_password
from data_transformer.db.database import SessionLocal, create_all_tables
from data_transformer.db.models import (
    CandidateProfile,
    MatchClassification,
    MatchPair,
    PipelineRun,
    RawRecordDB,
    RunStatus,
    UploadedFile,
)
from data_transformer.db.repositories import CandidateRepository, FileRepository, PipelineRunRepository
from data_transformer.db.seeder import seed_database
from data_transformer.deduplication.explainable_scorer import ExplainableScorer
from data_transformer.pipeline.runner import PipelineRunner
from data_transformer.schema.canonical import RawRecord, Source

app = FastAPI(
    title="DataTransformer API",
    description="Multi-Source Candidate Reconciliation & Intelligence Platform",
    version="2.0",
)

# ─── CORS Middleware ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Lifecycle / Startup ────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    """Initializes database schema and default credentials."""
    create_all_tables()
    db = SessionLocal()
    try:
        seed_database(db, hash_password)
        # Seed initial uploaded files from data/samples into data/uploads if empty
        file_repo = FileRepository(db)
        if len(file_repo.list(include_deleted=True)) == 0:
            upload_dir = Path("./data/uploads").resolve()
            upload_dir.mkdir(parents=True, exist_ok=True)
            samples_dir = Path("./data/samples").resolve()
            if samples_dir.exists():
                import shutil
                for sf in samples_dir.glob("*.*"):
                    target = upload_dir / sf.name
                    if not target.exists():
                        shutil.copy2(sf, target)
                    ext = sf.suffix.lower()
                    stype = "resume" if ext in [".pdf", ".docx", ".doc"] else ("csv" if ext == ".csv" else ("ats" if "ats" in sf.name.lower() or ext == ".json" else "notes"))
                    file_repo.create(
                        filename=sf.name,
                        file_path=str(target),
                        file_type=stype,
                        file_size=target.stat().st_size if target.exists() else 0,
                        status="processed",
                    )
    finally:
        db.close()


# ─── Include Enhanced Routers ───────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(auth_router, prefix="/v1")
app.include_router(profiles_router, prefix="/v1")
app.include_router(matching_router, prefix="/v1")
app.include_router(extraction_router, prefix="/v1")
app.include_router(conflict_router, prefix="/v1")
app.include_router(jobs_router, prefix="/v1")
app.include_router(analytics_router, prefix="/v1")
app.include_router(export_router, prefix="/v1")
app.include_router(files_router, prefix="/v1")


class HealthCheck(BaseModel):
    status: str
    database: str
    version: str


@app.get("/v1/health", response_model=HealthCheck)
def health_check():
    return {"status": "ok", "database": "connected", "version": "2.0"}


@app.get("/")
def read_root():
    return RedirectResponse(url="/ui/index.html")


frontend_dir = Path(__file__).parent.parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ─── Main Batch Ingestion & Merging Endpoint (Preserved & Extended) ─────────
@app.post("/v1/merge")
async def merge_candidates(
    config_json: str = Form(...),
    output_schema_json: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    github_url: Optional[str] = Form(None),
    linkedin_url: Optional[str] = Form(None),
):
    """
    Accepts pipeline config, output schema, and multiple source files.
    Runs the full 8-stage transformation, saves profiles & run statistics to DB,
    and returns merged canonical profiles and quality metrics.
    """
    try:
        config = json.loads(config_json)
        schema = json.loads(output_schema_json)

        # Auto-translate simple schema to Projector format if missing
        if "fields" not in schema:
            keys = list(schema.keys())
            if "candidate_id" not in keys:
                keys.append("candidate_id")

            fields_config = [{"path": k} for k in keys]
            schema = {
                "fields": fields_config,
                "include_confidence": False,
                "include_provenance": False,
                "on_missing": "null",
            }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in config or schema payload")

    runner = PipelineRunner(config, schema)
    sources: List[Source] = []
    file_records_to_save: List[dict] = []

    # Permanent storage directory for uploaded files
    upload_dir = Path("./data/uploads").resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        dest_path = upload_dir / file.filename
        content = await file.read()
        with open(dest_path, "wb") as buffer:
            buffer.write(content)

        # Source type detection
        ext = dest_path.suffix.lower()
        src_type = "notes"
        if ext in [".pdf", ".docx", ".doc"]:
            src_type = "resume"
        elif ext == ".csv":
            src_type = "csv"
        elif "github" in file.filename.lower():
            src_type = "github"
        elif "linkedin" in file.filename.lower():
            src_type = "linkedin"
        elif "ats" in file.filename.lower() or ext == ".json":
            src_type = "ats"
        elif "resume" in file.filename.lower():
            src_type = "resume"

        sources.append(Source(type=src_type, path=str(dest_path)))
        file_records_to_save.append({
            "filename": file.filename,
            "path": str(dest_path),
            "type": src_type,
            "size": len(content),
        })

    # Handle URLs
    if github_url:
        username = github_url.strip("/").split("/")[-1]
        try:
            resp = requests.get(f"https://api.github.com/users/{username}", timeout=5.0)
            if resp.status_code == 200:
                gh_path = upload_dir / f"{username}_github.json"
                with open(gh_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                sources.append(Source(type="github", path=str(gh_path)))
                file_records_to_save.append({
                    "filename": f"{username}_github.json",
                    "path": str(gh_path),
                    "type": "github",
                    "size": len(resp.text.encode("utf-8")),
                })
        except Exception:
            pass

    if linkedin_url:
        username = linkedin_url.strip("/").split("/")[-1]
        li_path = upload_dir / f"{username}_linkedin.json"
        li_content = json.dumps({"linkedin_id": username, "name": username})
        with open(li_path, "w", encoding="utf-8") as f:
            f.write(li_content)
        sources.append(Source(type="linkedin", path=str(li_path)))
        file_records_to_save.append({
            "filename": f"{username}_linkedin.json",
            "path": str(li_path),
            "type": "linkedin",
            "size": len(li_content.encode("utf-8")),
        })

    # Execute Pipeline
    try:
        result = runner.run(sources)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(exc)}")

    # ── Persist to Database for Analytics, Review Queue, and Audit ──────────
    db = SessionLocal()
    try:
        run_repo = PipelineRunRepository(db)
        cand_repo = CandidateRepository(db)
        file_repo = FileRepository(db)

        run_id = result.report.get("run_id", f"run_{uuid.uuid4().hex[:8]}")
        p_run = run_repo.create(run_id=run_id, config=config)

        # Register uploaded files in database
        uploaded_db_files = {}
        for finfo in file_records_to_save:
            existing_f = file_repo.get_by_filename(finfo["filename"])
            if existing_f:
                existing_f.file_path = finfo["path"]
                existing_f.file_size = finfo["size"]
                existing_f.file_type = finfo["type"]
                existing_f.pipeline_run_id = p_run.id
                existing_f.is_deleted = False
                db.commit()
                uploaded_db_files[finfo["filename"]] = existing_f
            else:
                new_f = file_repo.create(
                    filename=finfo["filename"],
                    file_path=finfo["path"],
                    file_type=finfo["type"],
                    file_size=finfo["size"],
                    pipeline_run_id=p_run.id,
                )
                uploaded_db_files[finfo["filename"]] = new_f

        # Persist generated canonical profiles with accurate identity resolution & deduplication
        saved_profiles = []
        for p in result.profiles:
            cid = p.get("candidate_id", str(uuid.uuid4()))
            conf = float(p.get("overall_confidence", 0.85))
            # Accurate source-to-candidate mapping: only actual contributing files
            sources_merged = p.get("sources_merged") or p.get("merge_summary", {}).get("sources_merged", [])
            if not sources_merged:
                # If only one source file uploaded, use it
                if len(file_records_to_save) == 1:
                    sources_merged = [file_records_to_save[0]["filename"]]
                else:
                    sources_merged = []

            # Check if this canonical candidate already exists in the database
            existing = cand_repo.find_matching_candidate(p)
            if existing:
                from data_transformer.matching.candidate_identity import merge_canonical_profiles
                merged_canonical = merge_canonical_profiles(existing.canonical_json or {}, p)
                combined_sources = sorted(list(set((existing.sources_merged or []) + sources_merged)))
                existing.sources_merged = combined_sources
                existing.pipeline_run_id = p_run.id
                existing.overall_confidence = float(merged_canonical.get("overall_confidence", conf))
                updated = cand_repo.update(
                    profile=existing,
                    canonical=merged_canonical,
                    projected=merged_canonical,
                    reason=f"Pipeline batch run {run_id}",
                )
                saved_profiles.append(updated)
                curr_profile_id = updated.id
            else:
                created = cand_repo.create(
                    candidate_id=cid,
                    canonical=p,
                    projected=p,
                    confidence=conf,
                    sources=sources_merged,
                    pipeline_run_id=p_run.id,
                )
                saved_profiles.append(created)
                curr_profile_id = created.id

            # Create RawRecordDB links ONLY for the files that actually contributed to this candidate
            for fname in sources_merged:
                ufile = uploaded_db_files.get(fname) or file_repo.get_by_filename(fname)
                if ufile:
                    raw_rec = RawRecordDB(
                        profile_id=curr_profile_id,
                        pipeline_run_id=p_run.id,
                        file_id=ufile.id,
                        source_type=ufile.file_type,
                        source_path=ufile.file_path,
                        raw_fields_json=_canon_to_fields(p),
                    )
                    db.add(raw_rec)

        # Pairwise match evaluation for Review Queue & Active Learning
        scorer = ExplainableScorer()
        for i in range(len(saved_profiles)):
            for j in range(i + 1, len(saved_profiles)):
                p1 = saved_profiles[i]
                p2 = saved_profiles[j]

                rec1 = RawRecord(
                    source=(p1.sources_merged or ["unknown"])[0],
                    source_id=p1.candidate_id,
                    fields=_canon_to_fields(p1.canonical_json),
                )
                rec2 = RawRecord(
                    source=(p2.sources_merged or ["unknown"])[0],
                    source_id=p2.candidate_id,
                    fields=_canon_to_fields(p2.canonical_json),
                )

                eval_res = scorer.score(rec1, rec2)
                if eval_res.score >= 0.40:  # Save relevant comparisons
                    match_class = (
                        MatchClassification.high_confidence
                        if eval_res.classification == "high_confidence"
                        else (
                            MatchClassification.uncertain
                            if eval_res.classification == "uncertain"
                            else MatchClassification.non_match
                        )
                    )
                    signals_data = [
                        {
                            "type": s.signal_type,
                            "matched": s.matched,
                            "weight": s.weight,
                            "contribution": s.score_contribution,
                            "value_a": s.value_a,
                            "value_b": s.value_b,
                            "explanation": s.explanation,
                        }
                        for s in eval_res.signals
                    ]
                    db_pair = MatchPair(
                        profile_a_id=p1.id,
                        profile_b_id=p2.id,
                        score=eval_res.score,
                        signals_json=signals_data,
                        classification=match_class,
                    )
                    db.add(db_pair)

        summary = result.summary or {}
        run_repo.complete(
            run=p_run,
            summary=summary,
            report=result.report,
            files=summary.get("files_processed", len(sources)),
            candidates=summary.get("candidates_detected", len(result.profiles)),
            duplicates=summary.get("duplicates_merged", 0),
            failed=summary.get("failed_files", 0),
            duration=result.report.get("runtime", {}).get("total_seconds", 0.0),
        )

        db.commit()

    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database persistence error: {str(e)}")
    finally:
        db.close()

    return {
        "report": result.report,
        "profiles": result.profiles,
        "invalid_profiles": result.invalid_profiles,
        "summary": result.summary,
    }


def _canon_to_fields(canon: dict) -> dict:
    f = {}
    nf = canon.get("full_name", {})
    f["full_name"] = nf.get("value") if isinstance(nf, dict) else str(nf or "")
    emails = canon.get("emails") or []
    f["emails"] = [e.get("value", e) if isinstance(e, dict) else e for e in emails]
    phones = canon.get("phones") or []
    f["phones"] = [p.get("value", p) if isinstance(p, dict) else p for p in phones]
    f["location"] = canon.get("location") or {}
    f["skills"] = [s.get("name", s) if isinstance(s, dict) else s for s in (canon.get("skills") or [])]
    f["experience"] = canon.get("experience") or []
    return f

"""
Extraction & Self-Correction API Router.

Executes the Extractor–Validator–Refiner (EVR) workflow on uploaded documents
(PDF, DOCX, TXT) or raw text payloads.
"""
from __future__ import annotations

import os
import tempfile
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from data_transformer.auth.dependencies import get_current_user
from data_transformer.db.database import get_db
from data_transformer.db.models import User
from data_transformer.extraction.extractor import DocumentParseOutput, DocumentParser
from data_transformer.extraction.refiner import ExtractorValidatorRefiner
from data_transformer.security.file_validator import UploadFileValidator

router = APIRouter(prefix="/extraction", tags=["extraction"])


class RawTextExtractionRequest(BaseModel):
    text: str
    source_name: Optional[str] = "recruiter_notes.txt"
    use_llm: bool = False


@router.post("/upload-and-extract", response_model=dict)
async def upload_and_extract_document(
    file: UploadFile = File(...),
    use_llm: bool = Form(False),
    current_user: User = Depends(get_current_user),
):
    """
    Uploads a resume (PDF, DOCX, or TXT) and runs the Extractor-Validator-Refiner workflow.
    Validates file integrity, extracts structured attributes, corrects inconsistencies,
    and returns field-level confidence and evidence snippets.
    """
    content = await file.read()
    file_size = len(content)

    # 1. Validate file format and size
    try:
        UploadFileValidator.validate_file(file.filename, file_size)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Save temporarily and parse text
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc_output = DocumentParser.parse_file(tmp_path)
        doc_output.file_path = file.filename

        # 3. Execute Extractor-Validator-Refiner Loop
        evr = ExtractorValidatorRefiner(use_llm=use_llm)
        evr_result = evr.process(doc_output)

        evidence_dict = {
            k: {
                "text": span.text,
                "start": span.start_char,
                "end": span.end_char,
                "page": span.page_num,
                "file": span.source_file,
            }
            for k, span in evr_result.evidence_spans.items()
        }

        errors_list = [
            {
                "field": err.field,
                "type": err.error_type,
                "message": err.message,
                "severity": err.severity,
            }
            for err in evr_result.validation_errors
        ]

        return {
            "filename": file.filename,
            "file_type": doc_output.file_type,
            "raw_text_preview": doc_output.raw_text[:500] + ("..." if len(doc_output.raw_text) > 500 else ""),
            "extracted_fields": evr_result.candidate_fields,
            "field_confidence": evr_result.field_confidence,
            "evidence_spans": evidence_dict,
            "validation_errors": errors_list,
            "correction_attempts": evr_result.correction_attempts,
            "requires_human_review": evr_result.requires_human_review,
            "unresolved_fields": evr_result.unresolved_fields,
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/text", response_model=dict)
def extract_from_raw_text(
    req: RawTextExtractionRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Extracts structured candidate data directly from raw text or recruiter notes.
    """
    if not req.text.strip():
        raise HTTPException(400, "Text payload cannot be empty.")

    doc_output = DocumentParseOutput(
        raw_text=req.text,
        page_texts=[req.text],
        file_type="txt",
        file_path=req.source_name,
    )

    evr = ExtractorValidatorRefiner(use_llm=req.use_llm)
    evr_result = evr.process(doc_output)

    evidence_dict = {
        k: {
            "text": span.text,
            "start": span.start_char,
            "end": span.end_char,
            "page": span.page_num,
        }
        for k, span in evr_result.evidence_spans.items()
    }

    errors_list = [
        {
            "field": err.field,
            "type": err.error_type,
            "message": err.message,
            "severity": err.severity,
        }
        for err in evr_result.validation_errors
    ]

    return {
        "extracted_fields": evr_result.candidate_fields,
        "field_confidence": evr_result.field_confidence,
        "evidence_spans": evidence_dict,
        "validation_errors": errors_list,
        "correction_attempts": evr_result.correction_attempts,
        "requires_human_review": evr_result.requires_human_review,
        "unresolved_fields": evr_result.unresolved_fields,
    }

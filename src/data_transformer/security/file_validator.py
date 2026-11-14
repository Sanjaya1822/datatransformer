"""
File Validation and Upload Security Guard.

Guards the application against malicious file uploads:
  - Whitelists permissible document extensions (.pdf, .docx, .doc, .txt, .csv, .json)
  - Enforces maximum file size thresholds (default: 10 MB per file)
  - Enforces maximum batch quantities (default: 50 files)
  - Inspects file headers to block executable/binary disguised uploads
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, List, Tuple

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".csv", ".json"}
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
MAX_FILES_PER_BATCH = int(os.getenv("MAX_FILES_PER_BATCH", "50"))

MAGIC_SIGNATURES = {
    ".pdf": b"%PDF",
    ".docx": b"PK\x03\x04",
}


class FileValidationError(Exception):
    pass


class UploadFileValidator:
    """Validates uploaded files for type, extension, size, and header integrity."""

    @classmethod
    def validate_file(cls, filename: str, file_size: int, stream: Optional[BinaryIO] = None) -> None:
        ext = Path(filename).suffix.lower()

        # 1. Check extension whitelist
        if ext not in ALLOWED_EXTENSIONS:
            raise FileValidationError(
                f"File '{filename}' has unsupported extension '{ext}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )

        # 2. Check maximum size
        if file_size > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            raise FileValidationError(
                f"File '{filename}' exceeds maximum allowable size of {max_mb:.0f} MB."
            )

        # 3. Check magic bytes if stream available
        if stream and ext in MAGIC_SIGNATURES:
            pos = stream.tell()
            header = stream.read(8)
            stream.seek(pos)
            expected = MAGIC_SIGNATURES[ext]
            if not header.startswith(expected):
                raise FileValidationError(
                    f"File '{filename}' content header does not match expected '{ext}' binary format."
                )

    @classmethod
    def validate_batch(cls, files: List[Tuple[str, int]]) -> None:
        if len(files) > MAX_FILES_PER_BATCH:
            raise FileValidationError(
                f"Batch exceeds maximum limit of {MAX_FILES_PER_BATCH} files at once."
            )
        for fname, size in files:
            cls.validate_file(fname, size)

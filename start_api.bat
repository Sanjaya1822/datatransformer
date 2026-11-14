@echo off
REM ===========================================================================
REM DataTransformer API & UI Launcher (Windows Batch)
REM ===========================================================================

echo Starting Multi-Source Candidate Data Transformer...
echo UI will be accessible at: http://localhost:8000/ui/index.html
echo API documentation at:     http://localhost:8000/docs
echo.

uvicorn data_transformer.api.app:app --host 0.0.0.0 --port 8000 --reload

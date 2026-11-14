# ===========================================================================
# DataTransformer API & UI Launcher (PowerShell)
# ===========================================================================

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Multi-Source Candidate Data Transformer & Reconciliation Engine" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Web Application: http://localhost:8000/ui/index.html" -ForegroundColor Green
Write-Host "Interactive API: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "Default Roles:   admin (admin123) | reviewer (reviewer123) | recruiter (recruiter123)" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Cyan

uvicorn data_transformer.api.app:app --host 0.0.0.0 --port 8000 --reload

"""API Routers package."""
from .analytics import router as analytics_router
from .conflict import router as conflict_router
from .export import router as export_router
from .extraction import router as extraction_router
from .files import router as files_router
from .jobs import router as jobs_router
from .matching import router as matching_router
from .profiles import router as profiles_router

__all__ = [
    "profiles_router",
    "matching_router",
    "extraction_router",
    "conflict_router",
    "jobs_router",
    "analytics_router",
    "export_router",
    "files_router",
]

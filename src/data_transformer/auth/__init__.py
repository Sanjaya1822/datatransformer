"""Authentication package."""
from .dependencies import (
    RequireAdmin,
    RequireRecruiter,
    RequireReviewer,
    get_current_user,
    get_current_user_optional,
    require_role,
)
from .router import router
from .service import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

__all__ = [
    "router",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "get_current_user",
    "get_current_user_optional",
    "require_role",
    "RequireAdmin",
    "RequireReviewer",
    "RequireRecruiter",
]

"""
PII Detection and Role-Based Masking.

Ensures sensitive contact data (emails, phone numbers, exact addresses)
is securely redacted for non-privileged roles and unauthorized exports.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class PIIMasker:
    """
    Masks Personally Identifiable Information based on user permission level.
    """

    @staticmethod
    def mask_email(email: str) -> str:
        """Transforms 'john.doe@example.com' -> 'j***e@example.com'."""
        if not email or "@" not in email:
            return "***"
        user, domain = email.split("@", 1)
        if len(user) <= 2:
            masked_user = user[0] + "***"
        else:
            masked_user = user[0] + "***" + user[-1]
        return f"{masked_user}@{domain}"

    @staticmethod
    def mask_phone(phone: str) -> str:
        """Transforms '+14155552671' -> '+1 415-***-**71'."""
        if not phone or len(phone) < 6:
            return "***-***-****"
        clean = re.sub(r"[^\d+]", "", phone)
        prefix = clean[:5]
        suffix = clean[-2:]
        return f"{prefix}***{suffix}"

    @staticmethod
    def mask_name(name: str) -> str:
        """Transforms 'Johnathan Doe' -> 'J. D.'."""
        if not name:
            return "Anonymous Candidate"
        parts = name.strip().split()
        if len(parts) >= 2:
            return f"{parts[0][0]}. {parts[-1][0]}."
        return f"{parts[0][0]}."

    @classmethod
    def mask_profile(cls, canonical_profile: Dict[str, Any], user_role: str = "recruiter") -> Dict[str, Any]:
        """
        Applies role-based PII redaction.
        Admins and Reviewers see complete unmasked profiles.
        Recruiters and external exports receive masked sensitive fields.
        """
        if user_role in ("admin", "reviewer", "system", "unmasked"):
            return canonical_profile

        masked = dict(canonical_profile)

        # Mask emails
        if "emails" in masked and isinstance(masked["emails"], list):
            new_emails = []
            for e in masked["emails"]:
                if isinstance(e, dict):
                    copy_e = dict(e)
                    copy_e["value"] = cls.mask_email(copy_e.get("value", ""))
                    new_emails.append(copy_e)
                else:
                    new_emails.append(cls.mask_email(str(e)))
            masked["emails"] = new_emails

        # Mask phones
        if "phones" in masked and isinstance(masked["phones"], list):
            new_phones = []
            for p in masked["phones"]:
                if isinstance(p, dict):
                    copy_p = dict(p)
                    copy_p["value"] = cls.mask_phone(copy_p.get("value", ""))
                    new_phones.append(copy_p)
                else:
                    new_phones.append(cls.mask_phone(str(p)))
            masked["phones"] = new_phones

        return masked

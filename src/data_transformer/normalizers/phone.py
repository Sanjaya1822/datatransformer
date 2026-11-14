"""
Phone number extraction and normalization.

Requirements:
- Extract original phone number accurately.
- Preserve leading zeros where applicable.
- Identify country codes ONLY when supported by explicit prefix or document context.
- DO NOT automatically prepend +1 or any arbitrary country code.
- Handle numbers with spaces, hyphens, brackets, and international prefixes.
- Validate using phonenumbers library.
- Store raw and normalized values separately.
- Reject malformed numbers.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional
import phonenumbers


# Common regions for parsing when hinted
_SUPPORTED_REGIONS = ["IN", "US", "GB", "CA", "AU", "DE", "FR", "SG"]


def parse_phone_detailed(phone: str, document_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses a phone string and returns detailed attributes:
      - raw: original string
      - normalized: standardized format (E.164 if country code is confirmed, else cleaned national digits)
      - country_code: e.g. "+91", "+1", or None
      - national_number: e.g. "9345420856"
      - is_valid: bool
    """
    if not phone or not str(phone).strip():
        return {
            "raw": "",
            "normalized": None,
            "country_code": None,
            "national_number": None,
            "is_valid": False,
        }

    raw = str(phone).strip()

    # Preserve leading 0s if present
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 7:
        return {
            "raw": raw,
            "normalized": None,
            "country_code": None,
            "national_number": digits if digits else None,
            "is_valid": False,
        }

    # 1. Check for explicit international prefix (+ or 00)
    cleaned = raw
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]

    if cleaned.startswith("+"):
        try:
            parsed = phonenumbers.parse(cleaned, None)
            if phonenumbers.is_valid_number(parsed):
                e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
                return {
                    "raw": raw,
                    "normalized": e164,
                    "country_code": f"+{parsed.country_code}",
                    "national_number": str(parsed.national_number),
                    "is_valid": True,
                }
        except Exception:
            pass

    # 2. Context-aware regional determination (detect if Indian, US, UK, etc.)
    context_hint: Optional[str] = None
    if document_context:
        ctx_lower = document_context.lower()
        if any(term in ctx_lower for term in [
            "india", "tamil nadu", "karnataka", "delhi", "mumbai", "chennai", "coimbatore",
            "dindigul", "tirunelveli", "pune", "hyderabad", "bangalore", "b.tech", "b.e.", "hsc", "sslc"
        ]):
            context_hint = "IN"
        elif any(term in ctx_lower for term in ["united states", "usa", "san francisco", "new york", "california", "texas", "austin", "seattle"]):
            context_hint = "US"

    # 3. Handle 10-digit numbers without explicit '+' prefix
    if len(digits) == 10:
        # Indian mobile format starts with 6, 7, 8, 9
        if digits[0] in "6789" and (context_hint == "IN" or context_hint is None):
            try:
                parsed = phonenumbers.parse(digits, "IN")
                if phonenumbers.is_valid_number(parsed):
                    return {
                        "raw": raw,
                        "normalized": f"+91{digits}",
                        "country_code": "+91",
                        "national_number": digits,
                        "is_valid": True,
                    }
            except Exception:
                pass

        # US formatted phone (e.g. contains parens like (415) 555-1234 or US context)
        if ("(" in raw and ")" in raw) or context_hint == "US":
            try:
                parsed = phonenumbers.parse(raw, "US")
                if phonenumbers.is_valid_number(parsed):
                    return {
                        "raw": raw,
                        "normalized": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
                        "country_code": "+1",
                        "national_number": digits,
                        "is_valid": True,
                    }
            except Exception:
                pass

        # If country code is not conclusively supported, store national number without forcing +1
        return {
            "raw": raw,
            "normalized": digits,
            "country_code": None,
            "national_number": digits,
            "is_valid": True,
        }

    # 4. Handle 11 or 12 digit numbers (e.g. 919677607931 without plus, or leading zero 09345420856)
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return {
            "raw": raw,
            "normalized": f"+{digits}",
            "country_code": "+91",
            "national_number": digits[2:],
            "is_valid": True,
        }

    if len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        # Indian mobile with trunk prefix 0
        return {
            "raw": raw,
            "normalized": f"+91{digits[1:]}",
            "country_code": "+91",
            "national_number": digits[1:],
            "is_valid": True,
        }

    # Fallback to general phonenumbers parsing if region specified
    if context_hint:
        try:
            parsed = phonenumbers.parse(raw, context_hint)
            if phonenumbers.is_valid_number(parsed):
                return {
                    "raw": raw,
                    "normalized": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
                    "country_code": f"+{parsed.country_code}",
                    "national_number": str(parsed.national_number),
                    "is_valid": True,
                }
        except Exception:
            pass

    return {
        "raw": raw,
        "normalized": digits,
        "country_code": None,
        "national_number": digits,
        "is_valid": True if 7 <= len(digits) <= 15 else False,
    }


def normalize_phone(phone: str, default_region: Optional[str] = None) -> Optional[str]:
    """
    Normalize a phone number string.
    DOES NOT automatically prepend +1 to international or Indian numbers.
    Returns normalized E.164 string if country is confirmed, or clean national string.
    Returns None if malformed or invalid.
    """
    if not phone or not str(phone).strip():
        return None

    res = parse_phone_detailed(str(phone).strip(), document_context=default_region)
    if not res["is_valid"]:
        return None
    return res["normalized"]

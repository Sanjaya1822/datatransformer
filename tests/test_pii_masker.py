"""
Tests for the PIIMasker — role-based PII redaction module.

Covers:
- mask_email: boundary cases (short user, long user, missing @)
- mask_phone: standard international format, short strings
- mask_name: single name, multi-word name
- mask_profile: admin/reviewer see full data; recruiter gets masked values
"""
from __future__ import annotations

import pytest

from data_transformer.security.pii_masker import PIIMasker


# ---------------------------------------------------------------------------
# mask_email
# ---------------------------------------------------------------------------

class TestMaskEmail:
    def test_standard_email_masked(self):
        result = PIIMasker.mask_email("john.doe@example.com")
        assert result.endswith("@example.com")
        assert "***" in result
        # First and last char of username preserved
        assert result.startswith("j")
        assert result[result.index("***") + 3] == "e"

    def test_short_user_masked(self):
        """1-2 char username: first char + *** preserved."""
        result = PIIMasker.mask_email("ab@domain.org")
        assert "@domain.org" in result
        assert "***" in result

    def test_no_at_sign_returns_stars(self):
        result = PIIMasker.mask_email("not-an-email")
        assert result == "***"

    def test_empty_string_returns_stars(self):
        result = PIIMasker.mask_email("")
        assert result == "***"

    def test_domain_preserved(self):
        """Domain part is never masked."""
        result = PIIMasker.mask_email("alice@mycompany.io")
        assert "@mycompany.io" in result


# ---------------------------------------------------------------------------
# mask_phone
# ---------------------------------------------------------------------------

class TestMaskPhone:
    def test_international_format(self):
        result = PIIMasker.mask_phone("+14155552671")
        assert "***" in result
        # Last 2 digits preserved
        assert result.endswith("71")

    def test_short_phone_fallback(self):
        result = PIIMasker.mask_phone("123")
        assert result == "***-***-****"

    def test_empty_phone(self):
        result = PIIMasker.mask_phone("")
        assert result == "***-***-****"

    def test_digits_only(self):
        result = PIIMasker.mask_phone("9876543210")
        assert "***" in result
        assert result.endswith("10")


# ---------------------------------------------------------------------------
# mask_name
# ---------------------------------------------------------------------------

class TestMaskName:
    def test_two_word_name(self):
        result = PIIMasker.mask_name("Johnathan Doe")
        assert result == "J. D."

    def test_single_name(self):
        result = PIIMasker.mask_name("Madonna")
        assert result == "M."

    def test_three_word_name(self):
        result = PIIMasker.mask_name("Mary Jane Watson")
        # Initials of first and last name
        assert result.startswith("M.")
        assert result.endswith("W.")

    def test_empty_name(self):
        result = PIIMasker.mask_name("")
        assert result == "Anonymous Candidate"


# ---------------------------------------------------------------------------
# mask_profile (role-based)
# ---------------------------------------------------------------------------

class TestMaskProfile:
    def _make_profile(self) -> dict:
        return {
            "candidate_id": "cand-001",
            "full_name": {"value": "Alice Smith"},
            "emails": [{"value": "alice.smith@corp.com", "source": "resume"}],
            "phones": [{"value": "+14155552671", "source": "ats"}],
            "skills": [{"name": "Python"}],
        }

    def test_admin_sees_full_profile(self):
        profile = self._make_profile()
        result = PIIMasker.mask_profile(profile, user_role="admin")
        assert result["emails"][0]["value"] == "alice.smith@corp.com"
        assert result["phones"][0]["value"] == "+14155552671"

    def test_reviewer_sees_full_profile(self):
        profile = self._make_profile()
        result = PIIMasker.mask_profile(profile, user_role="reviewer")
        assert result["emails"][0]["value"] == "alice.smith@corp.com"
        assert result["phones"][0]["value"] == "+14155552671"

    def test_recruiter_email_masked(self):
        profile = self._make_profile()
        result = PIIMasker.mask_profile(profile, user_role="recruiter")
        masked_email = result["emails"][0]["value"]
        assert "***" in masked_email
        assert "@corp.com" in masked_email

    def test_recruiter_phone_masked(self):
        profile = self._make_profile()
        result = PIIMasker.mask_profile(profile, user_role="recruiter")
        masked_phone = result["phones"][0]["value"]
        assert "***" in masked_phone

    def test_non_sensitive_fields_unchanged(self):
        """Skills and candidate_id must not be altered by masking."""
        profile = self._make_profile()
        result = PIIMasker.mask_profile(profile, user_role="recruiter")
        assert result["candidate_id"] == "cand-001"
        assert result["skills"] == [{"name": "Python"}]

    def test_plain_string_emails_masked(self):
        """Profiles with plain string email lists are also masked."""
        profile = {
            "emails": ["bob.jones@example.com"],
            "phones": ["+441234567890"],
        }
        result = PIIMasker.mask_profile(profile, user_role="recruiter")
        assert "***" in result["emails"][0]
        assert "***" in result["phones"][0]

    def test_missing_emails_key_is_safe(self):
        """Profile with no emails/phones key is not broken by masking."""
        profile = {"candidate_id": "cand-002", "full_name": {"value": "Bob"}}
        result = PIIMasker.mask_profile(profile, user_role="recruiter")
        assert result["candidate_id"] == "cand-002"

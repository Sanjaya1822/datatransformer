"""
Tests for Auth (FastAPI endpoints and auth service).

Covers:
- Password hashing & verification with direct bcrypt
- JWT token creation and decoding
- POST /auth/login (JSON & OAuth2 form formats)
- GET /auth/me
- POST /auth/register
- Single Recruiter role authentication and authorization
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from jose import jwt, JWTError

from data_transformer.api.app import app
from data_transformer.auth.service import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from data_transformer.db.database import Base, get_db
from data_transformer.db.models import User, UserRole
from data_transformer.db.seeder import seed_database


# ---------------------------------------------------------------------------
# In-memory DB fixture for API tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    db = TestingSessionLocal()
    seed_database(db, hash_password_func=hash_password)

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    db.close()
    Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# Password hashing & JWT unit tests
# ---------------------------------------------------------------------------

class TestAuthService:
    def test_hash_and_verify_password(self):
        plain = "SecurePass123!"
        hashed = hash_password(plain)
        assert hashed != plain
        assert verify_password(plain, hashed) is True
        assert verify_password("WrongPass", hashed) is False

    def test_create_and_decode_token(self):
        token = create_access_token({"sub": "recruiter", "role": "recruiter"})
        payload = decode_access_token(token)
        assert payload is not None
        assert payload.get("sub") == "recruiter"
        assert payload.get("role") == "recruiter"

    def test_decode_invalid_token(self):
        with pytest.raises(JWTError):
            decode_access_token("invalid.token.here")


# ---------------------------------------------------------------------------
# Auth API Endpoints
# ---------------------------------------------------------------------------

class TestAuthAPI:
    def test_login_json_success(self, client_with_db):
        resp = client_with_db.post("/auth/login", json={"username": "recruiter", "password": "recruiter123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["role"] == "recruiter"

    def test_login_form_success(self, client_with_db):
        resp = client_with_db.post(
            "/auth/login",
            data={"username": "recruiter", "password": "recruiter123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "recruiter"

    def test_login_invalid_password(self, client_with_db):
        resp = client_with_db.post("/auth/login", json={"username": "recruiter", "password": "wrongpassword"})
        assert resp.status_code == 401

    def test_get_current_user_me(self, client_with_db):
        login_resp = client_with_db.post("/auth/login", json={"username": "recruiter", "password": "recruiter123"})
        token = login_resp.json()["access_token"]

        me_resp = client_with_db.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp.status_code == 200
        user = me_resp.json()
        assert user["username"] == "recruiter"
        assert user["role"] == "recruiter"

    def test_recruiter_can_register_user(self, client_with_db):
        login_resp = client_with_db.post("/auth/login", json={"username": "recruiter", "password": "recruiter123"})
        token = login_resp.json()["access_token"]

        reg_resp = client_with_db.post(
            "/auth/register",
            json={
                "username": "new_recruiter",
                "email": "new_recruiter@example.com",
                "password": "Password123!",
                "role": "recruiter",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert reg_resp.status_code == 201
        assert reg_resp.json()["username"] == "new_recruiter"

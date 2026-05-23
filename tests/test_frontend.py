import pytest
from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app, follow_redirects=False)


def test_login_page_loads():
    resp = client.get("/app/login")
    assert resp.status_code == 200
    assert b"Login" in resp.content


def test_protected_route_redirects_to_login():
    resp = client.get("/app/overview")
    assert resp.status_code in (302, 303)
    assert "/app/login" in resp.headers["location"]


def test_login_wrong_password_shows_error():
    resp = client.post("/app/login", data={"password": "wrongpassword"})
    assert resp.status_code == 200
    assert b"Invalid password" in resp.content


def test_login_correct_password_redirects_and_sets_cookie(monkeypatch):
    monkeypatch.setattr("api.auth.DASHBOARD_PASSWORD", "testpass")
    resp = client.post("/app/login", data={"password": "testpass"})
    assert resp.status_code in (302, 303)
    assert "session_token" in resp.cookies


def test_logout_clears_cookie():
    resp = client.get("/app/logout")
    assert resp.status_code in (302, 303)
    # Cookie cleared (empty value or max-age=0)
    cookie_header = resp.headers.get("set-cookie", "")
    assert "session_token" in cookie_header

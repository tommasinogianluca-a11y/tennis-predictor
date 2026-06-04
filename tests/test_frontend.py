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


def _authed_client():
    """Returns a TestClient with a valid session cookie."""
    import hashlib
    from config import API_SECRET_KEY, DASHBOARD_PASSWORD
    token = hashlib.sha256(f"{DASHBOARD_PASSWORD}:{API_SECRET_KEY}".encode()).hexdigest()
    c = TestClient(app, follow_redirects=False)
    c.cookies.set("session_token", token)
    return c


def test_overview_page_loads():
    c = _authed_client()
    resp = c.get("/app/overview")
    assert resp.status_code == 200
    assert b"Overview" in resp.content
    assert b"Scommettitore" in resp.content
    assert b"Value Bets" in resp.content


def test_value_bets_page_loads():
    c = _authed_client()
    resp = c.get("/app/value-bets")
    assert resp.status_code == 200
    assert b"Value Bets" in resp.content


def test_predict_page_loads():
    c = _authed_client()
    resp = c.get("/app/predict")
    assert resp.status_code == 200
    assert b"Predict" in resp.content
    assert b"player1_id" in resp.content


def test_players_page_loads():
    c = _authed_client()
    resp = c.get("/app/players")
    assert resp.status_code == 200
    assert b"Players" in resp.content


def test_players_search_empty():
    c = _authed_client()
    resp = c.get("/app/players/search?q=")
    assert resp.status_code == 200


def test_players_search_with_query():
    c = _authed_client()
    resp = c.get("/app/players/search?q=sin")
    assert resp.status_code == 200


def test_system_page_loads():
    c = _authed_client()
    resp = c.get("/app/system")
    assert resp.status_code == 200
    assert b"System" in resp.content
    assert b"Quick Actions" in resp.content


def test_system_run_unknown_action_returns_400():
    c = _authed_client()
    resp = c.post("/app/system/run/invalid_action")
    assert resp.status_code == 400


def test_system_run_action_returns_job_fragment():
    c = _authed_client()
    resp = c.post("/app/system/run/refresh_odds")
    assert resp.status_code == 200
    assert b"job-" in resp.content


def test_system_job_status_not_found():
    c = _authed_client()
    resp = c.get("/app/system/job/nonexistent")
    assert resp.status_code == 200
    assert b"non trovato" in resp.content.lower()


def test_system_stats_fragment():
    c = _authed_client()
    resp = c.get("/app/system/stats")
    assert resp.status_code == 200
    assert b"matches" in resp.content

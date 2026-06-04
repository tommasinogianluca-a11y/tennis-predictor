import os
import pytest

os.environ.setdefault("API_SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:///test_tennis.db")

from fastapi.testclient import TestClient
from data.db import create_tables


@pytest.fixture(scope="module")
def client():
    create_tables()
    from api.server import app
    return TestClient(app)


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "uptime" in data


def test_report_today_missing_auth(client):
    resp = client.get("/report/today")
    assert resp.status_code == 422


def test_report_today_wrong_key(client):
    resp = client.get("/report/today", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_report_today_valid_key(client):
    resp = client.get("/report/today", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 200


def test_value_bets_requires_auth(client):
    resp = client.get("/value-bets")
    assert resp.status_code == 422


def test_matches_upcoming_requires_auth(client):
    resp = client.get("/matches/upcoming")
    assert resp.status_code == 422


def test_predict_endpoint_not_found(client):
    resp = client.post(
        "/predict",
        headers={"X-API-Key": "test-secret"},
        json={"player1_id": 9999, "player2_id": 9998, "surface": "hard", "tournament_category": "250"},
    )
    assert resp.status_code == 404

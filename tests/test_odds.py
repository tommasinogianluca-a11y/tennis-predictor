"""Tests for data/odds.py — value bet detection logic."""
import pytest
from unittest.mock import MagicMock, patch


# ── _normalize_odds ───────────────────────────────────────────────────────────

from data.odds import _normalize_odds, kelly_stake, EDGE_THRESHOLD


def test_normalize_odds_sums_to_one():
    result = _normalize_odds([2.0, 2.0])
    assert abs(sum(result) - 1.0) < 1e-9


def test_normalize_odds_favourite_higher_prob():
    # Lower odds = higher implied probability
    fav, dog = _normalize_odds([1.5, 3.0])
    assert fav > dog


def test_normalize_odds_removes_margin():
    # Raw implied: 1/1.5 + 1/3.0 = 0.667 + 0.333 = 1.0 → no margin
    # With margin: 1/1.4 + 1/2.8 ≈ 1.071 → normalized removes it
    result = _normalize_odds([1.4, 2.8])
    assert abs(sum(result) - 1.0) < 1e-9
    assert result[0] > 0.6   # favourite still > 60%


def test_normalize_odds_invalid_returns_empty():
    assert _normalize_odds([]) == []
    assert _normalize_odds([0.5, 0.5]) == []   # odds <= 1.0 invalid


def test_normalize_odds_single_valid():
    # One valid, one invalid — only one prob
    result = _normalize_odds([2.0, 0.5])
    assert len(result) == 1


# ── kelly_stake ───────────────────────────────────────────────────────────────

def test_kelly_stake_positive_edge():
    stake = kelly_stake(edge=0.10, p_model=0.60)
    assert stake > 0


def test_kelly_stake_zero_edge():
    stake = kelly_stake(edge=0.0, p_model=0.55)
    assert stake == 0.0


def test_kelly_stake_negative_edge():
    stake = kelly_stake(edge=-0.05, p_model=0.45)
    assert stake == 0.0


def test_kelly_stake_certainty_guard():
    # p_model >= 1.0 → returns 0 to avoid division by zero
    assert kelly_stake(edge=0.5, p_model=1.0) == 0.0


def test_kelly_stake_fraction_25pct():
    # Full Kelly = edge*p/(1-p); fractional = × 0.25
    edge, p = 0.10, 0.60
    full = (edge * p) / (1.0 - p)
    assert abs(kelly_stake(edge, p) - full * 0.25) < 1e-9


# ── EDGE_THRESHOLD ────────────────────────────────────────────────────────────

def test_edge_threshold_value():
    assert EDGE_THRESHOLD == 0.05


# ── detect_value_bets (integration-style with mocks) ─────────────────────────

def _make_db_mock(players, predictions_deleted=0):
    db = MagicMock()
    # .query().filter().delete() → return deleted count
    db.query.return_value.filter.return_value.delete.return_value = predictions_deleted
    # .query().filter().all() for OddsSnapshot pruning
    db.query.return_value.filter.return_value.all.return_value = []
    return db


def test_detect_value_bets_no_odds_returns_empty():
    from data.odds import detect_value_bets
    db = MagicMock()
    with patch("data.odds.scrape_upcoming_odds", return_value=[]):
        result = detect_value_bets(db, model_fn=MagicMock())
    assert result == []


def test_detect_value_bets_finds_value():
    from data.odds import detect_value_bets
    from data.db import Player, Prediction

    # Fake match: p1 odds=3.0, p2 odds=1.4 → bookie gives p1 ~30%
    # Model says p1 = 45% → edge = 45% - 30% = +15% > 5% threshold
    fake_odds = [{
        "player1": "Fake Player A",
        "player2": "Fake Player B",
        "odds_p1": 3.0,
        "odds_p2": 1.4,
        "bookmaker_count": 10,
        "surface": "clay",
        "category": "250",
        "match_date": None,
    }]

    p1 = MagicMock(spec=Player); p1.id = 1
    p2 = MagicMock(spec=Player); p2.id = 2

    model_fn = MagicMock(return_value={"p1_win_prob": 0.45, "p2_win_prob": 0.55})

    db = MagicMock()
    # Delete stale predictions
    db.query.return_value.filter.return_value.delete.return_value = 0

    with patch("data.odds.scrape_upcoming_odds", return_value=fake_odds), \
         patch("data.odds._find_player", side_effect=[p1, p2]):
        result = detect_value_bets(db, model_fn=model_fn)

    assert len(result) == 1
    assert result[0]["edge_pct"] > 5


def test_detect_value_bets_no_value_below_threshold():
    from data.odds import detect_value_bets
    from data.db import Player

    # Both players near 50% → no edge
    fake_odds = [{
        "player1": "A", "player2": "B",
        "odds_p1": 2.0, "odds_p2": 2.0,
        "bookmaker_count": 5,
        "surface": "hard", "category": "250",
        "match_date": None,
    }]

    p1 = MagicMock(spec=Player); p1.id = 1
    p2 = MagicMock(spec=Player); p2.id = 2
    model_fn = MagicMock(return_value={"p1_win_prob": 0.51, "p2_win_prob": 0.49})

    db = MagicMock()
    db.query.return_value.filter.return_value.delete.return_value = 0

    with patch("data.odds.scrape_upcoming_odds", return_value=fake_odds), \
         patch("data.odds._find_player", side_effect=[p1, p2]):
        result = detect_value_bets(db, model_fn=model_fn)

    assert result == []

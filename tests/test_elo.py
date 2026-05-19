from datetime import datetime, timedelta
import pytest


def test_expected_score_equal_ratings():
    from models.elo import expected_score
    assert abs(expected_score(1500, 1500) - 0.5) < 0.001


def test_expected_score_higher_wins():
    from models.elo import expected_score
    p = expected_score(1600, 1400)
    assert p > 0.7


def test_expected_score_formula():
    from models.elo import expected_score
    import math
    expected = 1.0 / (1.0 + 10 ** ((1400 - 1600) / 400))
    assert abs(expected_score(1600, 1400) - expected) < 0.0001


def test_k_factor_top50():
    from models.elo import get_k_factor
    assert get_k_factor(1) == 32.0
    assert get_k_factor(50) == 32.0


def test_k_factor_outside_top50():
    from models.elo import get_k_factor
    assert get_k_factor(51) == 40.0
    assert get_k_factor(None) == 40.0


def test_decay_weight_recent():
    from models.elo import decay_weight
    recent = datetime.utcnow() - timedelta(days=100)
    assert decay_weight(recent) == 1.0


def test_decay_weight_old():
    from models.elo import decay_weight
    old = datetime.utcnow() - timedelta(days=800)
    assert decay_weight(old) == 0.5


def test_update_ratings_winner_gains():
    from models.elo import update_ratings
    ra, rb = update_ratings(1500.0, 1500.0, winner=1, ka=32.0, kb=32.0)
    assert ra > 1500.0
    assert rb < 1500.0


def test_update_ratings_zero_sum():
    from models.elo import update_ratings
    ra, rb = update_ratings(1500.0, 1500.0, winner=1, ka=32.0, kb=32.0)
    assert abs((ra - 1500.0) + (rb - 1500.0)) < 0.001


def test_update_ratings_upset():
    from models.elo import update_ratings
    ra_low, rb_high = update_ratings(1400.0, 1600.0, winner=1, ka=40.0, kb=32.0)
    assert ra_low > 1400.0 + 20


def test_decay_weight_boundary_exactly_730_days():
    from models.elo import decay_weight
    # Exactly 730 days = NOT > DECAY_THRESHOLD_DAYS, should return 1.0
    exactly_730 = datetime.utcnow() - timedelta(days=730)
    assert decay_weight(exactly_730) == 1.0


def test_decay_weight_with_explicit_now():
    from models.elo import decay_weight
    fixed_now = datetime(2024, 1, 1)
    old = datetime(2021, 1, 1)  # 3 years before fixed_now
    assert decay_weight(old, now=fixed_now) == 0.5
    recent = datetime(2023, 12, 1)  # 31 days before fixed_now
    assert decay_weight(recent, now=fixed_now) == 1.0

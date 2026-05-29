import pytest
from datetime import date


def test_tournament_prestige_slam():
    from models.features import tournament_prestige
    assert tournament_prestige("Slam") == 4


def test_tournament_prestige_masters():
    from models.features import tournament_prestige
    assert tournament_prestige("Masters") == 3


def test_tournament_prestige_500():
    from models.features import tournament_prestige
    assert tournament_prestige("500") == 2


def test_tournament_prestige_250():
    from models.features import tournament_prestige
    assert tournament_prestige("250") == 1


def test_tournament_prestige_unknown():
    from models.features import tournament_prestige
    assert tournament_prestige("unknown") == 1


def test_exponential_weights_decay():
    from models.features import exponential_weights
    weights = exponential_weights(5)
    assert len(weights) == 5
    assert weights[-1] > weights[0]  # most recent has highest weight
    assert abs(sum(weights) - 1.0) < 0.001


def test_recent_form_all_wins():
    from models.features import recent_form
    results = [1, 1, 1, 1, 1]
    assert recent_form(results) > 0.9


def test_recent_form_all_losses():
    from models.features import recent_form
    results = [0, 0, 0, 0, 0]
    assert recent_form(results) < 0.1


_FEATURE_PATCHES = {
    "models.features._get_elo": 1500.0,
    "models.features._get_surface_winrate": 0.6,
    "models.features._get_h2h": (5, 3),
    "models.features._get_days_rest": 3,
    "models.features._get_rounds_played": 2,
    "models.features._get_serve_rating": 1.0,
    "models.features._get_return_rating": 1.0,
    "models.features._get_recent_results": [1, 1, 0, 1, 0],
    "models.features._get_fatigue_index": 3,
    "models.features._get_sentiment": 0.0,
    # v2 additions
    "models.features._get_ranking_raw": 50.0,
    "models.features._get_age_score": -1.5,
    "models.features._get_surface_form": 0.65,
    "models.features._get_second_serve_rate": 0.55,
    "models.features._get_aggression_index": 0.52,
    "models.features._get_big_match_winrate": 0.6,
}


def _all_patches():
    from unittest.mock import patch
    from contextlib import ExitStack
    stack = ExitStack()
    for target, val in _FEATURE_PATCHES.items():
        stack.enter_context(patch(target, return_value=val))
    return stack


def test_feature_vector_length():
    from models.features import build_feature_vector
    from unittest.mock import MagicMock

    mock_db = MagicMock()
    with _all_patches():
        vec = build_feature_vector(1, 2, "hard", "Slam", date.today(), mock_db)
    assert len(vec) == 18


def test_feature_vector_all_floats():
    from models.features import build_feature_vector
    from unittest.mock import MagicMock

    mock_db = MagicMock()
    with _all_patches():
        vec = build_feature_vector(1, 2, "hard", "Slam", date.today(), mock_db)
    for v in vec:
        assert isinstance(v, float)

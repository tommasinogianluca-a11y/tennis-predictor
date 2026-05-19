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


def test_feature_vector_length():
    from models.features import build_feature_vector
    from unittest.mock import MagicMock, patch

    mock_db = MagicMock()

    with patch("models.features._get_elo", return_value=1500.0), \
         patch("models.features._get_surface_winrate", return_value=0.6), \
         patch("models.features._get_h2h", return_value=(5, 3)), \
         patch("models.features._get_days_rest", return_value=3), \
         patch("models.features._get_rounds_played", return_value=2), \
         patch("models.features._get_serve_rating", return_value=1.0), \
         patch("models.features._get_return_rating", return_value=1.0), \
         patch("models.features._get_recent_results", return_value=[1, 1, 0, 1, 0]), \
         patch("models.features._get_fatigue_index", return_value=3), \
         patch("models.features._get_sentiment", return_value=0.0):

        vec = build_feature_vector(1, 2, "hard", "Slam", date.today(), mock_db)
        assert len(vec) == 12


def test_feature_vector_all_floats():
    from models.features import build_feature_vector
    from unittest.mock import MagicMock, patch

    mock_db = MagicMock()

    with patch("models.features._get_elo", return_value=1500.0), \
         patch("models.features._get_surface_winrate", return_value=0.6), \
         patch("models.features._get_h2h", return_value=(5, 3)), \
         patch("models.features._get_days_rest", return_value=3), \
         patch("models.features._get_rounds_played", return_value=2), \
         patch("models.features._get_serve_rating", return_value=1.0), \
         patch("models.features._get_return_rating", return_value=1.0), \
         patch("models.features._get_recent_results", return_value=[1, 1, 0, 1, 0]), \
         patch("models.features._get_fatigue_index", return_value=3), \
         patch("models.features._get_sentiment", return_value=0.0):

        vec = build_feature_vector(1, 2, "hard", "Slam", date.today(), mock_db)
        for v in vec:
            assert isinstance(v, float)

import numpy as np
from datetime import date, datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session


PRESTIGE_MAP = {"Slam": 4, "Masters": 3, "500": 2, "250": 1, "Finals": 2}
TWO_YEARS = timedelta(days=730)


def tournament_prestige(category: str) -> int:
    return PRESTIGE_MAP.get(category, 1)


def exponential_weights(n: int, alpha: float = 0.85) -> list:
    weights = [alpha ** (n - 1 - i) for i in range(n)]
    total = sum(weights)
    return [w / total for w in weights]


def recent_form(results: list) -> float:
    if not results:
        return 0.5
    weights = exponential_weights(len(results))
    return float(sum(w * r for w, r in zip(weights, results)))


def _get_elo(player_id: int, surface: str, db: Session) -> float:
    from data.db import EloRating
    record = db.query(EloRating).filter_by(player_id=player_id, surface=surface).first()
    return record.rating if record else 1500.0


def _get_surface_winrate(player_id: int, surface: str, as_of: date, db: Session) -> float:
    from data.db import Match
    cutoff = as_of - TWO_YEARS
    matches = (
        db.query(Match)
        .filter(
            Match.surface == surface,
            Match.date >= cutoff,
            Match.date < as_of,
            (Match.player1_id == player_id) | (Match.player2_id == player_id),
        )
        .all()
    )
    if not matches:
        return 0.5
    wins = sum(1 for m in matches if m.winner_id == player_id)
    return wins / len(matches)


def _get_h2h(p1_id: int, p2_id: int, surface: Optional[str], db: Session) -> tuple:
    from data.db import Match
    query = db.query(Match).filter(
        ((Match.player1_id == p1_id) & (Match.player2_id == p2_id)) |
        ((Match.player1_id == p2_id) & (Match.player2_id == p1_id))
    )
    if surface:
        query = query.filter(Match.surface == surface)
    matches = query.all()
    if not matches:
        return 0, 0
    total = len(matches)
    p1_wins = sum(1 for m in matches if m.winner_id == p1_id)
    return total, p1_wins


def _get_days_rest(player_id: int, as_of: date, db: Session) -> int:
    from data.db import Match
    last = (
        db.query(Match)
        .filter(
            (Match.player1_id == player_id) | (Match.player2_id == player_id),
            Match.date < as_of,
        )
        .order_by(Match.date.desc())
        .first()
    )
    if not last or not last.date:
        return 30
    return (as_of - last.date).days


def _get_rounds_played(player_id: int, tournament_name: str, as_of: date, db: Session) -> int:
    from data.db import Match
    return (
        db.query(Match)
        .filter(
            (Match.player1_id == player_id) | (Match.player2_id == player_id),
            Match.tournament_name == tournament_name,
            Match.date <= as_of,
        )
        .count()
    )


def _get_serve_rating(player_id: int, as_of: date, db: Session) -> float:
    from data.db import Match, MatchStats
    stats = (
        db.query(MatchStats)
        .join(Match, MatchStats.match_id == Match.id)
        .filter(
            MatchStats.player_id == player_id,
            Match.date < as_of,
        )
        .order_by(Match.date.desc())
        .limit(20)
        .all()
    )
    if not stats:
        return 1.0
    valid = [s for s in stats if s.first_serve_pct and s.first_serve_won_pct]
    if not valid:
        return 1.0
    return float(np.mean([
        (s.first_serve_pct / 100) * (s.first_serve_won_pct / 100) for s in valid
    ]))


def _get_return_rating(player_id: int, as_of: date, db: Session) -> float:
    from data.db import Match, MatchStats
    stats = (
        db.query(MatchStats)
        .join(Match, MatchStats.match_id == Match.id)
        .filter(
            MatchStats.player_id == player_id,
            Match.date < as_of,
        )
        .order_by(Match.date.desc())
        .limit(20)
        .all()
    )
    if not stats:
        return 1.0
    valid = [s for s in stats if s.bp_faced and s.bp_saved is not None]
    if not valid:
        return 1.0
    rates = [(s.bp_faced - s.bp_saved) / s.bp_faced for s in valid if s.bp_faced > 0]
    return float(np.mean(rates)) if rates else 1.0


def _get_recent_results(player_id: int, as_of: date, db: Session, n: int = 10) -> list:
    from data.db import Match
    matches = (
        db.query(Match)
        .filter(
            (Match.player1_id == player_id) | (Match.player2_id == player_id),
            Match.date < as_of,
            Match.winner_id.isnot(None),
        )
        .order_by(Match.date.desc())
        .limit(n)
        .all()
    )
    results = [1 if m.winner_id == player_id else 0 for m in reversed(matches)]
    return results


def _get_fatigue_index(player_id: int, as_of: date, db: Session) -> int:
    from data.db import Match
    cutoff = as_of - timedelta(days=30)
    return (
        db.query(Match)
        .filter(
            (Match.player1_id == player_id) | (Match.player2_id == player_id),
            Match.date >= cutoff,
            Match.date < as_of,
        )
        .count()
    )


def _get_sentiment(player_id: int, db: Session) -> float:
    from data.db import SentimentCache
    record = db.query(SentimentCache).filter_by(player_id=player_id).first()
    return record.sentiment_score if record else 0.0


def build_feature_vector(
    player1_id: int,
    player2_id: int,
    surface: str,
    tournament_category: str,
    as_of: date,
    db: Session,
    tournament_name: str = "",
) -> list:
    elo_p1 = _get_elo(player1_id, surface, db)
    elo_p2 = _get_elo(player2_id, surface, db)

    wr_p1 = _get_surface_winrate(player1_id, surface, as_of, db)
    wr_p2 = _get_surface_winrate(player2_id, surface, as_of, db)

    h2h_total, h2h_p1_wins = _get_h2h(player1_id, player2_id, None, db)
    h2h_rate = h2h_p1_wins / h2h_total if h2h_total else 0.5

    h2h_surf_total, h2h_surf_p1 = _get_h2h(player1_id, player2_id, surface, db)
    h2h_surf_rate = h2h_surf_p1 / h2h_surf_total if h2h_surf_total else 0.5

    rest_p1 = _get_days_rest(player1_id, as_of, db)
    rest_p2 = _get_days_rest(player2_id, as_of, db)

    rounds_p1 = _get_rounds_played(player1_id, tournament_name, as_of, db)
    rounds_p2 = _get_rounds_played(player2_id, tournament_name, as_of, db)

    serve_p1 = _get_serve_rating(player1_id, as_of, db)
    serve_p2 = _get_serve_rating(player2_id, as_of, db)

    ret_p1 = _get_return_rating(player1_id, as_of, db)
    ret_p2 = _get_return_rating(player2_id, as_of, db)

    form_p1 = recent_form(_get_recent_results(player1_id, as_of, db))
    form_p2 = recent_form(_get_recent_results(player2_id, as_of, db))

    fatigue_p1 = _get_fatigue_index(player1_id, as_of, db)
    fatigue_p2 = _get_fatigue_index(player2_id, as_of, db)

    sent_p1 = _get_sentiment(player1_id, db)
    sent_p2 = _get_sentiment(player2_id, db)

    return [
        float(elo_p1 - elo_p2),
        float(wr_p1 - wr_p2),
        float(h2h_rate),
        float(h2h_surf_rate),
        float(rest_p1 - rest_p2),
        float(rounds_p1 - rounds_p2),
        float(tournament_prestige(tournament_category)),
        float(serve_p1 - serve_p2),
        float(ret_p1 - ret_p2),
        float(form_p1 - form_p2),
        float(fatigue_p1 - fatigue_p2),
        float(sent_p1 - sent_p2),
    ]

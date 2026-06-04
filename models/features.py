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


def _get_h2h(p1_id: int, p2_id: int, surface: Optional[str], as_of: date, db: Session) -> float:
    """Recency-weighted H2H win rate for p1. Matches decay 10 % per year."""
    from data.db import Match
    query = db.query(Match).filter(
        ((Match.player1_id == p1_id) & (Match.player2_id == p2_id)) |
        ((Match.player1_id == p2_id) & (Match.player2_id == p1_id)),
        Match.date.isnot(None),
        Match.date < as_of,
    )
    if surface:
        query = query.filter(Match.surface == surface)
    matches = query.all()
    if not matches:
        return 0.5
    weighted_wins = total_w = 0.0
    for m in matches:
        days = (as_of - m.date).days
        w = 0.9 ** (days / 365.25)   # 10 % per-year decay
        total_w += w
        if m.winner_id == p1_id:
            weighted_wins += w
    return weighted_wins / total_w if total_w > 0 else 0.5


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
    # SentimentCache stores one current snapshot per player (no history).
    # No date filter needed — the cache is always the most recent analysis.
    from data.db import SentimentCache
    record = db.query(SentimentCache).filter_by(player_id=player_id).first()
    return record.sentiment_score if record else 0.0


# ── New features (v2) ─────────────────────────────────────────────────────────

_BIG_CATEGORIES = {"Slam", "Masters"}


def _get_ranking_raw(player_id: int, db: Session) -> float:
    """ATP ranking number. Missing / unranked → 500 (penalty for unknown)."""
    from data.db import Player
    p = db.query(Player).filter_by(id=player_id).first()
    rank = p.current_ranking if p and p.current_ranking else 500
    return float(rank)


def _get_age_score(player_id: int, as_of: date, db: Session) -> float:
    """Proximity to tennis prime (27 years). Returns -|age - 27|; higher = closer to prime."""
    from data.db import Player
    p = db.query(Player).filter_by(id=player_id).first()
    if not p or not p.dob:
        return -2.0  # neutral: 2 years off prime
    age = (as_of - p.dob).days / 365.25
    return -abs(age - 27.0)


def _get_surface_form(player_id: int, surface: str, as_of: date, db: Session, n: int = 10) -> float:
    """Exponentially-weighted recent form on a specific surface."""
    from data.db import Match
    matches = (
        db.query(Match)
        .filter(
            (Match.player1_id == player_id) | (Match.player2_id == player_id),
            Match.surface == surface,
            Match.date < as_of,
            Match.winner_id.isnot(None),
        )
        .order_by(Match.date.desc())
        .limit(n)
        .all()
    )
    results = [1 if m.winner_id == player_id else 0 for m in reversed(matches)]
    return recent_form(results)


def _get_second_serve_rate(player_id: int, as_of: date, db: Session) -> float:
    """Average second-serve won pct over last 20 matches with stats."""
    from data.db import Match, MatchStats
    stats = (
        db.query(MatchStats)
        .join(Match, MatchStats.match_id == Match.id)
        .filter(
            MatchStats.player_id == player_id,
            Match.date < as_of,
            MatchStats.second_serve_won_pct.isnot(None),
        )
        .order_by(Match.date.desc())
        .limit(20)
        .all()
    )
    if not stats:
        return 0.5
    return float(np.mean([s.second_serve_won_pct / 100 for s in stats]))


def _get_aggression_index(player_id: int, as_of: date, db: Session) -> float:
    """Winners / (winners + unforced_errors). Higher = more aggressive and clean."""
    from data.db import Match, MatchStats
    stats = (
        db.query(MatchStats)
        .join(Match, MatchStats.match_id == Match.id)
        .filter(
            MatchStats.player_id == player_id,
            Match.date < as_of,
            MatchStats.winners.isnot(None),
            MatchStats.unforced_errors.isnot(None),
        )
        .order_by(Match.date.desc())
        .limit(20)
        .all()
    )
    if not stats:
        return 0.5
    ratios = []
    for s in stats:
        total = (s.winners or 0) + (s.unforced_errors or 0)
        if total > 0:
            ratios.append(s.winners / total)
    return float(np.mean(ratios)) if ratios else 0.5


def _get_big_match_winrate(player_id: int, as_of: date, db: Session) -> float:
    """Win rate in Slams + Masters over last 2 years. Separates grinders from chokers."""
    from data.db import Match
    cutoff = as_of - TWO_YEARS
    matches = (
        db.query(Match)
        .filter(
            (Match.player1_id == player_id) | (Match.player2_id == player_id),
            Match.tournament_category.in_(_BIG_CATEGORIES),
            Match.date >= cutoff,
            Match.date < as_of,
            Match.winner_id.isnot(None),
        )
        .all()
    )
    if not matches:
        return 0.5
    wins = sum(1 for m in matches if m.winner_id == player_id)
    return wins / len(matches)


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

    h2h_rate = _get_h2h(player1_id, player2_id, None, as_of, db)
    h2h_surf_rate = _get_h2h(player1_id, player2_id, surface, as_of, db)

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

    # ── v2 features ──────────────────────────────────────────────────────────
    rank_p1 = _get_ranking_raw(player1_id, db)
    rank_p2 = _get_ranking_raw(player2_id, db)

    age_p1 = _get_age_score(player1_id, as_of, db)
    age_p2 = _get_age_score(player2_id, as_of, db)

    surf_form_p1 = _get_surface_form(player1_id, surface, as_of, db)
    surf_form_p2 = _get_surface_form(player2_id, surface, as_of, db)

    second_p1 = _get_second_serve_rate(player1_id, as_of, db)
    second_p2 = _get_second_serve_rate(player2_id, as_of, db)

    aggr_p1 = _get_aggression_index(player1_id, as_of, db)
    aggr_p2 = _get_aggression_index(player2_id, as_of, db)

    big_p1 = _get_big_match_winrate(player1_id, as_of, db)
    big_p2 = _get_big_match_winrate(player2_id, as_of, db)

    return [
        # ── v1 (12 features) ─────────────────────────────────────────────────
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
        # ── v2 (6 new features) ──────────────────────────────────────────────
        float(rank_p2 - rank_p1),          # positive = p1 better ranked
        float(age_p1 - age_p2),            # positive = p1 closer to prime age
        float(surf_form_p1 - surf_form_p2),
        float(second_p1 - second_p2),
        float(aggr_p1 - aggr_p2),
        float(big_p1 - big_p2),
    ]

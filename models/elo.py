from datetime import datetime
from typing import Optional

INITIAL_RATING = 1500.0
DECAY_THRESHOLD_DAYS = 730  # 2 years


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def get_k_factor(ranking: Optional[int]) -> float:
    if ranking is not None and ranking <= 50:
        return 32.0
    return 40.0


def decay_weight(match_date: datetime, now: Optional[datetime] = None) -> float:
    if now is None:
        now = datetime.utcnow()
    days_ago = (now - match_date).days
    return 0.5 if days_ago > DECAY_THRESHOLD_DAYS else 1.0


def update_ratings(
    ra: float, rb: float, winner: int, ka: float = 40.0, kb: float = 40.0
) -> tuple:
    ea = expected_score(ra, rb)
    sa = 1.0 if winner == 1 else 0.0
    new_ra = ra + ka * (sa - ea)
    new_rb = rb + kb * ((1.0 - sa) - (1.0 - ea))
    return new_ra, new_rb


def backfill_elo(db) -> None:
    """Compute and store Elo ratings for all historical matches ordered by date.

    Note: K-factor uses each player's CURRENT ranking, not their ranking at match time.
    Historical ranking data is not available in the dataset. This is a known limitation
    that introduces slight bias in ratings for players whose ranking has changed significantly.
    """
    from datetime import timezone
    from data.db import EloRating, Match, Player

    # Pre-load all players to avoid N+1 queries
    players = {p.id: p for p in db.query(Player).all()}
    ratings: dict = {}

    matches = (
        db.query(Match)
        .filter(Match.date.isnot(None))
        .order_by(Match.date)
        .all()
    )

    for match in matches:
        surface = match.surface or "hard"
        p1_key = (match.player1_id, surface)
        p2_key = (match.player2_id, surface)
        ra = ratings.get(p1_key, INITIAL_RATING)
        rb = ratings.get(p2_key, INITIAL_RATING)

        match_dt = datetime.combine(match.date, datetime.min.time())
        weight = decay_weight(match_dt)

        p1 = players.get(match.player1_id)
        p2 = players.get(match.player2_id)
        ka = get_k_factor(p1.current_ranking if p1 else None) * weight
        kb = get_k_factor(p2.current_ranking if p2 else None) * weight

        winner = 1 if match.winner_id == match.player1_id else 2
        new_ra, new_rb = update_ratings(ra, rb, winner, ka, kb)
        ratings[p1_key] = new_ra
        ratings[p2_key] = new_rb

    now = datetime.now(timezone.utc)
    for (player_id, surface), rating in ratings.items():
        record = (
            db.query(EloRating)
            .filter_by(player_id=player_id, surface=surface)
            .first()
        )
        if record:
            record.rating = rating
            record.updated_at = now
        else:
            db.add(EloRating(player_id=player_id, surface=surface, rating=rating))

    db.commit()

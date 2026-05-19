import csv
import io
import logging
from datetime import datetime, date
from typing import Optional

import requests
from sqlalchemy.orm import Session

from data.db import Match, MatchStats, Player

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
SEED_YEARS = list(range(1990, 2026))

SURFACE_MAP = {
    "Hard": "hard", "Clay": "clay",
    "Grass": "grass", "Carpet": "indoor", "": "hard",
}
LEVEL_MAP = {
    "G": "Slam", "M": "Masters", "A": "500",
    "D": "250", "F": "Finals", "C": "250", "S": "250",
}


def _parse_date(raw: str) -> Optional[date]:
    try:
        return datetime.strptime(raw.strip(), "%Y%m%d").date()
    except (ValueError, AttributeError):
        return None


def _safe_int(val: str) -> Optional[int]:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _upsert_player(db: Session, atp_id: str, name: str, nationality: str) -> Player:
    if atp_id:
        player = db.query(Player).filter_by(atp_id=atp_id).first()
        if player:
            return player
    player = db.query(Player).filter_by(name=name).first()
    if not player:
        player = Player(name=name, nationality=nationality, atp_id=atp_id or None)
        db.add(player)
        db.flush()
    elif atp_id and not player.atp_id:
        player.atp_id = atp_id
    return player


def _insert_stats(db: Session, match_id: int, player_id: int, row: dict, prefix: str) -> None:
    svpt = _safe_int(row.get(f"{prefix}_svpt", ""))
    first_in = _safe_int(row.get(f"{prefix}_1stIn", ""))
    first_won = _safe_int(row.get(f"{prefix}_1stWon", ""))
    second_won = _safe_int(row.get(f"{prefix}_2ndWon", ""))
    bp_faced = _safe_int(row.get(f"{prefix}_bpFaced", ""))
    bp_saved = _safe_int(row.get(f"{prefix}_bpSaved", ""))
    aces = _safe_int(row.get(f"{prefix}_ace", ""))
    df = _safe_int(row.get(f"{prefix}_df", ""))

    first_serve_pct = (
        (first_in / svpt * 100)
        if (svpt is not None and svpt > 0 and first_in is not None)
        else None
    )
    first_won_pct = (
        (first_won / first_in * 100)
        if (first_in is not None and first_in > 0 and first_won is not None)
        else None
    )
    second_denom = (svpt - first_in) if (svpt is not None and first_in is not None) else None
    second_won_pct = (
        (second_won / second_denom * 100)
        if (second_denom is not None and second_denom > 0 and second_won is not None)
        else None
    )

    stats = MatchStats(
        match_id=match_id, player_id=player_id,
        aces=aces, double_faults=df,
        first_serve_pct=first_serve_pct,
        first_serve_won_pct=first_won_pct,
        second_serve_won_pct=second_won_pct,
        bp_faced=bp_faced, bp_saved=bp_saved,
    )
    db.add(stats)


def seed_year(db: Session, year: int) -> int:
    url = BASE_URL.format(year=year)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return 0

    reader = csv.DictReader(io.StringIO(resp.text))
    count = 0
    try:
        for row in reader:
            winner = _upsert_player(
                db, row.get("winner_id", ""), row.get("winner_name", ""),
                row.get("winner_ioc", "")
            )
            loser = _upsert_player(
                db, row.get("loser_id", ""), row.get("loser_name", ""),
                row.get("loser_ioc", "")
            )
            match_date = _parse_date(row.get("tourney_date", ""))
            surface = SURFACE_MAP.get(row.get("surface", ""), "hard")
            category = LEVEL_MAP.get(row.get("tourney_level", ""), "250")

            match = Match(
                tournament_name=row.get("tourney_name", ""),
                tournament_category=category,
                surface=surface,
                round=row.get("round", ""),
                date=match_date,
                player1_id=winner.id,
                player2_id=loser.id,
                winner_id=winner.id,
                score_string=row.get("score", ""),
            )
            db.add(match)
            db.flush()

            _insert_stats(db, match.id, winner.id, row, "w")
            _insert_stats(db, match.id, loser.id, row, "l")
            count += 1

        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning("Rolled back year %d due to error: %s", year, e)
        return 0

    return count


def run_seed(db: Session) -> None:
    existing = db.query(Match).count()
    if existing > 0:
        logger.info("DB already seeded (%d matches), skipping.", existing)
        return

    logger.info("Starting Sackmann seed for years %d-%d", SEED_YEARS[0], SEED_YEARS[-1])
    total = 0
    for year in SEED_YEARS:
        n = seed_year(db, year)
        logger.info("  %d: %d matches", year, n)
        total += n
    logger.info("Seed complete. Total matches: %d", total)

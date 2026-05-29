import logging
import unicodedata
from datetime import datetime, timezone
from typing import Optional

import requests
from sqlalchemy.orm import Session

from config import ODDS_API_KEY
from data.db import OddsSnapshot, Player, Prediction

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
KELLY_FRACTION = 0.25
EDGE_THRESHOLD = 0.05

# Map Odds API sport key fragments → (surface, category)
_SPORT_KEY_META = {
    "french_open":    ("clay",  "Slam"),
    "roland_garros":  ("clay",  "Slam"),
    "wimbledon":      ("grass", "Slam"),
    "us_open":        ("hard",  "Slam"),
    "australian":     ("hard",  "Slam"),
    "miami":          ("hard",  "Masters"),
    "indian_wells":   ("hard",  "Masters"),
    "madrid":         ("clay",  "Masters"),
    "rome":           ("clay",  "Masters"),
    "montreal":       ("hard",  "Masters"),
    "cincinnati":     ("hard",  "Masters"),
    "shanghai":       ("hard",  "Masters"),
    "paris":          ("indoor","Masters"),
    "toronto":        ("hard",  "Masters"),
    "canada":         ("hard",  "Masters"),
}


def _surface_category_for_key(sport_key: str) -> tuple[str, str]:
    """Derive surface and tournament category from Odds API sport key."""
    key_lower = sport_key.lower()
    for fragment, (surface, category) in _SPORT_KEY_META.items():
        if fragment in key_lower:
            return surface, category
    return "hard", "250"  # safe default


def _active_atp_sport_keys() -> list[str]:
    """Return all currently active ATP tennis sport keys from the Odds API."""
    if not ODDS_API_KEY:
        return []
    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/",
            params={"apiKey": ODDS_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        return [
            s["key"] for s in resp.json()
            if s.get("group") == "Tennis"
            and "atp" in s["key"].lower()
            and s.get("active", False)
        ]
    except requests.RequestException as e:
        logger.error("Odds API sports list failed: %s", e)
        return []


def _normalize_odds(odds_list: list) -> list:
    """Remove bookmaker margin — return fair implied probabilities."""
    probs = [1.0 / o for o in odds_list if o > 1.0]
    if not probs:
        return []
    total = sum(probs)
    return [p / total for p in probs]


def kelly_stake(edge: float, p_model: float) -> float:
    """Fractional Kelly criterion (25%)."""
    if p_model >= 1.0:
        return 0.0
    return max(0.0, ((edge * p_model) / (1.0 - p_model)) * KELLY_FRACTION)


def _fetch_odds_for_sport(sport_key: str) -> list:
    """Fetch h2h odds for a single sport key. Returns list of match dicts."""
    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds/",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Odds API [%s] request failed: %s", sport_key, e)
        return []

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    logger.info("Odds API [%s]: used=%s remaining=%s", sport_key, used, remaining)

    surface, category = _surface_category_for_key(sport_key)
    results = []
    for match in resp.json():
        try:
            home = match["home_team"]
            away = match["away_team"]
            home_odds_list, away_odds_list = [], []
            bookmaker_names: set = set()
            for bookie in match.get("bookmakers", []):
                for market in bookie.get("markets", []):
                    if market["key"] != "h2h":
                        continue
                    has_home = any(o["name"] == home for o in market["outcomes"])
                    has_away = any(o["name"] == away for o in market["outcomes"])
                    if has_home and has_away:
                        bookmaker_names.add(bookie.get("key", bookie.get("title", "")))
                    for outcome in market["outcomes"]:
                        if outcome["name"] == home:
                            home_odds_list.append(outcome["price"])
                        elif outcome["name"] == away:
                            away_odds_list.append(outcome["price"])
            if not home_odds_list or not away_odds_list:
                continue
            commence_time = match.get("commence_time")
            match_dt = None
            if commence_time:
                try:
                    match_dt = datetime.fromisoformat(
                        commence_time.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
            results.append({
                "player1": home,
                "player2": away,
                "odds_p1": round(sum(home_odds_list) / len(home_odds_list), 3),
                "odds_p2": round(sum(away_odds_list) / len(away_odds_list), 3),
                "bookmaker_count": len(bookmaker_names),
                "surface": surface,
                "category": category,
                "match_date": match_dt,
            })
        except (KeyError, ZeroDivisionError):
            continue
    return results


def scrape_upcoming_odds(db: Session) -> list:
    """Fetch upcoming ATP match odds from the-odds-api.com."""
    if not ODDS_API_KEY:
        logger.warning("ODDS_API_KEY not set — skipping odds fetch.")
        return []

    sport_keys = _active_atp_sport_keys()
    if not sport_keys:
        logger.warning("No active ATP tennis events on Odds API right now.")
        return []

    logger.info("Active ATP sport keys: %s", sport_keys)
    results = []
    for key in sport_keys:
        results.extend(_fetch_odds_for_sport(key))

    logger.info("Odds API: parsed %d upcoming ATP matches total.", len(results))
    return results


def _normalize(s: str) -> str:
    """Lowercase, remove accents, collapse spaces."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def _name_variants(full_name: str) -> list[str]:
    """
    Generate candidate search strings for a player name.
    API format: "Alex de Minaur", "Thiago Agustin Tirante"
    DB may store: "De Minaur A.", "Minaur A.", "A. De Minaur", etc.
    """
    parts = full_name.strip().split()
    if not parts:
        return []

    first = parts[0]
    last = parts[-1]
    initial = first[0]

    variants = [
        full_name,            # Alex de Minaur
        f"{last} {first}",    # Minaur Alex
        f"{last} {initial}.", # Minaur A.
        f"{initial}. {last}", # A. Minaur
        last,                 # Minaur (last only)
    ]
    # For compound last names (de/van/del/da/dos/von): also try without particle
    particles = {"de", "van", "del", "da", "dos", "von", "le", "la"}
    non_particle = [p for p in parts[1:] if p.lower() not in particles]
    if non_particle and non_particle[-1] != last:
        real_last = non_particle[-1]
        variants += [
            f"{real_last} {initial}.",
            f"{initial}. {real_last}",
            real_last,
        ]
    return variants


def _find_player(db: Session, full_name: str) -> Optional[object]:
    """
    Find player by full name with normalization and multi-format fallback.
    If not found at all, auto-create a minimal record so predictions still work.
    """
    norm_target = _normalize(full_name)

    # Prefer active (ranked) players; fall back to full DB
    for ranked_only in (True, False):
        base = db.query(Player)
        if ranked_only:
            base = base.filter(Player.current_ranking.isnot(None))

        for variant in _name_variants(full_name):
            # Try exact ilike
            p = base.filter(Player.name.ilike(variant)).first()
            if p:
                logger.debug("Matched '%s' → '%s' (variant '%s')", full_name, p.name, variant)
                return p

        # Normalized full-scan: load top 500 ranked players and compare normalized names
        limit = 500 if ranked_only else 3000
        candidates = base.order_by(
            Player.current_ranking.asc().nulls_last()
        ).limit(limit).all()
        for c in candidates:
            if _normalize(c.name) == norm_target:
                logger.debug("Matched '%s' → '%s' (normalized)", full_name, c.name)
                return c
        # Partial normalized last-name match among top candidates
        last_norm = _normalize(full_name.split()[-1])
        first_initial = full_name[0].lower()
        name_matches = [
            c for c in candidates
            if last_norm in _normalize(c.name) and _normalize(c.name).startswith(first_initial)
        ]
        if len(name_matches) == 1:
            logger.debug("Matched '%s' → '%s' (partial norm)", full_name, name_matches[0].name)
            return name_matches[0]

    # Not found anywhere — auto-create so predictions still work (ELO defaults to 1500)
    logger.info("Auto-creating player record for '%s' (not in DB)", full_name)
    new_player = Player(name=full_name, current_ranking=None, nationality=None)
    db.add(new_player)
    db.flush()  # get ID without full commit
    return new_player


def detect_value_bets(db: Session, model_fn) -> list:
    """
    For each upcoming match with odds, run model, compute edge, flag value bets.
    model_fn: callable(player1_id, player2_id, surface, category, db) -> {p1_win_prob, p2_win_prob}
    """
    raw_odds = scrape_upcoming_odds(db)
    if not raw_odds:
        logger.info("No odds data — skipping prediction cleanup.")
        return []

    # Wipe all odds-sourced predictions before rebuilding fresh.
    # bookmaker_odds_p1 IS NOT NULL is the universal marker for API-sourced
    # predictions (manual predict form never sets bookmaker odds).
    deleted = (
        db.query(Prediction)
        .filter(Prediction.bookmaker_odds_p1.isnot(None))
        .delete(synchronize_session=False)
    )
    logger.info("Cleared %d stale odds predictions before refresh.", deleted)

    # Prune snapshots older than 14 days to avoid unbounded growth.
    from datetime import timedelta
    snap_cutoff = datetime.now(timezone.utc) - timedelta(days=14)
    db.query(OddsSnapshot).filter(OddsSnapshot.recorded_at < snap_cutoff).delete(
        synchronize_session=False
    )

    value_bets = []

    for odds_data in raw_odds:
        p1_name = odds_data["player1"]
        p2_name = odds_data["player2"]
        surface = odds_data.get("surface", "hard")
        category = odds_data.get("category", "250")

        p1 = _find_player(db, p1_name)
        p2 = _find_player(db, p2_name)
        if not p1 or not p2:
            logger.warning("Could not resolve players: %s vs %s", p1_name, p2_name)
            continue

        try:
            pred = model_fn(p1.id, p2.id, surface, category, db)
        except Exception as e:
            logger.warning("Prediction failed for %s vs %s: %s", p1_name, p2_name, e)
            continue

        p_model_p1 = pred["p1_win_prob"]
        p_model_p2 = pred["p2_win_prob"]

        normalized = _normalize_odds([odds_data["odds_p1"], odds_data["odds_p2"]])
        if len(normalized) < 2:
            continue
        p_bookie_p1, p_bookie_p2 = normalized

        edge_p1 = p_model_p1 - p_bookie_p1
        edge_p2 = p_model_p2 - p_bookie_p2

        value_bet_player: Optional[int] = None
        edge = 0.0
        if edge_p1 > EDGE_THRESHOLD:
            value_bet_player = 1
            edge = edge_p1
        elif edge_p2 > EDGE_THRESHOLD:
            value_bet_player = 2
            edge = edge_p2

        prediction = Prediction(
            player1_id=p1.id,
            player2_id=p2.id,
            surface=surface,
            tournament_category=category,
            match_date=odds_data.get("match_date"),
            p1_win_probability=p_model_p1,
            p2_win_probability=p_model_p2,
            value_bet_player=value_bet_player,
            edge_percentage=edge * 100 if edge else None,
            bookmaker_odds_p1=odds_data["odds_p1"],
            bookmaker_odds_p2=odds_data["odds_p2"],
            bookmaker_count=odds_data.get("bookmaker_count"),
        )
        db.add(prediction)

        # Record odds snapshot for line-movement tracking.
        # Normalise player order so (a_id < b_id) regardless of API ordering.
        pa_id = min(p1.id, p2.id)
        pb_id = max(p1.id, p2.id)
        if pa_id == p1.id:
            odds_a, odds_b = odds_data["odds_p1"], odds_data["odds_p2"]
        else:
            odds_a, odds_b = odds_data["odds_p2"], odds_data["odds_p1"]
        match_d = odds_data["match_date"].date() if odds_data.get("match_date") else None
        db.add(OddsSnapshot(
            player_a_id=pa_id,
            player_b_id=pb_id,
            match_date=match_d,
            odds_a=odds_a,
            odds_b=odds_b,
        ))

        if value_bet_player:
            stake = kelly_stake(edge, p_model_p1 if value_bet_player == 1 else p_model_p2)
            value_bets.append({
                "player1": p1_name,
                "player2": p2_name,
                "value_on": p1_name if value_bet_player == 1 else p2_name,
                "edge_pct": round(edge * 100, 2),
                "kelly_stake_fraction": round(stake, 4),
                "model_prob": round(p_model_p1 if value_bet_player == 1 else p_model_p2, 3),
                "bookie_odds": odds_data[f"odds_p{value_bet_player}"],
            })

    db.commit()
    logger.info("Value bets found: %d", len(value_bets))
    return value_bets

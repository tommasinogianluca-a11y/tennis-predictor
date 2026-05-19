import logging
import re
from typing import Optional

from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from data.db import Player, Prediction

logger = logging.getLogger(__name__)
ODDSPORTAL_BASE = "https://www.oddsportal.com/tennis/atp/"
KELLY_FRACTION = 0.25
EDGE_THRESHOLD = 0.05


def _normalize_odds(odds_list: list) -> list:
    """Remove bookmaker margin — return fair implied probabilities."""
    probs = [1.0 / o for o in odds_list if o > 1.0]
    if not probs:
        return []
    total = sum(probs)
    return [p / total for p in probs]


def _parse_decimal_odds(text: str) -> Optional[float]:
    try:
        val = float(re.sub(r"[^\d.]", "", text))
        return val if val > 1.0 else None
    except (ValueError, TypeError):
        return None


def kelly_stake(edge: float, p_model: float) -> float:
    """Fractional Kelly criterion (25%)."""
    if p_model >= 1.0:
        return 0.0
    return max(0.0, ((edge * p_model) / (1.0 - p_model)) * KELLY_FRACTION)


def scrape_upcoming_odds(db: Session) -> list:
    """Scrape upcoming ATP match odds from oddsportal.com."""
    from data.scraper import fetch_url
    url = ODDSPORTAL_BASE
    html = fetch_url(url, db)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    results = []

    try:
        for row in soup.select("tr.deactivate, tr[class*='odd'], tr[class*='deactivate']"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            try:
                participants = row.select(".name")
                if len(participants) < 2:
                    continue
                p1_name = participants[0].get_text(strip=True)
                p2_name = participants[1].get_text(strip=True)

                odds_cells = row.select(".odds-nowrp, td.right")
                odds_values = []
                for cell in odds_cells[:2]:
                    o = _parse_decimal_odds(cell.get_text(strip=True))
                    if o:
                        odds_values.append(o)

                if len(odds_values) < 2:
                    continue

                results.append({
                    "player1": p1_name,
                    "player2": p2_name,
                    "odds_p1": odds_values[0],
                    "odds_p2": odds_values[1],
                })
            except Exception:
                continue
    except Exception as e:
        logger.warning("oddsportal parse error: %s", e)

    logger.info("oddsportal: found %d upcoming matches with odds.", len(results))
    return results


def detect_value_bets(db: Session, model_fn) -> list:
    """
    For each upcoming match with odds, run model, compute edge, flag value bets.
    model_fn: callable(player1_id, player2_id, surface, category, db) -> {p1_win_prob, p2_win_prob}
    """
    raw_odds = scrape_upcoming_odds(db)
    value_bets = []

    for odds_data in raw_odds:
        p1_name = odds_data["player1"]
        p2_name = odds_data["player2"]

        p1 = db.query(Player).filter(Player.name.ilike(f"%{p1_name.split()[-1]}%")).first()
        p2 = db.query(Player).filter(Player.name.ilike(f"%{p2_name.split()[-1]}%")).first()
        if not p1 or not p2:
            continue

        try:
            pred = model_fn(p1.id, p2.id, "hard", "250", db)
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

        value_bet_player = None
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
            surface="hard",
            tournament_category="250",
            p1_win_probability=p_model_p1,
            p2_win_probability=p_model_p2,
            value_bet_player=value_bet_player,
            edge_percentage=edge * 100 if edge else None,
            bookmaker_odds_p1=odds_data["odds_p1"],
            bookmaker_odds_p2=odds_data["odds_p2"],
        )
        db.add(prediction)

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

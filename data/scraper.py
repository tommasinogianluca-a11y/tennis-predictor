import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from data.db import RawCache

logger = logging.getLogger(__name__)
CACHE_TTL_HOURS = 24
USER_AGENT = "TennisPredictor/1.0 (research)"
HEADERS = {"User-Agent": USER_AGENT}


def _robots_allowed(url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        return True
    return rp.can_fetch(USER_AGENT, url)


def _cached_get(url: str, db: Session) -> Optional[str]:
    from datetime import timezone
    cutoff = datetime.now(timezone.utc) - timedelta(hours=CACHE_TTL_HOURS)
    record = db.query(RawCache).filter(
        RawCache.url == url, RawCache.cached_at >= cutoff
    ).first()
    return record.response_body if record else None


def _cache_set(url: str, body: str, db: Session) -> None:
    from datetime import timezone
    existing = db.query(RawCache).filter_by(url=url).first()
    if existing:
        existing.response_body = body
        existing.cached_at = datetime.now(timezone.utc)
    else:
        db.add(RawCache(url=url, response_body=body))
    db.commit()


def fetch_url(url: str, db: Session, retries: int = 3) -> Optional[str]:
    cached = _cached_get(url, db)
    if cached:
        return cached

    if not _robots_allowed(url):
        logger.warning("robots.txt disallows %s", url)
        return None

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            _cache_set(url, resp.text, db)
            return resp.text
        except requests.RequestException as e:
            wait = 2 ** attempt
            logger.warning("Attempt %d/%d failed for %s: %s — retrying in %ds", attempt + 1, retries, url, e, wait)
            if attempt < retries - 1:
                time.sleep(wait)

    logger.error("All retries exhausted for %s", url)
    return None


def scrape_tennisabstract_player(player_name: str, db: Session) -> dict:
    slug = player_name.replace(" ", "-").lower()
    url = f"https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p={slug}"
    html = fetch_url(url, db)
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    stats = {}
    try:
        tables = soup.find_all("table")
        if tables:
            for row in tables[0].find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).lower()
                    val = cells[1].get_text(strip=True)
                    stats[key] = val
    except Exception as e:
        logger.warning("tennisabstract parse error for %s: %s", player_name, e)

    return stats


def scrape_recent_matches_flashscore(db: Session) -> list:
    url = "https://www.flashscore.com/tennis/atp-singles/"
    html = fetch_url(url, db)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    matches = []

    try:
        for row in soup.select(".event__match"):
            try:
                p1 = row.select_one(".event__participant--home")
                p2 = row.select_one(".event__participant--away")
                score_el = row.select_one(".event__score")
                date_el = row.select_one(".event__time")
                if not (p1 and p2):
                    continue
                matches.append({
                    "player1": p1.get_text(strip=True),
                    "player2": p2.get_text(strip=True),
                    "score": score_el.get_text(strip=True) if score_el else "",
                    "date_raw": date_el.get_text(strip=True) if date_el else "",
                })
            except Exception:
                continue
    except Exception as e:
        logger.warning("flashscore parse error: %s", e)

    return matches


def run_scraper(db: Session) -> None:
    logger.info("Running scraper update...")
    raw = scrape_recent_matches_flashscore(db)
    logger.info("Flashscore returned %d raw match records.", len(raw))

import logging
from datetime import datetime
from typing import Optional

import feedparser
from sqlalchemy.orm import Session

from data.db import News, Player

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    "https://www.tennis.com/feed/",
    "https://www.espn.com/espn/rss/tennis/news",
    "https://www.atptour.com/en/media/rss-feed/xml-feed",
]


def _fuzzy_match_player(name: str, db: Session) -> Optional[Player]:
    name_lower = name.lower()
    players = db.query(Player).all()
    for p in players:
        parts = p.name.lower().split()
        if any(
            (part in name_lower or name_lower in p.name.lower())
            for part in parts
            if len(part) > 3
        ):
            return p
    return None


def fetch_news(db: Session) -> int:
    inserted = 0
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            logger.warning("Failed to parse feed %s: %s", feed_url, e)
            continue

        for entry in feed.entries:
            headline = getattr(entry, "title", "")
            content = getattr(entry, "summary", "")
            source_url = getattr(entry, "link", "")
            published_raw = getattr(entry, "published_parsed", None)
            published_at = datetime(*published_raw[:6]) if published_raw else datetime.utcnow()

            if source_url and db.query(News).filter_by(source_url=source_url).first():
                continue

            player = _fuzzy_match_player(headline, db)
            db.add(News(
                player_id=player.id if player else None,
                headline=headline,
                content=content,
                source_url=source_url,
                published_at=published_at,
            ))
            inserted += 1

        db.commit()

    logger.info("News fetcher inserted %d new articles.", inserted)
    return inserted

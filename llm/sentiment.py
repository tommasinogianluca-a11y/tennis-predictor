import json
import logging
from datetime import datetime, timedelta

import anthropic
from sqlalchemy.orm import Session

from config import ANTHROPIC_API_KEY
from data.db import News, Player, SentimentCache

logger = logging.getLogger(__name__)
CACHE_TTL_HOURS = 6
MODEL = "claude-sonnet-4-20250514"
MAX_ARTICLES = 5

PROMPT_TEMPLATE = """You are a tennis analyst. Based on these recent news headlines and articles about {player_name}, assess their current match readiness and confidence.
Consider: injuries, recent results, coaching changes, personal issues, fatigue.
Return ONLY a JSON object:
{{"sentiment_score": float, "reasoning": string, "flags": [list]}}
sentiment_score between -1.0 and +1.0.
News:
{articles}"""


def _is_cache_valid(record: SentimentCache) -> bool:
    cutoff = datetime.utcnow() - timedelta(hours=CACHE_TTL_HOURS)
    return record.cached_at >= cutoff


def analyze_player(player_id: int, db: Session) -> dict:
    cached = db.query(SentimentCache).filter_by(player_id=player_id).first()
    if cached and _is_cache_valid(cached):
        return {
            "sentiment_score": cached.sentiment_score,
            "reasoning": cached.reasoning,
            "flags": json.loads(cached.flags or "[]"),
        }

    player = db.query(Player).filter_by(id=player_id).first()
    if not player:
        return {"sentiment_score": 0.0, "reasoning": "", "flags": []}

    articles = (
        db.query(News)
        .filter_by(player_id=player_id)
        .order_by(News.published_at.desc())
        .limit(MAX_ARTICLES)
        .all()
    )
    if not articles:
        return {"sentiment_score": 0.0, "reasoning": "No recent news.", "flags": []}

    articles_text = "\n".join(
        f"- {a.headline}: {(a.content or '')[:300]}" for a in articles
    )

    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — skipping sentiment for player %d.", player_id)
        return {"sentiment_score": 0.0, "reasoning": "API key not configured.", "flags": []}

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    player_name=player.name,
                    articles=articles_text,
                ),
            }],
        )
        raw = message.content[0].text.strip()
        result = json.loads(raw)
    except (json.JSONDecodeError, anthropic.APIError, IndexError, Exception) as e:
        logger.warning("Sentiment API error for player %d: %s", player_id, e)
        result = {"sentiment_score": 0.0, "reasoning": str(e), "flags": []}

    score = float(result.get("sentiment_score", 0.0))
    score = max(-1.0, min(1.0, score))  # clamp to [-1, 1]

    if cached:
        cached.sentiment_score = score
        cached.reasoning = result.get("reasoning", "")
        cached.flags = json.dumps(result.get("flags", []))
        cached.cached_at = datetime.utcnow()
    else:
        db.add(SentimentCache(
            player_id=player_id,
            sentiment_score=score,
            reasoning=result.get("reasoning", ""),
            flags=json.dumps(result.get("flags", [])),
        ))
    db.commit()
    return result


def run_sentiment_update(db: Session) -> None:
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(hours=CACHE_TTL_HOURS)
    player_ids = (
        db.query(News.player_id)
        .filter(
            News.player_id.isnot(None),
            News.published_at >= cutoff,
        )
        .distinct()
        .all()
    )
    ids = [row[0] for row in player_ids]
    logger.info("Running sentiment update for %d players.", len(ids))
    for pid in ids:
        analyze_player(pid, db)

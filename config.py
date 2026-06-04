import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///tennis.db")

if DATABASE_URL == "sqlite:///tennis.db":
    logger.warning("DATABASE_URL not set — using local SQLite (not suitable for production).")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    logger.warning("ANTHROPIC_API_KEY not set — sentiment analysis will be disabled.")

API_SECRET_KEY = os.getenv("API_SECRET_KEY")
if not API_SECRET_KEY:
    raise RuntimeError("API_SECRET_KEY environment variable is required. Set it in .env or Railway variables.")

PORT = int(os.getenv("PORT", 8000))

DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
if not DASHBOARD_PASSWORD:
    logger.warning("DASHBOARD_PASSWORD not set — dashboard login disabled (any password accepted).")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
if not ODDS_API_KEY:
    logger.warning("ODDS_API_KEY not set — odds refresh will be skipped.")

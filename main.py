import logging

import uvicorn
from rich.logging import RichHandler

from config import PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_path=False)],
)
logger = logging.getLogger(__name__)


def run_migrations():
    from alembic import command
    from alembic.config import Config
    alembic_cfg = Config("alembic.ini")
    try:
        command.upgrade(alembic_cfg, "head")
        logger.info("Alembic migrations applied.")
    except Exception as e:
        logger.warning("Alembic migration warning: %s (continuing)", e)


def init_db():
    from data.db import create_tables
    create_tables()
    logger.info("DB tables verified.")


def seed_if_empty():
    from data.db import Match, SessionLocal
    from data.seeder import run_seed
    db = SessionLocal()
    try:
        if db.query(Match).count() == 0:
            logger.info("Empty DB — running Sackmann seed (may take a few minutes).")
            run_seed(db)
    finally:
        db.close()


def backfill_elo_if_empty():
    from data.db import EloRating, SessionLocal
    from models.elo import backfill_elo
    db = SessionLocal()
    try:
        if db.query(EloRating).count() == 0:
            logger.info("No Elo ratings — running backfill.")
            backfill_elo(db)
            logger.info("Elo backfill complete.")
    finally:
        db.close()


def load_or_train_model():
    from data.db import SessionLocal
    from models.predictor import get_model
    db = SessionLocal()
    try:
        get_model(db)
    finally:
        db.close()


def main():
    logger.info("Tennis Predictor starting up...")
    run_migrations()
    init_db()
    seed_if_empty()
    backfill_elo_if_empty()
    load_or_train_model()

    from scheduler import start_scheduler
    start_scheduler()

    from api.server import app
    logger.info("Starting FastAPI on port %d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()

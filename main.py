import logging
import sys
import threading
import traceback

import uvicorn
from rich.logging import RichHandler

from config import PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_path=False)],
)
logger = logging.getLogger(__name__)

_init_done = False
_init_error = None
_init_step = "not_started"


def _log(msg):
    """Print-based log that flushes immediately — works reliably in background threads."""
    print(f"[INIT] {msg}", flush=True)


def run_migrations():
    from alembic import command
    from alembic.config import Config
    alembic_cfg = Config("alembic.ini")
    try:
        command.upgrade(alembic_cfg, "head")
        _log("Alembic migrations applied.")
    except Exception as e:
        _log(f"Alembic migration warning: {e} (continuing)")


def init_db():
    from data.db import create_tables
    create_tables()
    _log("DB tables verified.")


def seed_if_empty():
    from data.db import Match, SessionLocal
    from data.seeder import run_seed
    db = SessionLocal()
    try:
        count = db.query(Match).count()
        if count == 0:
            _log("Empty DB — running Sackmann seed (may take 10-20 min).")
            run_seed(db)
        else:
            _log(f"DB already has {count} matches, skipping seed.")
    finally:
        db.close()


def backfill_elo_if_empty():
    from data.db import EloRating, SessionLocal
    from models.elo import backfill_elo
    db = SessionLocal()
    try:
        if db.query(EloRating).count() == 0:
            _log("No Elo ratings — running backfill.")
            backfill_elo(db)
            _log("Elo backfill complete.")
        else:
            _log("ELO ratings already present, skipping.")
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


def _background_init():
    global _init_done, _init_error, _init_step
    try:
        _init_step = "migrations"
        _log("Step 1/5: running migrations...")
        run_migrations()

        _init_step = "create_tables"
        _log("Step 2/5: creating tables...")
        init_db()

        _init_step = "seed"
        _log("Step 3/5: seeding DB...")
        seed_if_empty()

        _init_step = "elo_backfill"
        _log("Step 4/5: ELO backfill...")
        backfill_elo_if_empty()

        _init_step = "model"
        _log("Step 5/5: loading/training model...")
        load_or_train_model()

        _init_step = "scheduler"
        _log("Starting scheduler...")
        from scheduler import start_scheduler
        start_scheduler()

        _init_done = True
        _init_step = "done"
        _log("Background init complete!")

    except Exception as e:
        _init_error = traceback.format_exc()
        _init_step = f"FAILED at {_init_step}"
        print(f"[INIT ERROR] Step={_init_step}\n{_init_error}", flush=True)


def main():
    _log("Tennis Predictor starting up...")

    # Kick off heavy init in background — server responds to healthcheck immediately
    t = threading.Thread(target=_background_init, daemon=True, name="bg-init")
    t.start()

    from api.server import app

    # Expose init status on health endpoint via app state
    app.state.get_init_status = lambda: (_init_done, _init_error, _init_step)

    _log(f"Starting FastAPI on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()

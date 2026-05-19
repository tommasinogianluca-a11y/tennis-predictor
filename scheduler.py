import logging

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from config import DATABASE_URL

logger = logging.getLogger(__name__)

jobstores = {"default": SQLAlchemyJobStore(url=DATABASE_URL)}
scheduler = BackgroundScheduler(jobstores=jobstores, timezone="UTC")


def _daily_scrape():
    from data.db import SessionLocal
    from data.scraper import run_scraper
    from models.elo import backfill_elo
    db = SessionLocal()
    try:
        run_scraper(db)
        backfill_elo(db)
    finally:
        db.close()


def _daily_report():
    from data.db import SessionLocal
    from reports.daily_report import generate_report
    db = SessionLocal()
    try:
        r = generate_report(db)
        logger.info("Daily report: %d predictions, %d value bets.", r["total_predictions"], r["value_bets_count"])
    finally:
        db.close()


def _news_and_sentiment():
    from data.db import SessionLocal
    from data.news_fetcher import fetch_news
    from llm.sentiment import run_sentiment_update
    db = SessionLocal()
    try:
        fetch_news(db)
        run_sentiment_update(db)
    finally:
        db.close()


def _refresh_odds():
    from data.db import SessionLocal
    from data.odds import detect_value_bets
    from models.predictor import predict
    db = SessionLocal()
    try:
        bets = detect_value_bets(db, predict)
        logger.info("Odds refresh: %d value bets.", len(bets))
    finally:
        db.close()


def _weekly_retrain():
    import models.predictor as pred_module
    from data.db import SessionLocal
    from models.predictor import train_model
    db = SessionLocal()
    try:
        new_model = train_model(db)
        with pred_module._model_lock:
            pred_module._cached_model = new_model
        logger.info("Weekly retrain complete.")
    finally:
        db.close()


def start_scheduler():
    scheduler.add_job(_daily_scrape, "cron", hour=6, minute=0, id="daily_scrape", replace_existing=True)
    scheduler.add_job(_daily_report, "cron", hour=7, minute=0, id="daily_report", replace_existing=True)
    scheduler.add_job(_news_and_sentiment, "interval", hours=4, id="news_sentiment", replace_existing=True)
    scheduler.add_job(_refresh_odds, "interval", hours=6, id="refresh_odds", replace_existing=True)
    scheduler.add_job(_weekly_retrain, "cron", day_of_week="sun", hour=2, minute=0, id="weekly_retrain", replace_existing=True)
    scheduler.start()
    logger.info("Scheduler started with %d jobs.", len(scheduler.get_jobs()))

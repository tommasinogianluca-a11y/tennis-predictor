from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Text, ForeignKey, Date, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime, timezone
from typing import Generator

from config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    nationality = Column(String(10))
    dob = Column(Date, nullable=True)
    atp_id = Column(String(50), nullable=True, unique=True)
    current_ranking = Column(Integer, nullable=True)


class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True)
    tournament_name = Column(String(200))
    tournament_category = Column(String(50))  # Slam/Masters/500/250/Finals
    surface = Column(String(20))              # hard/clay/grass/indoor
    round = Column(String(50))
    date = Column(Date)
    player1_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    player2_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    winner_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    score_string = Column(String(200), nullable=True)


class MatchStats(Base):
    __tablename__ = "match_stats"
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    aces = Column(Integer, nullable=True)
    double_faults = Column(Integer, nullable=True)
    first_serve_pct = Column(Float, nullable=True)
    first_serve_won_pct = Column(Float, nullable=True)
    second_serve_won_pct = Column(Float, nullable=True)
    bp_faced = Column(Integer, nullable=True)
    bp_saved = Column(Integer, nullable=True)
    winners = Column(Integer, nullable=True)
    unforced_errors = Column(Integer, nullable=True)
    net_points_won = Column(Integer, nullable=True)


class News(Base):
    __tablename__ = "news"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    headline = Column(String(500))
    content = Column(Text, nullable=True)
    source_url = Column(String(500), nullable=True)
    published_at = Column(DateTime, nullable=True)
    sentiment_score = Column(Float, nullable=True)


class EloRating(Base):
    __tablename__ = "elo_ratings"
    __table_args__ = (UniqueConstraint("player_id", "surface", name="uq_elo_player_surface"),)
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    surface = Column(String(20), nullable=False)
    rating = Column(Float, default=1500.0)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=True)
    player1_id = Column(Integer, ForeignKey("players.id"))
    player2_id = Column(Integer, ForeignKey("players.id"))
    surface = Column(String(20))
    tournament_category = Column(String(50))
    p1_win_probability = Column(Float)
    p2_win_probability = Column(Float)
    value_bet_player = Column(Integer, nullable=True)  # 1 or 2
    edge_percentage = Column(Float, nullable=True)
    bookmaker_odds_p1 = Column(Float, nullable=True)
    bookmaker_odds_p2 = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelStore(Base):
    __tablename__ = "model_store"
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), unique=True)
    model_data = Column(Text)       # base64-encoded pickle
    trained_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RawCache(Base):
    __tablename__ = "raw_cache"
    id = Column(Integer, primary_key=True)
    url = Column(String(1000), unique=True)
    response_body = Column(Text)
    cached_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SentimentCache(Base):
    __tablename__ = "sentiment_cache"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), unique=True)
    sentiment_score = Column(Float)
    reasoning = Column(Text, nullable=True)
    flags = Column(Text, nullable=True)   # JSON string
    cached_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Index("ix_match_date", Match.__table__.c.date)
Index("ix_matchstats_match_player", MatchStats.__table__.c.match_id, MatchStats.__table__.c.player_id)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)

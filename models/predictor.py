import base64
import hashlib
import logging
import pickle
import threading
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sqlalchemy.orm import Session

from models.features import build_feature_vector

logger = logging.getLogger(__name__)
MODEL_NAME = "tennis_xgb_v1"
TRAIN_CUTOFF = date(2023, 1, 1)


def _serialize(model) -> str:
    raw = pickle.dumps(model)
    digest = hashlib.sha256(raw).hexdigest()
    payload = digest.encode() + b":" + raw
    return base64.b64encode(payload).decode("utf-8")


def _deserialize(data: str):
    payload = base64.b64decode(data.encode("utf-8"))
    digest_bytes, _, raw = payload.partition(b":")
    expected = hashlib.sha256(raw).hexdigest().encode()
    if digest_bytes != expected:
        raise ValueError("Model data integrity check failed — possible corruption or tampering.")
    return pickle.loads(raw)


def save_model(model, db: Session) -> None:
    from data.db import ModelStore
    data = _serialize(model)
    record = db.query(ModelStore).filter_by(model_name=MODEL_NAME).first()
    if record:
        record.model_data = data
        record.trained_at = datetime.now(timezone.utc)
    else:
        db.add(ModelStore(model_name=MODEL_NAME, model_data=data))
    db.commit()
    logger.info("Model saved to DB.")


def load_model(db: Session) -> Optional[object]:
    from data.db import ModelStore
    record = db.query(ModelStore).filter_by(model_name=MODEL_NAME).first()
    if record:
        logger.info("Model loaded from DB (trained %s).", record.trained_at)
        return _deserialize(record.model_data)
    return None


def _build_memory_cache(db: Session) -> dict:
    """Load all data into memory for fast feature building during training.
    Single batch of DB queries instead of N+1."""
    from collections import defaultdict
    from data.db import EloRating, Match, MatchStats, SentimentCache

    logger.info("Loading all matches into memory...")
    all_matches = (
        db.query(Match)
        .filter(Match.date.isnot(None))
        .order_by(Match.date)
        .all()
    )

    # player_id -> sorted list of (date, match) tuples
    player_matches: dict = defaultdict(list)
    for m in all_matches:
        if m.player1_id:
            player_matches[m.player1_id].append(m)
        if m.player2_id:
            player_matches[m.player2_id].append(m)

    logger.info("Loading ELO ratings into memory...")
    elo_map: dict = {}
    for r in db.query(EloRating).all():
        elo_map[(r.player_id, r.surface)] = r.rating

    logger.info("Loading match stats into memory...")
    stats_map: dict = defaultdict(list)
    for s in db.query(MatchStats).all():
        stats_map[s.player_id].append(s)

    logger.info("Loading sentiment cache...")
    sentiment_map: dict = {}
    for sc in db.query(SentimentCache).all():
        sentiment_map[sc.player_id] = sc.sentiment_score

    return {
        "all_matches": all_matches,
        "player_matches": dict(player_matches),
        "elo_map": elo_map,
        "stats_map": dict(stats_map),
        "sentiment_map": sentiment_map,
    }


def _build_feature_fast(p1_id: int, p2_id: int, surface: str,
                         tournament_category: str, as_of: date,
                         tournament_name: str, cache: dict) -> list:
    """Build feature vector using in-memory cache — no DB queries."""
    from datetime import timedelta
    from models.features import exponential_weights, recent_form, tournament_prestige

    TWO_YEARS = timedelta(days=730)
    ONE_MONTH = timedelta(days=30)
    cutoff_2y = as_of - TWO_YEARS
    cutoff_1m = as_of - ONE_MONTH

    elo_map = cache["elo_map"]
    player_matches = cache["player_matches"]
    stats_map = cache["stats_map"]
    sentiment_map = cache["sentiment_map"]

    def get_elo(pid: int) -> float:
        return elo_map.get((pid, surface), 1500.0)

    def get_player_matches_before(pid: int):
        return [m for m in player_matches.get(pid, [])
                if m.date is not None and m.date < as_of]

    def get_surface_winrate(pid: int) -> float:
        ms = [m for m in get_player_matches_before(pid)
              if m.surface == surface and m.date >= cutoff_2y]
        if not ms:
            return 0.5
        wins = sum(1 for m in ms if m.winner_id == pid)
        return wins / len(ms)

    def get_h2h(with_surface: bool):
        ms = [m for m in player_matches.get(p1_id, [])
              if m.date < as_of
              and ((m.player1_id == p1_id and m.player2_id == p2_id)
                   or (m.player1_id == p2_id and m.player2_id == p1_id))]
        if with_surface:
            ms = [m for m in ms if m.surface == surface]
        if not ms:
            return 0.5
        p1_wins = sum(1 for m in ms if m.winner_id == p1_id)
        return p1_wins / len(ms)

    def get_days_rest(pid: int) -> int:
        ms = get_player_matches_before(pid)
        if not ms:
            return 30
        last = ms[-1]
        return (as_of - last.date).days if last.date else 30

    def get_rounds_played(pid: int) -> int:
        return sum(1 for m in player_matches.get(pid, [])
                   if m.tournament_name == tournament_name and m.date <= as_of)

    def get_serve_rating(pid: int) -> float:
        all_stats = stats_map.get(pid, [])
        valid = [s for s in all_stats
                 if s.first_serve_pct and s.first_serve_won_pct][-20:]
        if not valid:
            return 1.0
        return float(np.mean([(s.first_serve_pct / 100) * (s.first_serve_won_pct / 100)
                               for s in valid]))

    def get_return_rating(pid: int) -> float:
        all_stats = stats_map.get(pid, [])
        valid = [s for s in all_stats
                 if s.bp_faced and s.bp_saved is not None and s.bp_faced > 0][-20:]
        if not valid:
            return 1.0
        return float(np.mean([(s.bp_faced - s.bp_saved) / s.bp_faced for s in valid]))

    def get_recent_form(pid: int) -> float:
        ms = [m for m in get_player_matches_before(pid)
              if m.winner_id is not None][-10:]
        results = [1 if m.winner_id == pid else 0 for m in ms]
        return recent_form(results)

    def get_fatigue(pid: int) -> int:
        return sum(1 for m in get_player_matches_before(pid)
                   if m.date >= cutoff_1m)

    return [
        float(get_elo(p1_id) - get_elo(p2_id)),
        float(get_surface_winrate(p1_id) - get_surface_winrate(p2_id)),
        float(get_h2h(with_surface=False)),
        float(get_h2h(with_surface=True)),
        float(get_days_rest(p1_id) - get_days_rest(p2_id)),
        float(get_rounds_played(p1_id) - get_rounds_played(p2_id)),
        float(tournament_prestige(tournament_category)),
        float(get_serve_rating(p1_id) - get_serve_rating(p2_id)),
        float(get_return_rating(p1_id) - get_return_rating(p2_id)),
        float(get_recent_form(p1_id) - get_recent_form(p2_id)),
        float(get_fatigue(p1_id) - get_fatigue(p2_id)),
        float(sentiment_map.get(p1_id, 0.0) - sentiment_map.get(p2_id, 0.0)),
    ]


def train_model(db: Session) -> object:
    import random as _random
    logger.info("Building training dataset (in-memory fast path)...")
    cache = _build_memory_cache(db)
    all_matches = [m for m in cache["all_matches"]
                   if m.winner_id is not None]

    # Shuffle so StratifiedKFold doesn't see monotone class sequences
    _random.seed(42)
    _random.shuffle(all_matches)

    X_train, y_train, X_test, y_test = [], [], [], []
    skipped = 0
    total = len(all_matches)
    for i, match in enumerate(all_matches):
        if i % 10000 == 0:
            print(f"[TRAIN] Features {i}/{total}...", flush=True)
        try:
            # Seeder always stores winner as player1 → all labels would be 1.
            # Randomly flip 50% of samples to create balanced dataset.
            if _random.random() < 0.5:
                p1_id, p2_id = match.player1_id, match.player2_id
            else:
                p1_id, p2_id = match.player2_id, match.player1_id

            vec = _build_feature_fast(
                p1_id, p2_id,
                match.surface or "hard",
                match.tournament_category or "250",
                match.date, match.tournament_name or "",
                cache,
            )
            label = 1 if match.winner_id == p1_id else 0
            if match.date < TRAIN_CUTOFF:
                X_train.append(vec)
                y_train.append(label)
            else:
                X_test.append(vec)
                y_test.append(label)
        except Exception as exc:
            skipped += 1
            logger.debug("Skipped match %s: %s", getattr(match, 'id', '?'), exc)

    logger.info(
        "Dataset: %d train, %d test, %d skipped.",
        len(X_train), len(X_test), skipped,
    )
    print(f"[TRAIN] Dataset built: {len(X_train)} train, {len(X_test)} test, {skipped} skipped", flush=True)

    if len(X_train) < 100:
        raise RuntimeError("Not enough training data (need >= 100 matches).")

    X_tr = np.array(X_train, dtype=float)
    y_tr = np.array(y_train, dtype=int)

    print("[TRAIN] Fitting XGBoost (CalibratedCV cv=5)...", flush=True)
    base = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        n_jobs=1,  # serial to limit peak RAM on Railway
    )
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)  # cv=3 saves ~40% RAM vs cv=5
    model.fit(X_tr, y_tr)

    if X_test:
        X_te = np.array(X_test, dtype=float)
        y_te = np.array(y_test, dtype=int)
        proba = model.predict_proba(X_te)[:, 1]
        logger.info(
            "Eval — Brier: %.4f  LogLoss: %.4f",
            brier_score_loss(y_te, proba),
            log_loss(y_te, proba),
        )

    save_model(model, db)
    return model


_cached_model = None
_model_lock = threading.Lock()


def get_model(db: Session) -> object:
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    with _model_lock:
        if _cached_model is None:
            _cached_model = load_model(db)
        if _cached_model is None:
            logger.info("No model in DB — training now (first run).")
            _cached_model = train_model(db)
    return _cached_model


def predict(
    player1_id: int,
    player2_id: int,
    surface: str,
    tournament_category: str,
    db: Session,
    tournament_name: str = "",
    as_of: Optional[date] = None,
) -> dict:
    model = get_model(db)
    as_of = as_of or date.today()
    vec = build_feature_vector(
        player1_id, player2_id, surface, tournament_category,
        as_of, db, tournament_name,
    )
    X = np.array([vec], dtype=float)
    proba = model.predict_proba(X)[0]
    return {
        "p1_win_prob": float(proba[1]),
        "p2_win_prob": float(proba[0]),
    }

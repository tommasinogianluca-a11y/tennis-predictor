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

# ── Model versioning ──────────────────────────────────────────────────────────
SURFACES = ["clay", "hard", "grass", "indoor"]
_MODEL_VERSION = "v3"   # v3: surface-specific + temporal weights + H2H decay

TRAIN_CUTOFF = date(2023, 1, 1)
_MIN_SURFACE_SAMPLES = 500   # below this, skip surface model (use global)


def _model_name(surface: str) -> str:
    """DB key for a surface-specific or global model."""
    return f"tennis_xgb_{surface}_{_MODEL_VERSION}"


# ── Serialisation ─────────────────────────────────────────────────────────────

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
        raise ValueError("Model data integrity check failed.")
    return pickle.loads(raw)


def save_model(model, db: Session, model_name: str = "tennis_xgb_global_v3") -> None:
    from data.db import ModelStore
    data = _serialize(model)
    record = db.query(ModelStore).filter_by(model_name=model_name).first()
    if record:
        record.model_data = data
        record.trained_at = datetime.now(timezone.utc)
    else:
        db.add(ModelStore(model_name=model_name, model_data=data))
    db.commit()
    logger.info("Model '%s' saved to DB.", model_name)


def load_model(db: Session, model_name: str = "tennis_xgb_global_v3") -> Optional[object]:
    from data.db import ModelStore
    record = db.query(ModelStore).filter_by(model_name=model_name).first()
    if record:
        logger.info("Model '%s' loaded (trained %s).", model_name, record.trained_at)
        return _deserialize(record.model_data)
    return None


# ── In-memory cache ───────────────────────────────────────────────────────────

def _build_memory_cache(db: Session) -> dict:
    """Load all data into memory for fast feature building during training."""
    from collections import defaultdict
    from data.db import EloRating, Match, MatchStats, Player, SentimentCache

    logger.info("Loading all matches into memory...")
    all_matches = (
        db.query(Match)
        .filter(Match.date.isnot(None))
        .order_by(Match.date)
        .all()
    )

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

    logger.info("Loading sentiment + player data...")
    sentiment_map: dict = {}
    for sc in db.query(SentimentCache).all():
        sentiment_map[sc.player_id] = sc.sentiment_score

    player_map: dict = {p.id: p for p in db.query(Player).all()}

    return {
        "all_matches": all_matches,
        "player_matches": dict(player_matches),
        "elo_map": elo_map,
        "stats_map": dict(stats_map),
        "sentiment_map": sentiment_map,
        "player_map": player_map,
    }


# ── Fast feature path (no DB queries) ────────────────────────────────────────

def _build_feature_fast(p1_id: int, p2_id: int, surface: str,
                         tournament_category: str, as_of: date,
                         tournament_name: str, cache: dict) -> list:
    """18-feature vector using in-memory cache. H2H uses recency decay."""
    from datetime import timedelta
    from models.features import recent_form, tournament_prestige

    _BIG_CATEGORIES = {"Slam", "Masters"}
    TWO_YEARS = timedelta(days=730)
    ONE_MONTH = timedelta(days=30)
    cutoff_2y = as_of - TWO_YEARS
    cutoff_1m = as_of - ONE_MONTH

    elo_map = cache["elo_map"]
    player_matches = cache["player_matches"]
    stats_map = cache["stats_map"]
    sentiment_map = cache["sentiment_map"]
    player_map = cache["player_map"]

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

    def get_h2h(with_surface: bool) -> float:
        """Recency-weighted H2H win rate. 10 % per-year decay."""
        ms = [m for m in player_matches.get(p1_id, [])
              if m.date is not None and m.date < as_of
              and ((m.player1_id == p1_id and m.player2_id == p2_id)
                   or (m.player1_id == p2_id and m.player2_id == p1_id))]
        if with_surface:
            ms = [m for m in ms if m.surface == surface]
        if not ms:
            return 0.5
        weighted_wins = total_w = 0.0
        for m in ms:
            days = (as_of - m.date).days
            w = 0.9 ** (days / 365.25)
            total_w += w
            if m.winner_id == p1_id:
                weighted_wins += w
        return weighted_wins / total_w if total_w > 0 else 0.5

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

    def get_ranking(pid: int) -> float:
        p = player_map.get(pid)
        rank = p.current_ranking if p and p.current_ranking else 500
        return float(rank)

    def get_age_score(pid: int) -> float:
        p = player_map.get(pid)
        if not p or not p.dob:
            return -2.0
        age = (as_of - p.dob).days / 365.25
        return -abs(age - 27.0)

    def get_surface_form(pid: int) -> float:
        ms = [m for m in get_player_matches_before(pid)
              if m.surface == surface and m.winner_id is not None][-10:]
        results = [1 if m.winner_id == pid else 0 for m in ms]
        return recent_form(results)

    def get_second_serve(pid: int) -> float:
        all_stats = stats_map.get(pid, [])
        valid = [s for s in all_stats if s.second_serve_won_pct][-20:]
        if not valid:
            return 0.5
        return float(np.mean([s.second_serve_won_pct / 100 for s in valid]))

    def get_aggression(pid: int) -> float:
        all_stats = stats_map.get(pid, [])
        valid = [s for s in all_stats
                 if s.winners is not None and s.unforced_errors is not None][-20:]
        if not valid:
            return 0.5
        ratios = []
        for s in valid:
            total = (s.winners or 0) + (s.unforced_errors or 0)
            if total > 0:
                ratios.append(s.winners / total)
        return float(np.mean(ratios)) if ratios else 0.5

    def get_big_match_winrate(pid: int) -> float:
        ms = [m for m in get_player_matches_before(pid)
              if m.tournament_category in _BIG_CATEGORIES
              and m.date >= cutoff_2y and m.winner_id is not None]
        if not ms:
            return 0.5
        wins = sum(1 for m in ms if m.winner_id == pid)
        return wins / len(ms)

    return [
        # ── v1-v2 (12+6 = 18 features) ───────────────────────────────────────
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
        float(get_ranking(p2_id) - get_ranking(p1_id)),
        float(get_age_score(p1_id) - get_age_score(p2_id)),
        float(get_surface_form(p1_id) - get_surface_form(p2_id)),
        float(get_second_serve(p1_id) - get_second_serve(p2_id)),
        float(get_aggression(p1_id) - get_aggression(p2_id)),
        float(get_big_match_winrate(p1_id) - get_big_match_winrate(p2_id)),
    ]


# ── Training ──────────────────────────────────────────────────────────────────

def _xgb_config(n_samples: int) -> dict:
    """Scale estimators down for smaller datasets to avoid overfitting."""
    if n_samples >= 30_000:
        n_est = 150
    elif n_samples >= 10_000:
        n_est = 120
    elif n_samples >= 3_000:
        n_est = 80
    else:
        n_est = 60
    return dict(
        n_estimators=n_est, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=1, tree_method="hist",
    )


def train_model(db: Session, surface: str = "global", log_cb=None, _cache=None) -> object:
    """
    Train XGBoost model for a specific surface (or 'global' for all surfaces).
    log_cb(msg) is called in real-time for UI progress.
    _cache: pre-built memory cache (pass from train_all_models to avoid 5× DB reload).
    """
    import random as _random

    def emit(msg: str) -> None:
        logger.info(msg)
        if log_cb:
            try:
                log_cb(msg)
            except Exception:
                pass

    label = surface.upper()
    if _cache is None:
        emit(f"[INFO] [{label}] Loading data into memory...")
        cache = _build_memory_cache(db)
    else:
        emit(f"[INFO] [{label}] Using shared memory cache.")
        cache = _cache
    all_matches = [m for m in cache["all_matches"] if m.winner_id is not None]

    # Surface-specific models train only on matching surface matches
    if surface != "global":
        all_matches = [m for m in all_matches if m.surface == surface]

    emit(f"[INFO] [{label}] {len(all_matches)} completed matches.")

    _random.seed(42)
    _random.shuffle(all_matches)

    current_year = date.today().year
    X_train, y_train, W_train = [], [], []
    X_test, y_test = [], []
    skipped = 0
    total = len(all_matches)

    emit(f"[INFO] [{label}] Building {total} feature vectors...")
    for i, match in enumerate(all_matches):
        if i % 10000 == 0 and i > 0:
            emit(f"[INFO] [{label}] Features {i}/{total}...")
        try:
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
            label_y = 1 if match.winner_id == p1_id else 0

            # Temporal weight: 15 % decay per year — recent matches matter more
            w = 0.85 ** max(0, current_year - match.date.year)

            if match.date < TRAIN_CUTOFF:
                X_train.append(vec)
                y_train.append(label_y)
                W_train.append(w)
            else:
                X_test.append(vec)
                y_test.append(label_y)
        except Exception as exc:
            skipped += 1
            logger.debug("Skipped match %s: %s", getattr(match, 'id', '?'), exc)

    lbl = surface.upper()
    emit(f"[INFO] [{lbl}] Dataset: {len(X_train)} train / {len(X_test)} test / {skipped} skipped.")

    if len(X_train) < _MIN_SURFACE_SAMPLES:
        raise RuntimeError(
            f"[{lbl}] Only {len(X_train)} train samples (need ≥{_MIN_SURFACE_SAMPLES}). "
            "Run Scrape first or use global model."
        )

    X_tr = np.array(X_train, dtype=np.float32)
    y_tr = np.array(y_train, dtype=int)
    W_tr = np.array(W_train, dtype=np.float32)
    W_tr /= W_tr.mean()   # normalise to mean=1 for numerical stability

    from sklearn.model_selection import train_test_split
    strat = y_tr if np.bincount(y_tr).min() >= 2 else None
    X_fit, X_cal, y_fit, y_cal, w_fit, w_cal = train_test_split(
        X_tr, y_tr, W_tr, test_size=0.2, random_state=42, stratify=strat
    )

    cfg = _xgb_config(len(X_fit))
    emit(f"[INFO] [{lbl}] XGBoost n_estimators={cfg['n_estimators']} on {len(X_fit)} samples...")
    base = xgb.XGBClassifier(**cfg)
    base.fit(X_fit, y_fit, sample_weight=w_fit)

    emit(f"[INFO] [{lbl}] Calibrating (Platt scaling)...")
    model = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    model.fit(X_cal, y_cal, sample_weight=w_cal)

    if X_test and len(X_test) >= 10:
        X_te = np.array(X_test, dtype=np.float32)
        y_te = np.array(y_test, dtype=int)
        proba = model.predict_proba(X_te)[:, 1]
        brier = brier_score_loss(y_te, proba)
        ll = log_loss(y_te, proba)
        emit(f"[INFO] [{lbl}] Eval — Brier: {brier:.4f}  LogLoss: {ll:.4f}")

    mname = _model_name(surface)
    emit(f"[INFO] [{lbl}] Saving as '{mname}'...")
    save_model(model, db, model_name=mname)
    return model


def train_all_models(db: Session, log_cb=None) -> dict:
    """
    Train global + all surface-specific models in sequence.
    Builds memory cache ONCE and reuses across all 5 training runs.
    Returns dict of surface → model for all successfully trained models.
    """
    def emit(msg: str) -> None:
        logger.info(msg)
        if log_cb:
            try:
                log_cb(msg)
            except Exception:
                pass

    # Build DB cache once — avoids 5× full reload (main OOM cause)
    emit("[INFO] Building shared memory cache (1× DB load for all models)...")
    shared_cache = _build_memory_cache(db)
    emit(f"[INFO] Cache ready: {len(shared_cache['all_matches'])} matches loaded.")

    trained = {}

    # Global first — serves as fallback for any surface
    emit("[INFO] ===== Training GLOBAL model (1/5) =====")
    try:
        trained["global"] = train_model(db, surface="global", log_cb=log_cb, _cache=shared_cache)
    except Exception as e:
        emit(f"[ERROR] Global model failed: {e}")
        import traceback as _tb
        emit(_tb.format_exc())

    # Surface-specific models
    for idx, surf in enumerate(SURFACES, start=2):
        emit(f"[INFO] ===== Training {surf.upper()} model ({idx}/5) =====")
        try:
            trained[surf] = train_model(db, surface=surf, log_cb=log_cb, _cache=shared_cache)
        except Exception as e:
            emit(f"[WARN] {surf}: {e} — will use global fallback.")

    emit(f"[OK] Training complete — {len(trained)} models trained: {list(trained.keys())}. ✅")
    return trained


# ── Model cache + loading ─────────────────────────────────────────────────────

_cached_models: dict = {}   # surface/key → model object
_model_lock = threading.Lock()


def get_model(db: Session, surface: str = "global") -> object:
    """Return surface-specific model if trained, else global fallback."""
    surf = surface if surface in SURFACES else "global"

    if surf in _cached_models:
        return _cached_models[surf]

    with _model_lock:
        if surf not in _cached_models:
            # Try surface-specific first
            model = load_model(db, _model_name(surf))
            if model is None and surf != "global":
                logger.info("No %s model — falling back to global.", surf)
                model = load_model(db, _model_name("global"))
            if model is None:
                logger.info("No model in DB — training global now.")
                model = train_model(db, surface="global")
            _cached_models[surf] = model

    return _cached_models[surf]


def predict(
    player1_id: int,
    player2_id: int,
    surface: str,
    tournament_category: str,
    db: Session,
    tournament_name: str = "",
    as_of: Optional[date] = None,
) -> dict:
    model = get_model(db, surface)
    as_of = as_of or date.today()
    vec = build_feature_vector(
        player1_id, player2_id, surface, tournament_category,
        as_of, db, tournament_name,
    )
    X = np.array([vec], dtype=np.float32)
    proba = model.predict_proba(X)[0]
    return {
        "p1_win_prob": float(proba[1]),
        "p2_win_prob": float(proba[0]),
    }

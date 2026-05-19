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


def train_model(db: Session) -> object:
    from data.db import Match

    logger.info("Building training dataset...")
    matches = (
        db.query(Match)
        .filter(Match.date.isnot(None), Match.winner_id.isnot(None))
        .order_by(Match.date)
        .all()
    )

    X_train, y_train, X_test, y_test = [], [], [], []
    skipped = 0
    for match in matches:
        try:
            vec = build_feature_vector(
                match.player1_id, match.player2_id,
                match.surface or "hard",
                match.tournament_category or "250",
                match.date, db,
                match.tournament_name or "",
            )
            label = 1 if match.winner_id == match.player1_id else 0
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

    if len(X_train) < 100:
        raise RuntimeError("Not enough training data (need >= 100 matches).")

    X_tr = np.array(X_train, dtype=float)
    y_tr = np.array(y_train, dtype=int)

    base = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        n_jobs=-1,
    )
    model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
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

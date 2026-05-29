"""
Backtest — evaluate trained model on holdout matches (date >= TRAIN_CUTOFF).

CLI usage:
    python -m scripts.backtest
    python -m scripts.backtest --surface clay
    python -m scripts.backtest --from-date 2024-01-01
    python -m scripts.backtest --min-conf 0.65

Also callable as a library:
    from scripts.backtest import run_backtest
    run_backtest(db, log_cb=print)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from datetime import date

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def _parse_args():
    p = argparse.ArgumentParser(description="Backtest tennis prediction model.")
    p.add_argument("--surface", default="all",
                   choices=["all", "clay", "hard", "grass", "indoor"],
                   help="Filter test matches by surface (default: all)")
    p.add_argument("--from-date", default=None,
                   help="Start of test window, YYYY-MM-DD (default: TRAIN_CUTOFF)")
    p.add_argument("--to-date", default=str(date.today()),
                   help="End of test window, YYYY-MM-DD (default: today)")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="Only evaluate predictions where max(p1,p2) >= threshold")
    return p.parse_args()


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _metrics(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    preds = (y_prob >= 0.5).astype(int)
    acc = float(np.mean(preds == y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))
    return acc, brier, ll


def _fmt_metrics(label, y_true, y_prob) -> list[str]:
    """Return list of formatted lines for a metric block."""
    if len(y_true) < 10:
        return []
    acc, brier, ll = _metrics(y_true, y_prob)
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    n = len(y_true)

    mask_hc = y_prob_arr > 0.65
    hc_n = int(mask_hc.sum())
    hc_acc = (
        float(np.mean((y_prob_arr[mask_hc] >= 0.5) == y_true_arr[mask_hc]))
        if hc_n > 5 else None
    )

    lines = [f"  -- {label} (n={n})"]
    lines.append(f"     Accuracy:          {acc:.1%}")
    if hc_acc is not None:
        lines.append(f"     Accuracy (p>65%):  {hc_acc:.1%}  (n={hc_n})")
    lines.append(f"     Brier score:       {brier:.4f}  (random = 0.2500)")
    lines.append(f"     Log-loss:          {ll:.4f}  (random = 0.6931)")
    return lines


def _fmt_calibration(y_true, y_prob) -> list[str]:
    lines = ["  -- CALIBRATION (predicted prob vs actual win rate)"]
    lines.append(f"  {'Range':<10}  {'N':>6}  {'Actual%':>9}  {'Pred%':>8}  {'Delta':>7}")
    lines.append(f"  {'─'*50}")
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    for lo_i in range(0, 10):
        lo, hi = lo_i / 10, (lo_i + 1) / 10
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n < 10:
            continue
        actual = float(y_true[mask].mean())
        pred = float(y_prob[mask].mean())
        delta = actual - pred
        arrow = "^" if delta > 0.04 else ("v" if delta < -0.04 else "=")
        lines.append(
            f"  {lo:.0%}-{hi:.0%}       {n:>6}   {actual:>7.1%}   {pred:>7.1%}  {arrow}{delta:+.2f}"
        )
    return lines


def _fmt_roi(y_true, y_prob, kelly_fraction=0.25) -> list[str]:
    """
    Synthetic ROI: assumes 5% bookmaker margin applied to model probabilities.
    Directional signal only — not real P&L (no historical bookmaker odds available).
    """
    MARGIN = 1.05
    MIN_EDGE = 0.04

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bankroll = 1.0
    bets_placed = 0
    wins = 0
    peak = 1.0

    for p, outcome in zip(y_prob, y_true):
        p_bookie = p / MARGIN
        edge = p - p_bookie
        if edge < MIN_EDGE or p >= 1.0:
            continue
        kelly = (edge * p / (1.0 - p)) * kelly_fraction
        stake = min(kelly * bankroll, bankroll * 0.10)
        odds = 1.0 / p_bookie
        if outcome == 1:
            bankroll += stake * (odds - 1)
            wins += 1
        else:
            bankroll -= stake
        bets_placed += 1
        peak = max(peak, bankroll)

    if bets_placed == 0:
        return ["  -- ROI: no bets qualified (edge < 4%)"]

    roi = (bankroll - 1.0) * 100
    win_rate = wins / bets_placed
    max_dd = (peak - bankroll) / peak * 100

    lines = [f"  -- ROI SIMULATION (synthetic 5%-margin odds, Kelly {kelly_fraction:.0%})"]
    lines.append(f"     NOTE: proxy odds only — directional signal, not real P&L")
    lines.append(f"     Bets placed:    {bets_placed}")
    lines.append(f"     Win rate:       {win_rate:.1%}")
    lines.append(f"     Final bankroll: {bankroll:.4f}  (started 1.000)")
    lines.append(f"     Total ROI:      {roi:+.1f}%")
    lines.append(f"     Max drawdown:   {max_dd:.1f}%")
    return lines


# ── Core backtest logic (library-callable) ────────────────────────────────────

def run_backtest(
    db,
    surface: str = "all",
    from_date=None,
    to_date=None,
    min_conf: float = 0.0,
    log_cb=None,
) -> dict:
    """
    Run backtest and return results dict.
    log_cb(msg): called for each output line (for streaming to UI / CLI).
    """
    from data.db import Match
    from models.predictor import (
        TRAIN_CUTOFF,
        _build_feature_fast,
        _build_memory_cache,
        _model_name,
        load_model,
    )

    def emit(msg: str):
        if log_cb:
            try:
                log_cb(msg)
            except Exception:
                pass
        else:
            print(msg)

    if from_date is None:
        from_date = TRAIN_CUTOFF
    if to_date is None:
        to_date = date.today()

    sep = "=" * 60
    emit(sep)
    emit(f"  TENNIS MODEL BACKTEST")
    emit(f"  Period:  {from_date}  ->  {to_date}")
    emit(f"  Surface: {surface}")
    if min_conf > 0:
        emit(f"  Min confidence: >={min_conf:.0%}")
    emit(sep)

    # Load test matches
    emit("[INFO] Loading test matches...")
    query = (
        db.query(Match)
        .filter(
            Match.date >= from_date,
            Match.date <= to_date,
            Match.winner_id.isnot(None),
            Match.player1_id.isnot(None),
            Match.player2_id.isnot(None),
            Match.date.isnot(None),
        )
        .order_by(Match.date)
    )
    if surface != "all":
        query = query.filter(Match.surface == surface)
    test_matches = query.all()

    emit(f"[INFO] {len(test_matches)} test matches found.")
    if len(test_matches) < 30:
        emit("[ERROR] Too few matches (< 30). Expand date range or check DB.")
        return {"error": "too few matches"}

    # Build memory cache
    emit("[INFO] Building memory cache...")
    cache = _build_memory_cache(db)

    # Lazy model cache
    _loaded_models: dict = {}

    def get_model_cached(surf: str):
        if surf not in _loaded_models:
            m = load_model(db, _model_name(surf))
            if m is None:
                m = load_model(db, _model_name("global"))
            _loaded_models[surf] = m
        return _loaded_models[surf]

    # Run predictions
    emit("[INFO] Running predictions...")
    all_y_true, all_y_prob = [], []
    by_surface: dict = defaultdict(lambda: {"y_true": [], "y_prob": []})
    by_year: dict = defaultdict(lambda: {"y_true": [], "y_prob": []})
    skipped = 0

    for i, m in enumerate(test_matches):
        if i % 3000 == 0 and i > 0:
            emit(f"[INFO]   {i}/{len(test_matches)} (skipped {skipped})")

        surf = m.surface or "hard"
        model = get_model_cached(surf)
        if model is None:
            skipped += 1
            continue

        try:
            fv = _build_feature_fast(
                m.player1_id, m.player2_id,
                surf, m.tournament_category or "250",
                m.date, m.tournament_name or "",
                cache,
            )
            proba = model.predict_proba([fv])[0]
            p1_win = float(proba[1])
        except Exception:
            skipped += 1
            continue

        label = 1 if m.winner_id == m.player1_id else 0
        if min_conf > 0 and max(p1_win, 1 - p1_win) < min_conf:
            continue

        all_y_true.append(label)
        all_y_prob.append(p1_win)
        by_surface[surf]["y_true"].append(label)
        by_surface[surf]["y_prob"].append(p1_win)
        by_year[m.date.year]["y_true"].append(label)
        by_year[m.date.year]["y_prob"].append(p1_win)

    emit(f"[INFO] Evaluated: {len(all_y_true)}  Skipped: {skipped}")

    if len(all_y_true) < 10:
        emit("[ERROR] Not enough predictions.")
        return {"error": "not enough predictions"}

    # ── Emit results ─────────────────────────────────────────────────────────
    sep2 = "-" * 60

    emit(sep2)
    emit("  OVERALL METRICS")
    emit(sep2)
    for line in _fmt_metrics("ALL SURFACES", all_y_true, all_y_prob):
        emit(line)

    if len(by_surface) > 1:
        emit(sep2)
        emit("  BY SURFACE")
        emit(sep2)
        for surf in ["hard", "clay", "grass", "indoor"]:
            data = by_surface.get(surf)
            if data and len(data["y_true"]) >= 20:
                for line in _fmt_metrics(surf.capitalize(), data["y_true"], data["y_prob"]):
                    emit(line)

    if len(by_year) > 1:
        emit(sep2)
        emit("  BY YEAR")
        emit(sep2)
        for yr in sorted(by_year.keys()):
            data = by_year[yr]
            if len(data["y_true"]) >= 20:
                for line in _fmt_metrics(str(yr), data["y_true"], data["y_prob"]):
                    emit(line)

    emit(sep2)
    emit("  CALIBRATION")
    emit(sep2)
    for line in _fmt_calibration(all_y_true, all_y_prob):
        emit(line)

    emit(sep2)
    emit("  STAKING SIMULATION")
    emit(sep2)
    for line in _fmt_roi(all_y_true, all_y_prob):
        emit(line)

    emit(sep)
    emit("[OK] Backtest completato. ✅")

    # Return summary dict
    acc, brier, ll = _metrics(all_y_true, all_y_prob)
    return {
        "n": len(all_y_true),
        "skipped": skipped,
        "accuracy": acc,
        "brier": brier,
        "log_loss": ll,
        "by_surface": {
            s: _metrics(d["y_true"], d["y_prob"])
            for s, d in by_surface.items()
            if len(d["y_true"]) >= 20
        },
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    from data.db import SessionLocal
    from models.predictor import TRAIN_CUTOFF

    from_date = date.fromisoformat(args.from_date) if args.from_date else TRAIN_CUTOFF
    to_date = date.fromisoformat(args.to_date)

    db = SessionLocal()
    try:
        run_backtest(
            db,
            surface=args.surface,
            from_date=from_date,
            to_date=to_date,
            min_conf=args.min_conf,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()

"""
Backtest — evaluate trained model on holdout matches (date >= TRAIN_CUTOFF).

Usage:
    python -m scripts.backtest
    python -m scripts.backtest --surface clay
    python -m scripts.backtest --from-date 2024-01-01
    python -m scripts.backtest --surface hard --from-date 2023-06-01 --to-date 2024-12-31
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


# ── Metrics / display helpers ─────────────────────────────────────────────────

def _metrics(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    preds = (y_prob >= 0.5).astype(int)
    acc = float(np.mean(preds == y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))
    return acc, brier, ll


def _print_metrics_row(label, y_true, y_prob, width=52):
    if len(y_true) < 10:
        return
    acc, brier, ll = _metrics(y_true, y_prob)
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    n = len(y_true)

    # High-confidence subset (model prob > 0.65)
    mask_hc = y_prob_arr > 0.65
    hc_n = int(mask_hc.sum())
    hc_acc = float(np.mean((y_prob_arr[mask_hc] >= 0.5) == y_true_arr[mask_hc])) if hc_n > 5 else None

    print(f"\n  ── {label} (n={n}) {'─' * max(2, width - len(label) - len(str(n)) - 8)}")
    print(f"     Accuracy:          {acc:.1%}")
    if hc_acc is not None:
        print(f"     Accuracy (p>65%):  {hc_acc:.1%}  (n={hc_n})")
    print(f"     Brier score:       {brier:.4f}  (random baseline = 0.2500)")
    print(f"     Log-loss:          {ll:.4f}  (random baseline = 0.6931)")


def _print_calibration(y_true, y_prob):
    print("\n  ── CALIBRATION ─────────────────────────────────────────────")
    print(f"  {'Pred range':<12}  {'N':>6}  {'Actual%':>9}  {'Pred%':>8}  {'Δ':>7}  {'Bar'}")
    print(f"  {'─'*60}")
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
        # ASCII bar: each block = 5%
        bar_val = actual
        bar = "█" * int(bar_val * 20) if bar_val > 0 else ""
        arrow = "▲" if delta > 0.04 else ("▼" if delta < -0.04 else " ")
        print(f"  {lo:.0%}–{hi:.0%}         {n:>6}   {actual:>7.1%}   {pred:>7.1%}  {arrow}{delta:+.2f}  {bar}")


def _print_roi_simulation(y_true, y_prob, kelly_fraction=0.25):
    """
    Simulate Kelly staking assuming 'true' bookmaker implied prob = 1 - model_prob
    of the opponent (no real odds available for historical matches).
    Uses a synthetic 5% margin bookmaker as proxy:
      fair_prob = p_model  → implied_odds = 1 / (p_model * 1.05)
    This tests whether the model's confidence earns positive returns against
    a margin-adjusted bookie (not a real P&L — directional signal only).
    """
    MARGIN = 1.05    # 5% bookmaker overround (conservative)
    MIN_EDGE = 0.04  # minimum edge to place bet

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bankroll = 1.0
    history = [1.0]
    bets_placed = 0
    wins = 0

    for p, outcome in zip(y_prob, y_true):
        # Synthetic bookie odds (with 5% margin on favourited side)
        # p_bookie_implied = p / MARGIN  (bookie underestimates their own prob slightly)
        # We bet on p1 if edge > threshold
        p_bookie = p / MARGIN
        edge = p - p_bookie
        if edge < MIN_EDGE:
            history.append(bankroll)
            continue

        # Fractional Kelly: f = edge * p / (1 - p) * kelly_fraction
        if p >= 1.0:
            history.append(bankroll)
            continue
        kelly = (edge * p / (1.0 - p)) * kelly_fraction
        stake = min(kelly * bankroll, bankroll * 0.10)  # cap single bet at 10%

        # Implied decimal odds
        odds = 1.0 / p_bookie
        if outcome == 1:
            bankroll += stake * (odds - 1)
            wins += 1
        else:
            bankroll -= stake
        bets_placed += 1
        history.append(bankroll)

    if bets_placed == 0:
        return

    roi = (bankroll - 1.0) * 100
    win_rate = wins / bets_placed if bets_placed else 0

    print(f"\n  ── ROI SIMULATION (proxy odds, Kelly {kelly_fraction:.0%}) ─────────────────")
    print(f"     Note: uses synthetic 5%-margin odds — directional signal only.")
    print(f"     Bets placed:   {bets_placed}")
    print(f"     Win rate:      {win_rate:.1%}")
    print(f"     Final bankroll:{bankroll:.4f}  (started 1.000)")
    print(f"     Total ROI:     {roi:+.1f}%")

    # Equity curve (ASCII, 40 chars wide)
    _print_equity_curve(history, width=52)


def _print_equity_curve(history, width=52):
    if len(history) < 4:
        return
    height = 8
    h = np.array(history)
    lo, hi = h.min(), h.max()
    rng = hi - lo if hi != lo else 0.001

    print(f"\n  ── EQUITY CURVE ────────────────────────────────────────────")
    # Downsample to width points
    idx = np.linspace(0, len(h) - 1, width).astype(int)
    vals = h[idx]
    norm = ((vals - lo) / rng * (height - 1)).astype(int)

    # Build grid
    grid = [[" "] * width for _ in range(height)]
    for x, y in enumerate(norm):
        grid[height - 1 - y][x] = "·"

    # Baseline at y=1.0 (start)
    baseline_y = int((1.0 - lo) / rng * (height - 1))
    baseline_row = height - 1 - baseline_y
    for x in range(width):
        if grid[baseline_row][x] == " ":
            grid[baseline_row][x] = "─"

    print(f"  {hi:.3f} ┤ {''.join(grid[0])}")
    for row in grid[1:-1]:
        print(f"         │ {''.join(row)}")
    print(f"  {lo:.3f} ┤ {''.join(grid[-1])}")
    print(f"         └─{'─' * width}")
    print(f"           {'start':^{width // 2}}{'end':^{width // 2}}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    from data.db import Match, SessionLocal
    from models.predictor import (
        TRAIN_CUTOFF,
        _build_feature_fast,
        _build_memory_cache,
        _model_name,
        load_model,
    )

    from_date = date.fromisoformat(args.from_date) if args.from_date else TRAIN_CUTOFF
    to_date = date.fromisoformat(args.to_date)

    print(f"\n{'═' * 64}")
    print(f"  TENNIS MODEL BACKTEST")
    print(f"  Period:  {from_date}  →  {to_date}")
    print(f"  Surface: {args.surface}")
    if args.min_conf > 0:
        print(f"  Min confidence filter: ≥{args.min_conf:.0%}")
    print(f"{'═' * 64}")

    db = SessionLocal()
    try:
        # Load test matches
        print("\n  Loading test matches from DB...")
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
        if args.surface != "all":
            query = query.filter(Match.surface == args.surface)
        test_matches = query.all()

        print(f"  Found {len(test_matches)} test matches.")
        if len(test_matches) < 30:
            print("  ✗ Too few matches (< 30). Expand date range or check DB.")
            return

        # Build in-memory cache
        print("  Building memory cache...")
        cache = _build_memory_cache(db)

        # Load models (lazy per surface)
        _loaded_models: dict = {}

        def get_model_cached(surface: str):
            if surface not in _loaded_models:
                m = load_model(db, _model_name(surface))
                if m is None:
                    m = load_model(db, _model_name("global"))
                _loaded_models[surface] = m
            return _loaded_models[surface]

        # Run predictions
        print("  Running predictions...\n")
        all_y_true, all_y_prob = [], []
        by_surface: dict = defaultdict(lambda: {"y_true": [], "y_prob": []})
        by_year: dict = defaultdict(lambda: {"y_true": [], "y_prob": []})
        skipped = 0

        for i, m in enumerate(test_matches):
            if i % 2000 == 0 and i > 0:
                print(f"    {i}/{len(test_matches)}  (skipped {skipped})")

            surface = m.surface or "hard"
            model = get_model_cached(surface)
            if model is None:
                skipped += 1
                continue

            try:
                fv = _build_feature_fast(
                    m.player1_id, m.player2_id,
                    surface, m.tournament_category or "250",
                    m.date, m.tournament_name or "",
                    cache,
                )
                proba = model.predict_proba([fv])[0]
                p1_win = float(proba[1])
            except Exception:
                skipped += 1
                continue

            label = 1 if m.winner_id == m.player1_id else 0

            # Min-confidence filter
            if args.min_conf > 0 and max(p1_win, 1 - p1_win) < args.min_conf:
                continue

            all_y_true.append(label)
            all_y_prob.append(p1_win)
            by_surface[surface]["y_true"].append(label)
            by_surface[surface]["y_prob"].append(p1_win)
            by_year[m.date.year]["y_true"].append(label)
            by_year[m.date.year]["y_prob"].append(p1_win)

        total_evaluated = len(all_y_true)
        print(f"    Done.  Evaluated: {total_evaluated}  Skipped: {skipped}")

        if total_evaluated < 10:
            print("  ✗ Not enough predictions to evaluate.")
            return

        # ── Results ──────────────────────────────────────────────────────────
        print(f"\n{'─' * 64}")
        print("  OVERALL METRICS")
        print(f"{'─' * 64}")
        _print_metrics_row("ALL SURFACES", all_y_true, all_y_prob)

        if len(by_surface) > 1:
            print(f"\n{'─' * 64}")
            print("  BY SURFACE")
            print(f"{'─' * 64}")
            for surf in ["hard", "clay", "grass", "indoor"]:
                data = by_surface.get(surf)
                if data and len(data["y_true"]) >= 20:
                    _print_metrics_row(surf.capitalize(), data["y_true"], data["y_prob"])

        if len(by_year) > 1:
            print(f"\n{'─' * 64}")
            print("  BY YEAR")
            print(f"{'─' * 64}")
            for yr in sorted(by_year.keys()):
                data = by_year[yr]
                if len(data["y_true"]) >= 20:
                    _print_metrics_row(str(yr), data["y_true"], data["y_prob"])

        print(f"\n{'─' * 64}")
        print("  CALIBRATION  (predicted prob vs actual win rate)")
        print(f"{'─' * 64}")
        _print_calibration(all_y_true, all_y_prob)

        print(f"\n{'─' * 64}")
        print("  STAKING SIMULATION")
        print(f"{'─' * 64}")
        _print_roi_simulation(all_y_true, all_y_prob)

        print(f"\n{'═' * 64}\n")

    finally:
        db.close()


if __name__ == "__main__":
    main()

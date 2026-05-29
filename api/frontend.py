import html as _html
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from api.auth import (
    _expected_token,
    check_password,
    clear_session_cookie,
    require_auth,
    set_session_cookie,
)
from data.db import (
    EloRating, Match, News, OddsSnapshot, Player, Prediction, SentimentCache, SystemJob, get_db,
)
from data.db import SessionLocal as _SessionLocal
from reports.daily_report import generate_report

router = APIRouter(prefix="/app")
templates = Jinja2Templates(directory="templates")


# ── DB-backed job helpers ───────────────────────────────────────────────────

def _new_job(action: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    db = _SessionLocal()
    try:
        db.add(SystemJob(
            id=job_id,
            action=action,
            status="running",
            log="",
            started_at=datetime.now(timezone.utc),
        ))
        db.commit()
    finally:
        db.close()
    return job_id


def _get_job(job_id: str) -> Optional[dict]:
    db = _SessionLocal()
    try:
        j = db.query(SystemJob).filter_by(id=job_id).first()
        if not j:
            return None
        return {
            "status": j.status,
            "log": j.log.splitlines() if j.log else [],
            "started_at": j.started_at.isoformat() if j.started_at else None,
        }
    finally:
        db.close()


def _append_log(job_id: str, msg: str) -> None:
    db = _SessionLocal()
    try:
        j = db.query(SystemJob).filter_by(id=job_id).with_for_update().first()
        if j:
            j.log = (j.log or "") + msg + "\n"
            db.commit()
    finally:
        db.close()


def _finish_job(job_id: str, status: str) -> None:
    db = _SessionLocal()
    try:
        j = db.query(SystemJob).filter_by(id=job_id).first()
        if j:
            j.status = status
            j.finished_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


# Prune jobs older than 7 days to avoid unbounded growth
def _prune_old_jobs() -> None:
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    db = _SessionLocal()
    try:
        db.query(SystemJob).filter(SystemJob.started_at < cutoff).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _get_movement(bet: dict, db: Session) -> Optional[dict]:
    """Return line-movement stats for a value bet (first snapshot vs current odds)."""
    p1_id = bet.get("p1_id")
    p2_id = bet.get("p2_id")
    vbp = bet.get("value_bet_player")    # 1 or 2
    if not p1_id or not p2_id or not vbp:
        return None

    pa = min(p1_id, p2_id)
    pb = max(p1_id, p2_id)
    match_d = bet["match_date"].date() if bet.get("match_date") else None

    snaps = (
        db.query(OddsSnapshot)
        .filter_by(player_a_id=pa, player_b_id=pb, match_date=match_d)
        .order_by(OddsSnapshot.recorded_at.asc())
        .all()
    )
    if len(snaps) < 2:
        return None  # need at least 2 points to show movement

    first = snaps[0]
    # Resolve which odds column corresponds to the value player
    if vbp == 1:
        first_odds = first.odds_a if pa == p1_id else first.odds_b
        current_odds = bet.get("odds_p1")
    else:
        first_odds = first.odds_b if pa == p1_id else first.odds_a
        current_odds = bet.get("odds_p2")

    if not first_odds or not current_odds or first_odds <= 0:
        return None

    pct = (current_odds - first_odds) / first_odds * 100
    return {
        "first_odds": round(first_odds, 2),
        "current_odds": round(current_odds, 2),
        "pct_change": round(pct, 1),
        "direction": "up" if pct > 0.5 else "down" if pct < -0.5 else "flat",
        "snap_count": len(snaps),
    }


def _sidebar_context(db: Session) -> dict:
    """Shared sidebar data injected into every page context."""
    vb_count = (
        db.query(Prediction)
        .filter(
            Prediction.value_bet_player.isnot(None),
            Prediction.edge_percentage > 5,
        )
        .count()
    )
    return {"vb_count": vb_count}


# ── Auth routes ─────────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")


@router.post("/login")
def login_submit(request: Request, password: str = Form(...)):
    if not check_password(password):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid password"},
        )
    token = _expected_token()
    response = RedirectResponse(url="/app/overview", status_code=303)
    set_session_cookie(response, token)
    return response


@router.get("/logout")
def logout():
    response = RedirectResponse(url="/app/login", status_code=303)
    clear_session_cookie(response)
    return response


@router.get("/", response_class=HTMLResponse)
def root_redirect():
    return RedirectResponse(url="/app/overview", status_code=303)


# ── Overview ─────────────────────────────────────────────────────────────────

@router.get("/overview", response_class=HTMLResponse)
def overview(
    request: Request,
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    report = generate_report(db)

    # System status from app state
    init_done, init_error, init_step = True, None, "done"
    if hasattr(request.app.state, "get_init_status"):
        init_done, init_error, init_step = request.app.state.get_init_status()

    # Top value bet (highest edge)
    top_bet = None
    if report["value_bets"]:
        top_bet = max(report["value_bets"], key=lambda x: x.get("edge_pct") or 0)

    ctx = {
        "active": "overview",
        "report": report,
        "top_bet": top_bet,
        "init_done": init_done,
        "init_step": init_step,
        "init_error": init_error,
        **_sidebar_context(db),
    }
    return templates.TemplateResponse(request, "overview.html", ctx)


# ── Value Bets ───────────────────────────────────────────────────────────────

@router.get("/value-bets", response_class=HTMLResponse)
def value_bets_page(
    request: Request,
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    from data.odds import kelly_stake
    report = generate_report(db)
    bets = sorted(
        report["value_bets"],
        key=lambda x: x.get("edge_pct") or 0,
        reverse=True,
    )
    # Attach line-movement data and Kelly % to each bet
    for bet in bets:
        bet["movement"] = _get_movement(bet, db)
        edge_frac = (bet.get("edge_pct") or 0) / 100.0
        is_p1 = (bet.get("value_bet_player") == 1)
        p_model = bet["p1_win_prob"] if is_p1 else bet["p2_win_prob"]
        bet["kelly_pct"] = round(kelly_stake(edge_frac, p_model) * 100, 1)

    ctx = {
        "active": "value_bets",
        "bets": bets,
        **_sidebar_context(db),
    }
    return templates.TemplateResponse(request, "value_bets.html", ctx)


# ── Predict ──────────────────────────────────────────────────────────────────

@router.get("/predict", response_class=HTMLResponse)
def predict_page(
    request: Request,
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    surfaces = ["hard", "clay", "grass", "indoor"]
    categories = ["Slam", "Masters", "500", "250", "Finals"]
    ctx = {
        "active": "predict",
        "surfaces": surfaces,
        "categories": categories,
        **_sidebar_context(db),
    }
    return templates.TemplateResponse(request, "predict.html", ctx)


@router.post("/predict", response_class=HTMLResponse)
def predict_submit(
    request: Request,
    player1_id: int = Form(...),
    player2_id: int = Form(...),
    surface: str = Form(...),
    category: str = Form(...),
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    from models.predictor import predict as _predict
    _VALID_SURFACES = {"hard", "clay", "grass", "indoor"}
    _VALID_CATEGORIES = {"Slam", "Masters", "500", "250", "Finals"}
    error = None
    result = None

    p1 = db.query(Player).filter_by(id=player1_id).first()
    p2 = db.query(Player).filter_by(id=player2_id).first()

    if surface not in _VALID_SURFACES or category not in _VALID_CATEGORIES:
        error = "Superficie o categoria non valida."
    elif not p1 or not p2:
        error = f"Giocatore non trovato: {'P1' if not p1 else 'P2'} (id={player1_id if not p1 else player2_id})"
    else:
        try:
            pred = _predict(player1_id, player2_id, surface, category, db)
            # ELO delta for selected surface
            elo1 = db.query(EloRating).filter_by(player_id=player1_id, surface=surface).first()
            elo2 = db.query(EloRating).filter_by(player_id=player2_id, surface=surface).first()
            elo_delta = None
            if elo1 and elo2:
                elo_delta = round(elo1.rating - elo2.rating, 0)
            # Surface win rate for p1
            total_p1 = (
                db.query(Match).filter(Match.surface == surface, Match.player1_id == player1_id).count()
                + db.query(Match).filter(Match.surface == surface, Match.player2_id == player1_id).count()
            )
            wins_p1 = db.query(Match).filter(Match.surface == surface, Match.winner_id == player1_id).count()
            win_rate_p1 = round(wins_p1 / total_p1 * 100, 1) if total_p1 > 0 else None

            result = {
                "p1_name": p1.name,
                "p2_name": p2.name,
                "p1_win_prob": round(pred["p1_win_prob"] * 100, 1),
                "p2_win_prob": round(pred["p2_win_prob"] * 100, 1),
                "surface": surface,
                "category": category,
                "elo_delta": elo_delta,
                "win_rate_p1": win_rate_p1,
            }
        except Exception as exc:
            error = str(exc)

    return templates.TemplateResponse(
        request, "partials/predict_result.html",
        {"result": result, "error": error},
    )


# ── News ─────────────────────────────────────────────────────────────────────

@router.get("/news", response_class=HTMLResponse)
def news_page(
    request: Request,
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    # Recent news, most recent first, limit 100
    articles = (
        db.query(News)
        .order_by(News.published_at.desc().nulls_last(), News.id.desc())
        .limit(100)
        .all()
    )

    # Player name lookup
    player_ids = {a.player_id for a in articles if a.player_id}
    players = {
        p.id: p.name
        for p in db.query(Player).filter(Player.id.in_(player_ids)).all()
    } if player_ids else {}

    # Sentiment cache per player
    sentiments = {
        s.player_id: s
        for s in db.query(SentimentCache)
        .filter(SentimentCache.player_id.in_(player_ids))
        .all()
    } if player_ids else {}

    # Group articles by player
    from collections import defaultdict
    grouped: dict = defaultdict(list)
    ungrouped = []
    for a in articles:
        if a.player_id:
            grouped[a.player_id].append(a)
        else:
            ungrouped.append(a)

    player_groups = []
    for pid, arts in sorted(grouped.items(), key=lambda x: x[0]):
        sent = sentiments.get(pid)
        player_groups.append({
            "player_id": pid,
            "player_name": players.get(pid, f"Player #{pid}"),
            "articles": arts,
            "sentiment": sent,
        })

    ctx = {
        "active": "news",
        "player_groups": player_groups,
        "ungrouped": ungrouped,
        "total": len(articles),
        **_sidebar_context(db),
    }
    return templates.TemplateResponse(request, "news.html", ctx)


# ── Players ──────────────────────────────────────────────────────────────────

@router.get("/players", response_class=HTMLResponse)
def players_page(
    request: Request,
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    ctx = {
        "active": "players",
        **_sidebar_context(db),
    }
    return templates.TemplateResponse(request, "players.html", ctx)


# ── System ───────────────────────────────────────────────────────────────────

_ACTION_LABELS = {
    "scrape": "🔄 Scrape",
    "retrain": "🧠 Retrain",
    "refresh_odds": "💹 Refresh Odds",
    "fetch_news": "📰 Fetch News",
}


def _run_action(job_id: str, action: str) -> None:
    """Run a system action in a background thread, persisting log lines to DB."""

    def log(msg: str) -> None:
        _append_log(job_id, msg)

    try:
        if action == "scrape":
            from data.db import SessionLocal
            from data.scraper import run_scraper
            from models.elo import backfill_elo
            log("[INFO] Starting scraper...")
            db = SessionLocal()
            try:
                run_scraper(db)
                log("[INFO] Scraper done. Running ELO update...")
                backfill_elo(db)
                log("[OK] ELO updated. ✅")
            finally:
                db.close()

        elif action == "retrain":
            import models.predictor as pred_module
            from data.db import SessionLocal
            from models.predictor import train_all_models
            log("[INFO] Retrain avviato (tutti i modelli superficie + globale)...")
            db = SessionLocal()
            try:
                trained = train_all_models(db, log_cb=log)
                with pred_module._model_lock:
                    pred_module._cached_models.clear()
                    pred_module._cached_models.update(trained)
                log(f"[OK] Retrain completato: {list(trained.keys())}. ✅")
            finally:
                db.close()

        elif action == "refresh_odds":
            from data.db import SessionLocal
            from data.odds import detect_value_bets
            from models.predictor import predict as _predict
            log("[INFO] Refreshing odds and detecting value bets...")
            db = SessionLocal()
            try:
                bets = detect_value_bets(db, _predict)
                log(f"[OK] {len(bets)} value bets found. ✅")
            finally:
                db.close()

        elif action == "fetch_news":
            from data.db import SessionLocal
            from data.news_fetcher import fetch_news
            from llm.sentiment import run_sentiment_update
            log("[INFO] Fetching news...")
            db = SessionLocal()
            try:
                fetch_news(db)
                log("[INFO] Running sentiment analysis...")
                run_sentiment_update(db)
                log("[OK] News and sentiment updated. ✅")
            finally:
                db.close()

        else:
            log(f"[ERROR] Unknown action: {action}")
            _finish_job(job_id, "failed")
            return

        _finish_job(job_id, "done")

    except Exception as exc:
        import traceback
        log(f"[ERROR] {exc}")
        log(traceback.format_exc())
        _finish_job(job_id, "failed")


@router.get("/system", response_class=HTMLResponse)
def system_page(
    request: Request,
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    ctx = {
        "active": "system",
        "actions": _ACTION_LABELS,
        **_sidebar_context(db),
    }
    return templates.TemplateResponse(request, "system.html", ctx)


@router.post("/system/run/{action}", response_class=HTMLResponse)
def system_run_action(
    action: str,
    request: Request,
    _: None = Depends(require_auth),
):
    if action not in _ACTION_LABELS:
        return HTMLResponse(
            f'<div class="text-red-400 text-sm p-3">Unknown action: {_html.escape(action)}</div>',
            status_code=400,
        )
    _prune_old_jobs()
    job_id = _new_job(action)
    t = threading.Thread(target=_run_action, args=(job_id, action), daemon=True)
    t.start()
    job = _get_job(job_id)
    return templates.TemplateResponse(
        request, "partials/job_status.html",
        {"job": job, "job_id": job_id},
    )


@router.get("/system/job/{job_id}", response_class=HTMLResponse)
def system_job_status(
    job_id: str,
    request: Request,
    _: None = Depends(require_auth),
):
    job = _get_job(job_id)
    if not job:
        # No polling attribute → HTMX stops automatically
        return HTMLResponse(
            '<div class="text-amber-500 text-xs p-3 bg-[#1a1a2e] rounded">'
            '⚠️ Job non trovato — il servizio è stato riavviato durante l\'esecuzione. '
            'Rilancia l\'azione dal pannello System.'
            '</div>'
        )
    return templates.TemplateResponse(
        request, "partials/job_status.html",
        {"job": job, "job_id": job_id},
    )


@router.get("/system/stats", response_class=HTMLResponse)
def system_stats(
    request: Request,
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    stats = {
        "matches": db.query(Match).count(),
        "players": db.query(Player).count(),
        "predictions": db.query(Prediction).count(),
        "jobs": 0,
    }
    try:
        from scheduler import scheduler
        stats["jobs"] = len(scheduler.get_jobs())
    except Exception:
        pass
    return templates.TemplateResponse(
        request, "partials/db_stats.html",
        {"stats": stats},
    )


@router.get("/players/autocomplete", response_class=HTMLResponse)
def players_autocomplete(
    request: Request,
    q: str = "",
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    players = []
    if q and len(q) >= 2:
        players = (
            db.query(Player)
            .filter(Player.name.ilike(f"%{q}%"))
            .order_by(Player.current_ranking.asc())
            .limit(8)
            .all()
        )
    return templates.TemplateResponse(
        request, "partials/player_autocomplete.html",
        {"players": players},
    )


@router.get("/players/search", response_class=HTMLResponse)
def players_search(
    request: Request,
    q: str = "",
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    player_data = []
    if q and len(q) >= 2:
        players = (
            db.query(Player)
            .filter(Player.name.ilike(f"%{q}%"))
            .order_by(Player.current_ranking.asc())
            .limit(20)
            .all()
        )
        for p in players:
            ratings = db.query(EloRating).filter_by(player_id=p.id).all()
            player_data.append({
                "id": p.id,
                "name": p.name,
                "nationality": p.nationality or "—",
                "ranking": p.current_ranking,
                "elo": {r.surface: round(r.rating, 0) for r in ratings},
            })

    return templates.TemplateResponse(
        request, "partials/player_results.html",
        {"players": player_data, "q": q},
    )

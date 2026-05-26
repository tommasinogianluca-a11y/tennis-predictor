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
    EloRating, Match, Player, Prediction, get_db,
)
from reports.daily_report import generate_report

router = APIRouter(prefix="/app")
templates = Jinja2Templates(directory="templates")

# ── In-memory job registry ──────────────────────────────────────────────────
_jobs: dict[str, dict] = {}


def _new_job() -> str:
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "running",
        "log": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    return job_id


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
    report = generate_report(db)
    bets = sorted(
        report["value_bets"],
        key=lambda x: x.get("edge_pct") or 0,
        reverse=True,
    )
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
    """Run a system action in a background thread, appending log lines to _jobs[job_id]."""

    def log(msg: str) -> None:
        _jobs[job_id]["log"].append(msg)

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
            import io
            from contextlib import redirect_stdout
            import models.predictor as pred_module
            from data.db import SessionLocal
            from models.predictor import train_model
            log("[INFO] Starting model retrain (~5 min)...")
            db = SessionLocal()
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    new_model = train_model(db)
                for line in buf.getvalue().strip().splitlines():
                    log(line)
                with pred_module._model_lock:
                    pred_module._cached_model = new_model
                log("[OK] Retrain complete. ✅")
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
            _jobs[job_id]["status"] = "failed"
            return

        _jobs[job_id]["status"] = "done"

    except Exception as exc:
        import traceback
        log(f"[ERROR] {exc}")
        log(traceback.format_exc())
        _jobs[job_id]["status"] = "failed"


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
    job_id = _new_job()
    t = threading.Thread(target=_run_action, args=(job_id, action), daemon=True)
    t.start()
    return templates.TemplateResponse(
        request, "partials/job_status.html",
        {"job": _jobs[job_id], "job_id": job_id},
    )


@router.get("/system/job/{job_id}", response_class=HTMLResponse)
def system_job_status(
    job_id: str,
    request: Request,
    _: None = Depends(require_auth),
):
    job = _jobs.get(job_id)
    if not job:
        return HTMLResponse('<div class="text-red-400 text-sm p-3">Job not found.</div>')
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

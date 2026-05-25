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
    error = None
    result = None

    p1 = db.query(Player).filter_by(id=player1_id).first()
    p2 = db.query(Player).filter_by(id=player2_id).first()

    if not p1 or not p2:
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

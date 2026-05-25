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
    from reports.daily_report import generate_report
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

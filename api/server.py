import hmac
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import API_SECRET_KEY
from data.db import EloRating, Player, Prediction, get_db
from models.predictor import predict as _predict
from reports.daily_report import generate_markdown, generate_report

START_TIME = time.time()


async def verify_api_key(x_api_key: str = Header(...)):
    if not API_SECRET_KEY or not hmac.compare_digest(x_api_key, API_SECRET_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key.")


app = FastAPI(title="Tennis Predictor", version="1.0.0")


@app.get("/")
def health_check(request: Request):
    init_done, init_error, init_step = True, None, "done"
    if hasattr(request.app.state, "get_init_status"):
        init_done, init_error, init_step = request.app.state.get_init_status()
    return {
        "status": "ok",
        "uptime": round(time.time() - START_TIME, 1),
        "init_done": init_done,
        "init_step": init_step,
        "init_error": init_error if init_error else None,
    }


@app.get("/report/today")
def report_today(
    _: None = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    return generate_report(db)


@app.get("/report/today/md", response_class=PlainTextResponse)
def report_today_md(
    _: None = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    return generate_markdown(db)


@app.get("/matches/upcoming")
def matches_upcoming(
    _: None = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    # Returns 50 most recent predictions (not filtered to future dates).
    # Use /report/today for same-day predictions.
    predictions = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id": p.id,
            "player1_id": p.player1_id,
            "player2_id": p.player2_id,
            "surface": p.surface,
            "category": p.tournament_category,
            "p1_win_prob": p.p1_win_probability,
            "p2_win_prob": p.p2_win_probability,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in predictions
    ]


@app.get("/player/{player_id}")
def player_detail(
    player_id: int,
    _: None = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    player = db.query(Player).filter_by(id=player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found.")
    ratings = db.query(EloRating).filter_by(player_id=player_id).all()
    return {
        "id": player.id,
        "name": player.name,
        "nationality": player.nationality,
        "current_ranking": player.current_ranking,
        "elo_ratings": {r.surface: round(r.rating, 1) for r in ratings},
    }


@app.get("/value-bets")
def value_bets(
    _: None = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    preds = (
        db.query(Prediction)
        .filter(
            Prediction.value_bet_player.isnot(None),
            Prediction.edge_percentage > 5,
        )
        .order_by(Prediction.created_at.desc())
        .limit(20)
        .all()
    )
    return [
        {
            "player1_id": p.player1_id,
            "player2_id": p.player2_id,
            "value_on": p.value_bet_player,
            "edge_pct": p.edge_percentage,
            "p1_win_prob": p.p1_win_probability,
            "bookmaker_odds_p1": p.bookmaker_odds_p1,
            "bookmaker_odds_p2": p.bookmaker_odds_p2,
        }
        for p in preds
    ]


class PredictRequest(BaseModel):
    player1_id: int
    player2_id: int
    surface: str
    tournament_category: str
    tournament_name: Optional[str] = ""


@app.post("/predict")
def predict_endpoint(
    body: PredictRequest,
    _: None = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    p1 = db.query(Player).filter_by(id=body.player1_id).first()
    p2 = db.query(Player).filter_by(id=body.player2_id).first()
    if not p1 or not p2:
        raise HTTPException(status_code=404, detail="One or both players not found.")
    return _predict(
        body.player1_id, body.player2_id,
        body.surface, body.tournament_category,
        db, body.tournament_name or "",
    )

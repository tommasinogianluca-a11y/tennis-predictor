import logging
from datetime import date, datetime

from sqlalchemy.orm import Session

from data.db import Player, Prediction

logger = logging.getLogger(__name__)


def generate_report(db: Session) -> dict:
    today = date.today()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())

    predictions = (
        db.query(Prediction)
        .filter(Prediction.created_at >= start, Prediction.created_at <= end)
        .all()
    )

    player_ids = set()
    for pred in predictions:
        player_ids.add(pred.player1_id)
        player_ids.add(pred.player2_id)
    players = {
        p.id: p.name
        for p in db.query(Player).filter(Player.id.in_(player_ids)).all()
    }

    def _name(pid: int) -> str:
        return players.get(pid, str(pid))

    entries = []
    for pred in predictions:
        entries.append({
            "player1": _name(pred.player1_id),
            "player2": _name(pred.player2_id),
            "surface": pred.surface,
            "category": pred.tournament_category,
            "p1_win_prob": round(pred.p1_win_probability, 3),
            "p2_win_prob": round(pred.p2_win_probability, 3),
            "value_bet": _name(
                pred.player1_id if pred.value_bet_player == 1 else pred.player2_id
            ) if pred.value_bet_player else None,
            "edge_pct": pred.edge_percentage,
            "odds_p1": pred.bookmaker_odds_p1,
            "odds_p2": pred.bookmaker_odds_p2,
            "match_date": pred.match_date,
        })

    value_bets = [e for e in entries if e["value_bet"]]

    return {
        "date": today.isoformat(),
        "total_predictions": len(entries),
        "value_bets_count": len(value_bets),
        "predictions": entries,
        "value_bets": value_bets,
    }


def generate_markdown(db: Session) -> str:
    report = generate_report(db)
    lines = [
        f"# Tennis Predictions — {report['date']}",
        f"**Total predictions:** {report['total_predictions']}  ",
        f"**Value bets:** {report['value_bets_count']}",
        "",
        "## All Predictions",
        "",
        "| Match | Surface | P1 Win% | P2 Win% | Value Bet | Edge |",
        "|---|---|---|---|---|---|",
    ]
    for e in report["predictions"]:
        vb = e["value_bet"] or "—"
        edge = f"{e['edge_pct']:.1f}%" if e["edge_pct"] else "—"
        lines.append(
            f"| {e['player1']} vs {e['player2']} | {e['surface']} | "
            f"{e['p1_win_prob']:.1%} | {e['p2_win_prob']:.1%} | {vb} | {edge} |"
        )
    if report["value_bets"]:
        lines += ["", "## Value Bets", ""]
        for e in report["value_bets"]:
            edge_str = f"{e['edge_pct']:.1f}%" if e["edge_pct"] is not None else "—"
            lines.append(
                f"- **{e['value_bet']}** — edge {edge_str}, "
                f"model prob {e['p1_win_prob']:.1%}"
            )
    return "\n".join(lines)

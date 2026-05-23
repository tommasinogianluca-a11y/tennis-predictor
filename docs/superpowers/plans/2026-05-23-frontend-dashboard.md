# Frontend Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Jinja2/HTMX dashboard at `/app/*` to the existing FastAPI container, providing overview stats, value bets, prediction form, player search, and system management with live log streaming.

**Architecture:** A new `api/frontend.py` router (prefix `/app/`) handles all HTML routes; Jinja2 templates extend a shared `base.html` sidebar layout; system long-running actions run in background threads with an in-memory job registry polled via HTMX every 2 s. Cookie-based auth guards all `/app/*` routes using a token derived from `DASHBOARD_PASSWORD`.

**Tech Stack:** FastAPI + Jinja2 3.1.4 + HTMX 1.9.12 + Alpine.js 3.x + Tailwind CSS CDN (arbitrary values build). No JS build step. Deployed to existing Railway container.

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `api/auth.py` | Cookie validation, session token, `require_auth` dependency |
| Create | `api/frontend.py` | All `/app/*` page + partial routes, job registry |
| Modify | `api/server.py` | Mount `/static`, include frontend router |
| Modify | `config.py` | Add `DASHBOARD_PASSWORD` |
| Modify | `requirements.txt` | Add jinja2, python-multipart |
| Create | `static/css/app.css` | Minimal custom styles (terminal font, scrollbar) |
| Create | `static/js/app.js` | Alpine.js helpers (empty stub) |
| Create | `templates/base.html` | Sidebar layout shell |
| Create | `templates/login.html` | Login form |
| Create | `templates/overview.html` | Overview page |
| Create | `templates/value_bets.html` | Value bets table |
| Create | `templates/predict.html` | Predict form |
| Create | `templates/players.html` | Player search |
| Create | `templates/system.html` | System management |
| Create | `templates/partials/predict_result.html` | HTMX predict result card |
| Create | `templates/partials/player_results.html` | HTMX player search results |
| Create | `templates/partials/job_status.html` | HTMX polling job log fragment |
| Create | `templates/partials/db_stats.html` | HTMX db stats fragment |
| Create | `tests/test_frontend.py` | Auth + frontend route tests |

---

## Task 1: Foundation — Dependencies, Config, Static Dirs

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py`
- Create: `static/css/app.css`
- Create: `static/js/app.js`

- [ ] **Step 1: Add jinja2 and python-multipart to requirements.txt**

Open `requirements.txt` and append two lines:

```
jinja2==3.1.4
python-multipart==0.0.9
```

- [ ] **Step 2: Add DASHBOARD_PASSWORD to config.py**

Current `config.py` ends with `PORT = int(os.getenv("PORT", 8000))`. Append:

```python
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
if not DASHBOARD_PASSWORD:
    logger.warning("DASHBOARD_PASSWORD not set — dashboard login disabled (any password accepted).")
```

- [ ] **Step 3: Create static directories and files**

```bash
mkdir -p static/css static/js templates/partials
```

Create `static/css/app.css`:

```css
/* Terminal-style monospace log output */
.terminal {
  font-family: 'SF Mono', 'Fira Code', 'Menlo', monospace;
  font-size: 11px;
  line-height: 1.5;
}

/* Custom scrollbar for dark theme */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f0f1a; }
::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a5e; }
```

Create `static/js/app.js`:

```js
// Alpine.js helpers — extend as needed
document.addEventListener('alpine:init', () => {
  // No global stores needed yet
});
```

- [ ] **Step 4: Install locally to verify**

```bash
cd /Users/gianlucatommasino/Desktop/AI/scommettitore
pip install jinja2==3.1.4 python-multipart==0.0.9 --quiet
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config.py static/
git commit -m "feat: add frontend deps, config, static dirs"
```

---

## Task 2: Auth — Cookie-Based Login

**Files:**
- Create: `api/auth.py`
- Modify: `config.py` (already done in Task 1)

- [ ] **Step 1: Write the failing auth tests**

Create `tests/test_frontend.py`:

```python
import pytest
from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app, follow_redirects=False)


def test_login_page_loads():
    resp = client.get("/app/login")
    assert resp.status_code == 200
    assert b"Login" in resp.content


def test_protected_route_redirects_to_login():
    resp = client.get("/app/overview")
    assert resp.status_code in (302, 303)
    assert "/app/login" in resp.headers["location"]


def test_login_wrong_password_shows_error():
    resp = client.post("/app/login", data={"password": "wrongpassword"})
    assert resp.status_code == 200
    assert b"Invalid password" in resp.content


def test_login_correct_password_redirects_and_sets_cookie(monkeypatch):
    monkeypatch.setattr("api.auth.DASHBOARD_PASSWORD", "testpass")
    resp = client.post("/app/login", data={"password": "testpass"})
    assert resp.status_code in (302, 303)
    assert "session_token" in resp.cookies


def test_logout_clears_cookie():
    resp = client.get("/app/logout")
    assert resp.status_code in (302, 303)
    # Cookie cleared (empty value or max-age=0)
    cookie_header = resp.headers.get("set-cookie", "")
    assert "session_token" in cookie_header
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/gianlucatommasino/Desktop/AI/scommettitore
pytest tests/test_frontend.py -v 2>&1 | head -40
```

Expected: multiple errors — `api.auth` not found, routes not registered.

- [ ] **Step 3: Create api/auth.py**

```python
import hashlib
import hmac

from fastapi import HTTPException, Request

from config import API_SECRET_KEY, DASHBOARD_PASSWORD


def _expected_token() -> str:
    """Derive a stable session token from DASHBOARD_PASSWORD + API_SECRET_KEY."""
    secret = f"{DASHBOARD_PASSWORD}:{API_SECRET_KEY}"
    return hashlib.sha256(secret.encode()).hexdigest()


def set_session_cookie(response, token: str) -> None:
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,  # 30 days
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie("session_token")


def check_password(password: str) -> bool:
    if not DASHBOARD_PASSWORD:
        return True  # No password set — allow all (dev mode)
    return hmac.compare_digest(password, DASHBOARD_PASSWORD)


def require_auth(request: Request) -> None:
    token = request.cookies.get("session_token")
    expected = _expected_token()
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=303,
            headers={"Location": "/app/login"},
            detail="Login required",
        )
```

- [ ] **Step 4: Create login/logout routes in api/frontend.py (auth section only)**

Create `api/frontend.py`:

```python
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
    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/login")
def login_submit(request: Request, password: str = Form(...)):
    if not check_password(password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid password"},
            status_code=401,
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
```

- [ ] **Step 5: Mount the router in api/server.py**

Open `api/server.py`. After the line `from fastapi import Depends, FastAPI, Header, HTTPException, Request` add:

```python
from fastapi.staticfiles import StaticFiles
from api.frontend import router as frontend_router
```

After `app = FastAPI(title="Tennis Predictor", version="1.0.0")` add:

```python
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(frontend_router)
```

Full updated top of `api/server.py`:

```python
import hmac
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.frontend import router as frontend_router
from config import API_SECRET_KEY
from data.db import EloRating, Player, Prediction, get_db
from models.predictor import predict as _predict
from reports.daily_report import generate_markdown, generate_report

START_TIME = time.time()


async def verify_api_key(x_api_key: str = Header(...)):
    if not API_SECRET_KEY or not hmac.compare_digest(x_api_key, API_SECRET_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key.")


app = FastAPI(title="Tennis Predictor", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(frontend_router)
```

- [ ] **Step 6: Create minimal login.html so tests can run**

Create `templates/login.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Login — Scommettitore</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-[#0f0f1a] flex items-center justify-center">
  <div class="bg-[#1a1a2e] rounded-lg p-8 w-80 border border-[#2a2a3e]">
    <div class="text-purple-500 font-bold text-lg mb-6 text-center">🎾 Scommettitore</div>

    {% if error %}
    <div class="bg-red-900/30 border border-red-700 text-red-400 text-sm rounded p-2 mb-4">
      {{ error }}
    </div>
    {% endif %}

    <form method="post" action="/app/login">
      <label class="block text-gray-400 text-sm mb-1">Password</label>
      <input
        type="password"
        name="password"
        autofocus
        class="w-full bg-[#0f0f1a] border border-[#2a2a3e] rounded px-3 py-2 text-white text-sm mb-4 focus:outline-none focus:border-purple-600"
        placeholder="Enter password"
      >
      <button
        type="submit"
        class="w-full bg-purple-600 hover:bg-purple-700 text-white rounded py-2 text-sm font-medium transition-colors"
      >
        Sign In
      </button>
    </form>
  </div>
</body>
</html>
```

- [ ] **Step 7: Run auth tests**

```bash
pytest tests/test_frontend.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 8: Commit**

```bash
git add api/auth.py api/frontend.py api/server.py templates/login.html tests/test_frontend.py
git commit -m "feat: cookie-based auth, login/logout routes"
```

---

## Task 3: Base Template + CSS

**Files:**
- Create: `templates/base.html`

- [ ] **Step 1: Create templates/base.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}Scommettitore{% endblock %}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js"></script>
  <script defer src="https://unpkg.com/alpinejs@3.13.3/dist/cdn.min.js"></script>
  <link rel="stylesheet" href="/static/css/app.css">
</head>
<body class="flex h-screen bg-[#0f0f1a] text-white overflow-hidden">

  <!-- Sidebar -->
  <aside class="w-40 bg-[#1a1a2e] border-r border-[#2a2a3e] flex flex-col p-3 flex-shrink-0">

    <div class="text-purple-500 font-bold text-sm mb-5 flex items-center gap-1.5">
      🎾 <span>Scommettitore</span>
    </div>

    <!-- TODAY -->
    <div class="text-[9px] text-gray-600 uppercase tracking-widest mb-1.5">Today</div>

    <a href="/app/overview"
       class="px-2 py-1.5 rounded text-[11px] mb-0.5 flex items-center gap-1
              {% if active == 'overview' %}bg-[#2d2d4e] text-white border-l-2 border-purple-600
              {% else %}text-gray-500 hover:text-gray-300{% endif %}">
      📊 Overview
    </a>

    <a href="/app/value-bets"
       class="px-2 py-1.5 rounded text-[11px] mb-0.5 flex justify-between items-center
              {% if active == 'value_bets' %}bg-[#2d2d4e] text-white border-l-2 border-purple-600
              {% else %}text-gray-500 hover:text-gray-300{% endif %}">
      <span>💰 Value Bets</span>
      {% if vb_count %}
      <span class="bg-purple-600 text-white text-[8px] rounded-full px-1.5">{{ vb_count }}</span>
      {% endif %}
    </a>

    <!-- TOOLS -->
    <div class="text-[9px] text-gray-600 uppercase tracking-widest mt-3 mb-1.5">Tools</div>

    <a href="/app/predict"
       class="px-2 py-1.5 rounded text-[11px] mb-0.5
              {% if active == 'predict' %}bg-[#2d2d4e] text-white border-l-2 border-purple-600
              {% else %}text-gray-500 hover:text-gray-300{% endif %}">
      🔮 Predict
    </a>

    <a href="/app/players"
       class="px-2 py-1.5 rounded text-[11px] mb-0.5
              {% if active == 'players' %}bg-[#2d2d4e] text-white border-l-2 border-purple-600
              {% else %}text-gray-500 hover:text-gray-300{% endif %}">
      👤 Players
    </a>

    <!-- SYSTEM (bottom) -->
    <div class="flex-1"></div>
    <a href="/app/system"
       class="px-2 py-1.5 rounded text-[11px] border-t border-[#2a2a3e] pt-2.5
              {% if active == 'system' %}bg-[#2d2d4e] text-white border-l-2 border-purple-600
              {% else %}text-gray-500 hover:text-gray-300{% endif %}">
      ⚙️ System
    </a>

  </aside>

  <!-- Main content -->
  <main class="flex-1 overflow-y-auto p-5">
    {% block content %}{% endblock %}
  </main>

</body>
</html>
```

- [ ] **Step 2: Add test for base template**

Add to `tests/test_frontend.py`:

```python
def _authed_client():
    """Returns a TestClient with a valid session cookie."""
    import hashlib
    from config import API_SECRET_KEY, DASHBOARD_PASSWORD
    token = hashlib.sha256(f"{DASHBOARD_PASSWORD}:{API_SECRET_KEY}".encode()).hexdigest()
    c = TestClient(app, follow_redirects=False)
    c.cookies.set("session_token", token)
    return c


def test_overview_page_loads():
    c = _authed_client()
    resp = c.get("/app/overview")
    assert resp.status_code == 200
    assert b"Overview" in resp.content
    assert b"Scommettitore" in resp.content
```

- [ ] **Step 3: Commit**

```bash
git add templates/base.html tests/test_frontend.py
git commit -m "feat: base template with sidebar layout"
```

---

## Task 4: Overview Page

**Files:**
- Create: `templates/overview.html`
- Modify: `api/frontend.py` (add overview route)

- [ ] **Step 1: Add overview route to api/frontend.py**

Add after the logout route:

```python
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
        "request": request,
        "active": "overview",
        "report": report,
        "top_bet": top_bet,
        "init_done": init_done,
        "init_step": init_step,
        "init_error": init_error,
        **_sidebar_context(db),
    }
    return templates.TemplateResponse("overview.html", ctx)
```

- [ ] **Step 2: Create templates/overview.html**

```html
{% extends "base.html" %}
{% block title %}Overview — Scommettitore{% endblock %}

{% block content %}
<div class="flex justify-between items-start mb-5">
  <div>
    <h1 class="text-white font-bold text-lg">Overview</h1>
    <p class="text-gray-600 text-xs">{{ report.date }}</p>
  </div>
  <div class="bg-[#1a1a2e] rounded px-2 py-1 text-xs
              {% if init_done %}text-emerald-400{% else %}text-amber-400{% endif %}">
    {% if init_done %}● Sistema online{% elif init_error %}● Errore init{% else %}⏳ {{ init_step }}{% endif %}
  </div>
</div>

<!-- Stat Cards -->
<div class="grid grid-cols-2 gap-3 mb-4">
  <div class="bg-[#1a1a2e] rounded-lg p-4">
    <div class="text-purple-400 text-2xl font-bold">{{ report.value_bets_count }}</div>
    <div class="text-gray-500 text-xs mt-1">Value Bets oggi</div>
  </div>
  <div class="bg-[#1a1a2e] rounded-lg p-4">
    {% if report.value_bets %}
      {% set avg_edge = (report.value_bets | map(attribute='edge_pct') | select | list | sum) / report.value_bets | length %}
      <div class="text-emerald-400 text-2xl font-bold">+{{ "%.1f"|format(avg_edge) }}%</div>
    {% else %}
      <div class="text-gray-500 text-2xl font-bold">—</div>
    {% endif %}
    <div class="text-gray-500 text-xs mt-1">Edge medio</div>
  </div>
</div>

<!-- Top Value Bet -->
{% if top_bet %}
<div class="bg-[#1a1a2e] rounded-lg p-4 mb-4 border-l-4 border-emerald-500">
  <div class="text-emerald-400 text-xs font-bold mb-1.5">🏆 TOP VALUE BET</div>
  <div class="text-white text-sm mb-1">
    {{ top_bet.player1 }} vs {{ top_bet.player2 }}
    <span class="text-gray-500">· {{ top_bet.surface }} · {{ top_bet.category }}</span>
  </div>
  <div class="flex gap-4 text-xs">
    <span class="text-amber-400">Edge +{{ "%.1f"|format(top_bet.edge_pct) }}%</span>
    <span class="text-gray-400">p({{ top_bet.value_bet }}) {{ "%.0f"|format(top_bet.p1_win_prob * 100) }}%</span>
  </div>
</div>
{% else %}
<div class="bg-[#1a1a2e] rounded-lg p-4 mb-4 border-l-4 border-gray-700">
  <div class="text-gray-500 text-sm">Nessuna value bet oggi</div>
</div>
{% endif %}

<!-- System Status -->
<div class="bg-[#1a1a2e] rounded-lg p-4">
  <div class="text-gray-500 text-[10px] uppercase tracking-widest mb-3">Stato Sistema</div>
  <div class="grid grid-cols-2 gap-y-2">
    <div class="flex justify-between pr-4">
      <span class="text-gray-400 text-xs">Model</span>
      <span class="text-xs {% if init_done %}text-emerald-400{% else %}text-amber-400{% endif %}">
        {% if init_done %}✅ loaded{% else %}⏳ {{ init_step }}{% endif %}
      </span>
    </div>
    <div class="flex justify-between">
      <span class="text-gray-400 text-xs">DB</span>
      <span class="text-emerald-400 text-xs">✅ ok</span>
    </div>
    {% if init_error %}
    <div class="col-span-2 mt-2">
      <details class="text-red-400 text-xs">
        <summary class="cursor-pointer">⚠️ Init error (click to expand)</summary>
        <pre class="mt-1 text-[10px] text-red-300 overflow-x-auto">{{ init_error }}</pre>
      </details>
    </div>
    {% endif %}
  </div>
</div>
{% endblock %}
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_frontend.py::test_overview_page_loads -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add api/frontend.py templates/overview.html
git commit -m "feat: overview page"
```

---

## Task 5: Value Bets Page

**Files:**
- Create: `templates/value_bets.html`
- Modify: `api/frontend.py`

- [ ] **Step 1: Add value-bets route to api/frontend.py**

```python
# ── Value Bets ───────────────────────────────────────────────────────────────

@router.get("/value-bets", response_class=HTMLResponse)
def value_bets_page(
    request: Request,
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    from reports.daily_report import generate_report
    report = generate_report(db)
    bets = sorted(
        report["value_bets"],
        key=lambda x: x.get("edge_pct") or 0,
        reverse=True,
    )
    ctx = {
        "request": request,
        "active": "value_bets",
        "bets": bets,
        **_sidebar_context(db),
    }
    return templates.TemplateResponse("value_bets.html", ctx)
```

- [ ] **Step 2: Create templates/value_bets.html**

```html
{% extends "base.html" %}
{% block title %}Value Bets — Scommettitore{% endblock %}

{% block content %}
<div class="flex items-center gap-3 mb-5">
  <h1 class="text-white font-bold text-lg">Value Bets</h1>
  {% if bets %}
  <span class="bg-purple-600 text-white text-xs rounded-full px-2 py-0.5">{{ bets | length }}</span>
  {% endif %}
</div>

{% if bets %}
<div class="bg-[#1a1a2e] rounded-lg overflow-hidden">
  <!-- Header -->
  <div class="grid grid-cols-[2fr_1fr_1fr_1fr] gap-0 bg-[#2d2d4e] px-4 py-2">
    <span class="text-gray-500 text-[10px] uppercase tracking-wider">Match</span>
    <span class="text-gray-500 text-[10px] uppercase tracking-wider">Superficie</span>
    <span class="text-gray-500 text-[10px] uppercase tracking-wider">p(win)</span>
    <span class="text-gray-500 text-[10px] uppercase tracking-wider">Edge</span>
  </div>
  <!-- Rows -->
  {% for bet in bets %}
  <div class="grid grid-cols-[2fr_1fr_1fr_1fr] gap-0 px-4 py-2.5 border-b border-[#2a2a3e] last:border-b-0 hover:bg-[#2d2d4e]/40 transition-colors">
    <div>
      <span class="text-white text-sm">{{ bet.player1 }} vs {{ bet.player2 }}</span>
      {% if bet.value_bet %}
      <span class="text-gray-500 text-xs ml-1">· {{ bet.value_bet }}</span>
      {% endif %}
    </div>
    <span class="text-sm
      {% if bet.surface == 'clay' %}text-amber-400
      {% elif bet.surface == 'grass' %}text-emerald-400
      {% elif bet.surface == 'hard' %}text-blue-400
      {% else %}text-gray-400{% endif %}">
      {{ bet.surface }}
    </span>
    <span class="text-gray-400 text-sm">{{ "%.0f"|format(bet.p1_win_prob * 100) }}%</span>
    <span class="font-bold text-sm
      {% if bet.edge_pct >= 10 %}text-emerald-400
      {% elif bet.edge_pct >= 5 %}text-amber-400
      {% else %}text-gray-400{% endif %}">
      +{{ "%.1f"|format(bet.edge_pct) }}%
    </span>
  </div>
  {% endfor %}
</div>
{% else %}
<div class="bg-[#1a1a2e] rounded-lg p-8 text-center">
  <div class="text-gray-500 text-sm">Nessuna value bet trovata oggi.</div>
  <div class="text-gray-600 text-xs mt-1">Lo scheduler aggiorna le odds ogni 6 ore.</div>
</div>
{% endif %}
{% endblock %}
```

- [ ] **Step 3: Add test**

Add to `tests/test_frontend.py`:

```python
def test_value_bets_page_loads():
    c = _authed_client()
    resp = c.get("/app/value-bets")
    assert resp.status_code == 200
    assert b"Value Bets" in resp.content
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_frontend.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add api/frontend.py templates/value_bets.html tests/test_frontend.py
git commit -m "feat: value bets page"
```

---

## Task 6: Predict Page + HTMX Result Fragment

**Files:**
- Create: `templates/predict.html`
- Create: `templates/partials/predict_result.html`
- Modify: `api/frontend.py`

- [ ] **Step 1: Add predict routes to api/frontend.py**

```python
# ── Predict ──────────────────────────────────────────────────────────────────

@router.get("/predict", response_class=HTMLResponse)
def predict_page(
    request: Request,
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    # Surface + category options
    surfaces = ["hard", "clay", "grass", "indoor"]
    categories = ["Slam", "Masters", "500", "250", "Finals"]
    ctx = {
        "request": request,
        "active": "predict",
        "surfaces": surfaces,
        "categories": categories,
        **_sidebar_context(db),
    }
    return templates.TemplateResponse("predict.html", ctx)


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
            total_p1 = db.query(Match).filter(
                Match.surface == surface,
                Match.player1_id == player1_id,
            ).count() + db.query(Match).filter(
                Match.surface == surface,
                Match.player2_id == player1_id,
            ).count()
            wins_p1 = db.query(Match).filter(
                Match.surface == surface,
                Match.winner_id == player1_id,
            ).count()
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
        "partials/predict_result.html",
        {"request": request, "result": result, "error": error},
    )
```

- [ ] **Step 2: Create templates/predict.html**

```html
{% extends "base.html" %}
{% block title %}Predict — Scommettitore{% endblock %}

{% block content %}
<h1 class="text-white font-bold text-lg mb-5">🔮 Predict Match</h1>

<div class="bg-[#1a1a2e] rounded-lg p-5 max-w-md">
  <form
    hx-post="/app/predict"
    hx-target="#predict-result"
    hx-swap="innerHTML"
    hx-indicator="#predict-spinner"
  >
    <div class="grid grid-cols-2 gap-3 mb-3">
      <div>
        <label class="block text-gray-400 text-xs mb-1">Player 1 ID</label>
        <input type="number" name="player1_id" required
          class="w-full bg-[#0f0f1a] border border-[#2a2a3e] rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-purple-600">
      </div>
      <div>
        <label class="block text-gray-400 text-xs mb-1">Player 2 ID</label>
        <input type="number" name="player2_id" required
          class="w-full bg-[#0f0f1a] border border-[#2a2a3e] rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-purple-600">
      </div>
    </div>

    <div class="grid grid-cols-2 gap-3 mb-4">
      <div>
        <label class="block text-gray-400 text-xs mb-1">Superficie</label>
        <select name="surface"
          class="w-full bg-[#0f0f1a] border border-[#2a2a3e] rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-purple-600">
          {% for s in surfaces %}
          <option value="{{ s }}">{{ s }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label class="block text-gray-400 text-xs mb-1">Categoria</label>
        <select name="category"
          class="w-full bg-[#0f0f1a] border border-[#2a2a3e] rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-purple-600">
          {% for c in categories %}
          <option value="{{ c }}">{{ c }}</option>
          {% endfor %}
        </select>
      </div>
    </div>

    <button type="submit"
      class="w-full bg-purple-600 hover:bg-purple-700 text-white rounded py-2 text-sm font-medium transition-colors">
      Calcola Predizione
    </button>

    <div id="predict-spinner" class="htmx-indicator text-center mt-3 text-gray-500 text-sm hidden">
      ⏳ Calcolando...
    </div>
  </form>
</div>

<div id="predict-result" class="mt-4 max-w-md"></div>

<style>
  .htmx-indicator { display: none; }
  .htmx-request .htmx-indicator { display: block !important; }
</style>
{% endblock %}
```

- [ ] **Step 3: Create templates/partials/predict_result.html**

```html
{% if error %}
<div class="bg-red-900/30 border border-red-700 text-red-400 text-sm rounded-lg p-4">
  ⚠️ {{ error }}
</div>
{% elif result %}
<div class="bg-[#1a1a2e] rounded-lg p-4 border-l-4 border-purple-600">
  <div class="text-gray-500 text-[10px] uppercase tracking-widest mb-3">Risultato</div>

  <div class="text-white text-sm font-medium mb-3">
    {{ result.p1_name }} vs {{ result.p2_name }}
    <span class="text-gray-500 font-normal">· {{ result.surface }} · {{ result.category }}</span>
  </div>

  <div class="grid grid-cols-2 gap-3 mb-3">
    <div class="bg-[#0f0f1a] rounded p-3 text-center">
      <div class="text-purple-400 text-xl font-bold">{{ result.p1_win_prob }}%</div>
      <div class="text-gray-500 text-xs mt-0.5">p({{ result.p1_name.split()[0] }})</div>
    </div>
    <div class="bg-[#0f0f1a] rounded p-3 text-center">
      <div class="text-blue-400 text-xl font-bold">{{ result.p2_win_prob }}%</div>
      <div class="text-gray-500 text-xs mt-0.5">p({{ result.p2_name.split()[0] }})</div>
    </div>
  </div>

  <div class="flex gap-4 text-xs">
    {% if result.elo_delta is not none %}
    <span class="text-gray-400">
      ELO delta:
      <span class="{% if result.elo_delta > 0 %}text-emerald-400{% else %}text-red-400{% endif %}">
        {{ "%+.0f"|format(result.elo_delta) }}
      </span>
    </span>
    {% endif %}
    {% if result.win_rate_p1 is not none %}
    <span class="text-gray-400">
      Win rate {{ result.surface }}:
      <span class="text-amber-400">{{ result.win_rate_p1 }}%</span>
    </span>
    {% endif %}
  </div>
</div>
{% endif %}
```

- [ ] **Step 4: Add test**

Add to `tests/test_frontend.py`:

```python
def test_predict_page_loads():
    c = _authed_client()
    resp = c.get("/app/predict")
    assert resp.status_code == 200
    assert b"Predict" in resp.content
    assert b"player1_id" in resp.content
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_frontend.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add api/frontend.py templates/predict.html templates/partials/predict_result.html tests/test_frontend.py
git commit -m "feat: predict page with HTMX result fragment"
```

---

## Task 7: Players Page + Live Search

**Files:**
- Create: `templates/players.html`
- Create: `templates/partials/player_results.html`
- Modify: `api/frontend.py`

- [ ] **Step 1: Add player routes to api/frontend.py**

```python
# ── Players ──────────────────────────────────────────────────────────────────

@router.get("/players", response_class=HTMLResponse)
def players_page(
    request: Request,
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    ctx = {
        "request": request,
        "active": "players",
        **_sidebar_context(db),
    }
    return templates.TemplateResponse("players.html", ctx)


@router.get("/players/search", response_class=HTMLResponse)
def players_search(
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
            .order_by(Player.current_ranking.asc().nullslast())
            .limit(20)
            .all()
        )
        # Attach ELO ratings to each player
        player_data = []
        for p in players:
            ratings = db.query(EloRating).filter_by(player_id=p.id).all()
            player_data.append({
                "id": p.id,
                "name": p.name,
                "nationality": p.nationality or "—",
                "ranking": p.current_ranking,
                "elo": {r.surface: round(r.rating, 0) for r in ratings},
            })
    else:
        player_data = []

    return templates.TemplateResponse(
        "partials/player_results.html",
        {"request": request, "players": player_data, "q": q},
    )
```

- [ ] **Step 2: Create templates/players.html**

```html
{% extends "base.html" %}
{% block title %}Players — Scommettitore{% endblock %}

{% block content %}
<h1 class="text-white font-bold text-lg mb-5">👤 Players</h1>

<div class="bg-[#1a1a2e] rounded-lg p-4 mb-4 max-w-xl">
  <input
    type="text"
    name="q"
    placeholder="Cerca giocatore (es. Sinner, Djokovic...)"
    hx-get="/app/players/search"
    hx-target="#player-results"
    hx-trigger="keyup changed delay:300ms"
    hx-swap="innerHTML"
    class="w-full bg-[#0f0f1a] border border-[#2a2a3e] rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-purple-600"
  >
</div>

<div id="player-results" class="max-w-xl">
  <p class="text-gray-600 text-sm">Digita almeno 2 caratteri per cercare.</p>
</div>
{% endblock %}
```

- [ ] **Step 3: Create templates/partials/player_results.html**

```html
{% if not players and q | length >= 2 %}
<p class="text-gray-500 text-sm">Nessun giocatore trovato per "{{ q }}".</p>
{% elif not q or q | length < 2 %}
<p class="text-gray-600 text-sm">Digita almeno 2 caratteri per cercare.</p>
{% else %}
{% for p in players %}
<div class="bg-[#1a1a2e] rounded-lg p-4 mb-2 hover:bg-[#2d2d4e]/60 transition-colors">
  <div class="flex justify-between items-start mb-2">
    <div>
      <span class="text-white font-medium text-sm">{{ p.name }}</span>
      <span class="text-gray-500 text-xs ml-2">{{ p.nationality }}</span>
    </div>
    <div class="text-right">
      {% if p.ranking %}
      <span class="text-gray-400 text-xs">ATP #{{ p.ranking }}</span>
      {% else %}
      <span class="text-gray-600 text-xs">unranked</span>
      {% endif %}
      <div class="text-gray-600 text-[10px]">id: {{ p.id }}</div>
    </div>
  </div>
  {% if p.elo %}
  <div class="flex gap-3 flex-wrap">
    {% for surface, rating in p.elo.items() %}
    <span class="text-[10px] text-gray-500">
      {{ surface }}:
      <span class="text-gray-300">{{ rating }}</span>
    </span>
    {% endfor %}
  </div>
  {% endif %}
</div>
{% endfor %}
{% endif %}
```

- [ ] **Step 4: Add test**

Add to `tests/test_frontend.py`:

```python
def test_players_page_loads():
    c = _authed_client()
    resp = c.get("/app/players")
    assert resp.status_code == 200
    assert b"Players" in resp.content


def test_players_search_empty():
    c = _authed_client()
    resp = c.get("/app/players/search?q=")
    assert resp.status_code == 200


def test_players_search_with_query():
    c = _authed_client()
    resp = c.get("/app/players/search?q=sin")
    assert resp.status_code == 200
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_frontend.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add api/frontend.py templates/players.html templates/partials/player_results.html tests/test_frontend.py
git commit -m "feat: players page with HTMX live search"
```

---

## Task 8: System Page — Actions, Job Polling, DB Stats

**Files:**
- Create: `templates/system.html`
- Create: `templates/partials/job_status.html`
- Create: `templates/partials/db_stats.html`
- Modify: `api/frontend.py`

- [ ] **Step 1: Add system routes to api/frontend.py**

The action functions run their work in a background thread and manually append log lines to the job's log list. `retrain` captures stdout from `train_model()` to surface the progress prints.

```python
# ── System ───────────────────────────────────────────────────────────────────

_ACTION_LABELS = {
    "scrape": "🔄 Scrape",
    "retrain": "🧠 Retrain",
    "refresh_odds": "💹 Refresh Odds",
    "fetch_news": "📰 Fetch News",
}


def _run_action(job_id: str, action: str) -> None:
    """Runs a system action in a background thread, writing log lines to _jobs[job_id]."""

    def log(msg: str):
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
            import sys
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
                output = buf.getvalue()
                for line in output.strip().splitlines():
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
        "request": request,
        "active": "system",
        "actions": _ACTION_LABELS,
        **_sidebar_context(db),
    }
    return templates.TemplateResponse("system.html", ctx)


@router.post("/system/run/{action}", response_class=HTMLResponse)
def system_run_action(
    action: str,
    request: Request,
    _: None = Depends(require_auth),
):
    if action not in _ACTION_LABELS:
        return HTMLResponse(
            f'<div class="text-red-400 text-sm">Unknown action: {action}</div>',
            status_code=400,
        )
    job_id = _new_job()
    t = threading.Thread(target=_run_action, args=(job_id, action), daemon=True)
    t.start()
    return templates.TemplateResponse(
        "partials/job_status.html",
        {"request": request, "job": _jobs[job_id], "job_id": job_id},
    )


@router.get("/system/job/{job_id}", response_class=HTMLResponse)
def system_job_status(
    job_id: str,
    request: Request,
    _: None = Depends(require_auth),
):
    job = _jobs.get(job_id)
    if not job:
        return HTMLResponse('<div class="text-red-400 text-sm">Job not found.</div>')
    return templates.TemplateResponse(
        "partials/job_status.html",
        {"request": request, "job": job, "job_id": job_id},
    )


@router.get("/system/stats", response_class=HTMLResponse)
def system_stats(
    request: Request,
    _: None = Depends(require_auth),
    db: Session = Depends(get_db),
):
    from data.db import Match, Player, Prediction
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
        "partials/db_stats.html",
        {"request": request, "stats": stats},
    )
```

- [ ] **Step 2: Create templates/partials/job_status.html**

This fragment is self-replacing: if running, it carries its own HTMX polling attributes; when done/failed, it drops them to stop polling.

```html
<div
  id="job-{{ job_id }}"
  {% if job.status == "running" %}
    hx-get="/app/system/job/{{ job_id }}"
    hx-trigger="every 2s"
    hx-swap="outerHTML"
    hx-target="#job-{{ job_id }}"
  {% endif %}
  class="bg-[#0a0a0f] rounded-lg p-3 terminal mt-3"
>
  <div class="flex items-center gap-2 mb-2">
    {% if job.status == "running" %}
      <span class="inline-block w-2 h-2 bg-amber-400 rounded-full animate-pulse"></span>
      <span class="text-amber-400 text-xs font-medium">Running...</span>
    {% elif job.status == "done" %}
      <span class="text-emerald-400 text-xs font-medium">✅ Done</span>
    {% else %}
      <span class="text-red-400 text-xs font-medium">❌ Failed</span>
    {% endif %}
    <span class="text-gray-600 text-[10px] ml-auto">{{ job.started_at }}</span>
  </div>

  <div class="max-h-48 overflow-y-auto space-y-0.5">
    {% for line in job.log %}
    <div class="text-[11px]
      {% if '[ERROR]' in line %}text-red-400
      {% elif '[OK]' in line or '✅' in line %}text-emerald-400
      {% elif '[WARN]' in line %}text-amber-400
      {% else %}text-gray-400{% endif %}">
      {{ line }}
    </div>
    {% endfor %}
    {% if not job.log %}
    <div class="text-gray-600 text-[11px]">Waiting for output...</div>
    {% endif %}
  </div>
</div>
```

- [ ] **Step 3: Create templates/partials/db_stats.html**

```html
<div class="flex justify-around bg-[#1a1a2e] rounded-lg p-3">
  <div class="text-center">
    <div class="text-purple-400 text-sm font-bold">{{ "{:,}".format(stats.matches) }}</div>
    <div class="text-gray-500 text-[10px]">matches</div>
  </div>
  <div class="text-center">
    <div class="text-emerald-400 text-sm font-bold">{{ "{:,}".format(stats.players) }}</div>
    <div class="text-gray-500 text-[10px]">players</div>
  </div>
  <div class="text-center">
    <div class="text-amber-400 text-sm font-bold">{{ stats.predictions }}</div>
    <div class="text-gray-500 text-[10px]">predictions</div>
  </div>
  <div class="text-center">
    <div class="text-blue-400 text-sm font-bold">{{ stats.jobs }}</div>
    <div class="text-gray-500 text-[10px]">jobs ✅</div>
  </div>
</div>
```

- [ ] **Step 4: Create templates/system.html**

```html
{% extends "base.html" %}
{% block title %}System — Scommettitore{% endblock %}

{% block content %}
<h1 class="text-white font-bold text-lg mb-5">⚙️ System</h1>

<!-- Quick Actions -->
<div class="bg-[#1a1a2e] rounded-lg p-4 mb-4">
  <div class="text-gray-500 text-[10px] uppercase tracking-widest mb-3">Quick Actions</div>
  <div id="action-area">
    <div class="flex gap-2 flex-wrap">
      {% for action_key, action_label in actions.items() %}
      <button
        hx-post="/app/system/run/{{ action_key }}"
        hx-target="#action-area"
        hx-swap="beforeend"
        hx-indicator="#spin-{{ action_key }}"
        class="bg-[#2d2d4e] hover:bg-purple-700 text-gray-300 hover:text-white text-xs px-3 py-2 rounded transition-colors flex items-center gap-1"
      >
        {{ action_label }}
        <span id="spin-{{ action_key }}" class="htmx-indicator text-gray-400">⏳</span>
      </button>
      {% endfor %}
    </div>
  </div>
</div>

<!-- DB Stats (loaded via HTMX on page load) -->
<div
  hx-get="/app/system/stats"
  hx-trigger="load"
  hx-swap="innerHTML"
  class="mb-4"
>
  <div class="bg-[#1a1a2e] rounded-lg p-3 text-center text-gray-600 text-sm">
    Loading stats...
  </div>
</div>

<style>
  .htmx-indicator { display: none; }
  .htmx-request .htmx-indicator { display: inline !important; }
</style>
{% endblock %}
```

- [ ] **Step 5: Add system tests**

Add to `tests/test_frontend.py`:

```python
def test_system_page_loads():
    c = _authed_client()
    resp = c.get("/app/system")
    assert resp.status_code == 200
    assert b"System" in resp.content
    assert b"Quick Actions" in resp.content


def test_system_run_unknown_action_returns_400():
    c = _authed_client()
    resp = c.post("/app/system/run/invalid_action")
    assert resp.status_code == 400


def test_system_run_action_returns_job_fragment():
    c = _authed_client()
    resp = c.post("/app/system/run/refresh_odds")
    assert resp.status_code == 200
    assert b"job-" in resp.content  # job ID in div id


def test_system_job_status_not_found():
    c = _authed_client()
    resp = c.get("/app/system/job/nonexistent")
    assert resp.status_code == 200
    assert b"not found" in resp.content.lower()


def test_system_stats_fragment():
    c = _authed_client()
    resp = c.get("/app/system/stats")
    assert resp.status_code == 200
    assert b"matches" in resp.content
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/test_frontend.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add api/frontend.py templates/system.html templates/partials/ tests/test_frontend.py
git commit -m "feat: system page with job polling, DB stats"
```

---

## Task 9: Wire Up, Deploy & Verify

**Files:**
- No new files — verify everything integrates

- [ ] **Step 1: Run the full test suite locally**

```bash
cd /Users/gianlucatommasino/Desktop/AI/scommettitore
pytest -v
```

Expected: all existing tests plus new frontend tests pass. Note: existing tests in `test_api.py`, `test_elo.py`, `test_features.py` must still pass.

- [ ] **Step 2: Smoke test the server locally**

```bash
# Start server with a test password
DASHBOARD_PASSWORD=test API_SECRET_KEY=testkey DATABASE_URL=sqlite:///test_tennis.db uvicorn api.server:app --port 8001 &
sleep 3
curl -s http://localhost:8001/app/login | grep -c "Scommettitore"
# Expected: 1
kill %1
```

- [ ] **Step 3: Set DASHBOARD_PASSWORD on Railway**

```bash
cd /Users/gianlucatommasino/Desktop/AI/scommettitore
# Ensure Railway link is active
railway link --project tennis-predictor
# Set the env var
railway variables set DASHBOARD_PASSWORD="<choose a strong password>"
```

Replace `<choose a strong password>` with the actual password before running.

- [ ] **Step 4: Deploy to Railway**

```bash
railway up
```

Watch logs until `Background init complete!` appears (model already trained, should be fast).

- [ ] **Step 5: Verify dashboard is live**

```bash
curl -s https://tennis-predictor-production-5726.up.railway.app/app/login | grep -c "Scommettitore"
# Expected: 1
```

Then open `https://tennis-predictor-production-5726.up.railway.app/app/login` in browser, log in with the password set in Step 3.

- [ ] **Step 6: Final commit with deploy tag**

```bash
git add -A
git commit -m "feat: frontend dashboard — overview, value bets, predict, players, system"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Overview page: stat cards, top value bet, system status
- ✅ Value Bets page: table sorted by edge%, color-coded
- ✅ Predict page: form with HTMX result card (p1/p2 prob, ELO delta, surface win rate)
- ✅ Players page: live HTMX search with ELO ratings
- ✅ System page: quick actions, job polling every 2s, terminal log, DB stats
- ✅ Auth: cookie-based, `DASHBOARD_PASSWORD`, login/logout
- ✅ `API_SECRET_KEY` never exposed to browser (all data fetched server-side)
- ✅ Sidebar badge for Value Bets count
- ✅ `/app/` redirects to `/app/overview`
- ✅ Static files mounted at `/static/`
- ✅ Dependencies added to requirements.txt

**Types/signatures consistent:**
- `_sidebar_context(db)` returns `{"vb_count": int}` — used in all page routes ✅
- `_new_job()` returns `str` (job_id), `_jobs[job_id]` has `status`, `log`, `started_at` ✅
- `job_status.html` reads `job.status`, `job.log`, `job.started_at` — matches dict keys ✅
- `predict_result.html` reads `result.p1_name`, `result.p2_name`, `result.p1_win_prob`, etc. — matches dict built in route ✅
- `player_results.html` reads `p.name`, `p.nationality`, `p.ranking`, `p.elo` — matches dict ✅

**No placeholders:** All steps have complete code. ✅

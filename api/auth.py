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

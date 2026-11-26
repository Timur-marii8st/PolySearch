"""Session authentication and explicit route-level RBAC for one organization."""
import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from contextvars import ContextVar
from urllib.parse import urlsplit
from starlette.middleware.trustedhost import TrustedHostMiddleware

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from .models import StrictModel

ROLES = ("admin", "researcher", "reviewer", "viewer")
COOKIE = "polysearch_session"
ITERATIONS = 600_000
SESSION_SECONDS = 8 * 3600
LOGIN_WINDOW = 300
LOGIN_LIMIT = 12
principal = ContextVar("polysearch_principal", default=None)
log = logging.getLogger("polysearch.access")


def auth_enabled():
    raw = os.environ.get("POLYSEARCH_AUTH", "local")
    if raw not in ("local", "required"):
        raise ValueError("POLYSEARCH_AUTH must be local or required")
    return raw == "required"


def actor_name(fallback):
    user = principal.get()
    return user["username"] if user else fallback


def username_key(username):
    value = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,63}", value):
        raise ValueError("Username must have 3-64 ASCII characters")
    return value


def password_hash(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    result = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS)
    return salt, result.hex()


def token_hash(token):
    return hashlib.sha256(token.encode("ascii")).hexdigest()


class Accounts:
    def __init__(self, db):
        self.db = db
        with db.transaction() as s:
            s.execute("CREATE TABLE IF NOT EXISTS accounts (username TEXT PRIMARY KEY, role TEXT NOT NULL, salt TEXT NOT NULL, password_hash TEXT NOT NULL, active INTEGER NOT NULL)")
            s.execute("CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, username TEXT NOT NULL REFERENCES accounts(username), expires BIGINT NOT NULL)")
            s.execute("CREATE INDEX IF NOT EXISTS sessions_user ON sessions(username)")
            s.execute("CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires)")
            s.execute("CREATE TABLE IF NOT EXISTS login_attempts (key TEXT PRIMARY KEY, window_start BIGINT NOT NULL, count INTEGER NOT NULL)")

    def provision(self, username, password, role, replace=False):
        username = username_key(username)
        if role not in ROLES or not 12 <= len(password) <= 256:
            raise ValueError("Choose a supported role and password of 12-256 characters")
        salt, hashed = password_hash(password)
        with self.db.transaction() as s:
            existing = s.execute("SELECT username FROM accounts WHERE username=?", (username,)).fetchone()
            if existing and not replace:
                raise ValueError("Account exists; explicit --replace is required")
            s.execute("INSERT INTO accounts VALUES (?, ?, ?, ?, 1) ON CONFLICT(username) DO UPDATE SET role=excluded.role, salt=excluded.salt, password_hash=excluded.password_hash, active=1",
                      (username, role, salt, hashed))
            s.execute("DELETE FROM sessions WHERE username=?", (username,))
        return username

    def disable(self, username):
        username = username_key(username)
        with self.db.transaction() as s:
            s.execute("UPDATE accounts SET active=0 WHERE username=?", (username,))
            s.execute("DELETE FROM sessions WHERE username=?", (username,))

    def login(self, username, password, peer):
        try:
            username = username_key(username)
        except ValueError:
            username = ""
        now = int(time.time())
        key = hashlib.sha256(peer.encode()).hexdigest()
        with self.db.transaction() as s:
            s.execute("DELETE FROM login_attempts WHERE window_start<?", (now - LOGIN_WINDOW,))
            row = s.execute("SELECT count FROM login_attempts WHERE key=?", (key,)).fetchone()
            if row and row[0] >= LOGIN_LIMIT:
                return None, 429
            s.execute("INSERT INTO login_attempts VALUES (?, ?, 1) ON CONFLICT(key) DO UPDATE SET count=login_attempts.count+1", (key, now))
            user = s.execute("SELECT username, role, salt, password_hash, active FROM accounts WHERE username=?", (username,)).fetchone()
            salt = user[2] if user else "00" * 16
            _, candidate = password_hash(password, salt)
            valid = hmac.compare_digest(candidate, user[3] if user else "00" * 32)
            if not user or not user[4] or not valid:
                return None, 401
            token = secrets.token_urlsafe(32)
            s.execute("DELETE FROM sessions WHERE expires<=?", (now,))
            s.execute("INSERT INTO sessions VALUES (?, ?, ?)", (token_hash(token), username, now + SESSION_SECONDS))
            return token, 200

    def identify(self, token):
        if not token or not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
            return None
        with self.db.transaction() as s:
            row = s.execute("SELECT a.username, a.role FROM accounts a JOIN sessions t ON a.username=t.username WHERE t.token_hash=? AND t.expires>? AND a.active=1",
                            (token_hash(token), int(time.time()))).fetchone()
        return {"username": row[0], "role": row[1]} if row else None

    def revoke(self, token):
        if token and re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
            with self.db.transaction() as s:
                s.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(token),))


class Login(StrictModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


def authorized(role, method, path):
    # Explicit allowlists; unknown mutations fail closed.
    if method in ("GET", "HEAD"):
        if path == "/exchange" or path == "/api/exchange/inbox" or re.fullmatch(r"/api/exchange/inbox/[^/]+", path) or re.fullmatch(r"/api/exchange/campaigns/[^/]+/orders/[^/]+", path):
            return role in ("admin", "researcher", "reviewer")
        if path == "/metrics":
            return role == "admin"
        if path in ("/", "/pilot", "/docs", "/openapi.json", "/redoc", "/api/auth/me", "/api/state", "/api/export.json", "/api/pilot/campaigns"):
            return True
        if re.fullmatch(r"/api/pilot/campaigns/[^/]+(?:/(?:report|export\.json|attempts\.csv|plans/[^/]+/replay))?", path):
            return True
        return False
    if method != "POST":
        return False
    if path == "/api/exchange/inbox":
        return role in ("admin", "researcher")
    if re.fullmatch(r"/api/exchange/inbox/[^/]+/review", path):
        return role in ("admin", "reviewer")
    if path == "/api/auth/logout":
        return True
    if path in ("/api/config", "/api/demo"):
        return role == "admin"
    if path in ("/api/recommend", "/api/results", "/api/simulate", "/api/import.csv", "/api/pilot/campaigns"):
        return role in ("admin", "researcher")
    if re.fullmatch(r"/api/pilot/campaigns/[^/]+/plans(?:/[^/]+/measurement)?", path):
        return role in ("admin", "researcher")
    if re.fullmatch(r"/api/pilot/campaigns/[^/]+/(?:activate|close|plans/[^/]+/(?:review|cancel))", path):
        return role in ("admin", "reviewer")
    return False


def install_security(app, db, enabled):
    accounts = Accounts(db) if enabled else None
    secure = os.environ.get("POLYSEARCH_COOKIE_SECURE", "1") == "1"
    origin = os.environ.get("POLYSEARCH_ORIGIN", "").rstrip("/")
    if enabled and not re.fullmatch(r"https?://[^/]+", origin):
        raise ValueError("POLYSEARCH_ORIGIN must be the exact browser origin in authenticated mode")
    if enabled and (not secure or origin.startswith("http://")) and not (
        not secure and re.fullmatch(r"http://(?:127\.0\.0\.1|localhost)(?::\d+)?", origin)
    ):
        raise ValueError("Insecure cookies/HTTP allowed only on explicit loopback origin")

    configured = os.environ.get("POLYSEARCH_TRUSTED_HOSTS")
    hosts = [x.strip() for x in configured.split(",") if x.strip()] if configured else ["localhost", "127.0.0.1", "[::1]"]
    if enabled and configured is None:
        hosts.append(urlsplit(origin).hostname)
    if not hosts or any(not h or "*" in h or "/" in h or "@" in h for h in hosts):
        raise ValueError("Explicit trusted host names required")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts, www_redirect=False)

    router = APIRouter(prefix="/api/auth", tags=["auth"])

    @router.get("/me")
    def me():
        return {"authenticated_mode": enabled, "user": principal.get()}

    @router.post("/login")
    def login(payload: Login, request: Request):
        if not accounts:
            raise HTTPException(409, "Authentication is disabled in local mode")
        token, status = accounts.login(payload.username, payload.password,
                                       request.client.host if request.client else "unknown")
        if token is None:
            return JSONResponse({"detail": "Too many login attempts" if status == 429 else "Invalid credentials"}, status_code=status)
        response = JSONResponse({"ok": True})
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
                            secure=secure, samesite="strict", path="/")
        return response

    @router.post("/logout")
    def logout(request: Request):
        if accounts:
            accounts.revoke(request.cookies.get(COOKIE))
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE, path="/", secure=secure, httponly=True, samesite="strict")
        return response

    app.include_router(router)

    @app.middleware("http")
    async def guard(request, call_next):
        path = request.url.path.rstrip("/") or "/"
        method = request.method
        # Public assets contain code only, no data. Health is intentionally public.
        public = path in ("/health", "/ready", "/login") or path.startswith("/static/")
        user = None
        if method not in ("GET", "HEAD", "OPTIONS"):
            allowed_origin = origin if enabled else str(request.base_url).rstrip("/")
            supplied = request.headers.get("Origin")
            if (supplied is not None and supplied != allowed_origin) or request.headers.get("Sec-Fetch-Site") == "cross-site":
                return JSONResponse({"detail": "Rejected cross-origin mutation"}, status_code=403,
                                    headers={"Cache-Control": "no-store"})
        if enabled and not public:
            if method not in ("GET", "HEAD"):
                if request.headers.get("X-PolySearch-Request") != "1" or request.headers.get("Origin") not in (None, origin):
                    return JSONResponse({"detail": "Rejected cross-origin or unmarked mutation"}, status_code=403)
                if request.headers.get("Sec-Fetch-Site") == "cross-site":
                    return JSONResponse({"detail": "Rejected cross-site mutation"}, status_code=403)
            if path != "/api/auth/login":
                user = await run_in_threadpool(accounts.identify, request.cookies.get(COOKIE))
                if not user:
                    if method == "GET" and path in ("/", "/pilot", "/exchange", "/docs", "/redoc"):
                        return RedirectResponse("/login", status_code=303)
                    return JSONResponse({"detail": "Authentication required"}, status_code=401)
                if not authorized(user["role"], method, path):
                    return JSONResponse({"detail": "Permission denied"}, status_code=403)
            elif method != "POST":
                return JSONResponse({"detail": "Method not allowed"}, status_code=405)
        token = principal.set(user)
        try:
            response = await call_next(request)
        finally:
            principal.reset(token)
        if not path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if path in ("/", "/pilot", "/exchange", "/login"):
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        if enabled and method == "POST":
            # No credentials, request bodies, tokens or experimental values.
            log.info("mutation actor=%s role=%s route=%s status=%d", user["username"] if user else "anonymous",
                     user["role"] if user else "-", path, response.status_code)
        return response

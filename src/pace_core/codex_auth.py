"""Credentials for the ChatGPT (Codex) subscription backend, from the Codex CLI's own login.

`codex login` stores an OAuth token pair in $CODEX_HOME/auth.json (default
~/.codex). This module reads that file and refreshes the access token only when
it is about to expire, so the CLI and every pace-core process share one login
and one refresh-token chain. A refresh rotates the refresh token, so it runs
under a file lock, re-reads the file first (another process may have refreshed
already) and replaces the file atomically. Tokens are never logged.
"""
from __future__ import annotations

import base64
import fcntl
import json
import os
import platform
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

API_BASE = "https://chatgpt.com/backend-api/codex"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # the Codex CLI's public OAuth client
CLIENT_VERSION = "0.162.1"
REFRESH_MARGIN_S = 300


class CodexAuthError(RuntimeError):
    """No usable Codex login: not signed in, or the refresh was refused."""


def auth_file() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "auth.json"


def signed_in() -> bool:
    try:
        return bool(_read(auth_file())["tokens"].get("refresh_token"))
    except (CodexAuthError, KeyError, TypeError):
        return False


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise CodexAuthError(f"no Codex login at {path}; run `codex login`") from None
    except json.JSONDecodeError as e:
        raise CodexAuthError(f"unreadable Codex login at {path}: {e.msg}") from None


def _expires_at(jwt: str) -> float:
    try:
        payload = jwt.split(".")[1]
        return float(json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"])
    except (IndexError, KeyError, ValueError):
        return 0.0  # unknown expiry: refresh


def _fresh(tokens: dict) -> bool:
    return _expires_at(tokens.get("access_token", "")) - time.time() > REFRESH_MARGIN_S


@contextmanager
def _locked(path: Path):
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _refresh(path: Path, auth: dict) -> dict:
    body = json.dumps({"client_id": CLIENT_ID, "grant_type": "refresh_token",
                       "refresh_token": auth["tokens"]["refresh_token"], "scope": "openid profile email"})
    request = urllib.request.Request(TOKEN_URL, data=body.encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            issued = json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise CodexAuthError(f"Codex token refresh refused (HTTP {e.code}); run `codex login` again") from None
    auth = {**auth, "tokens": {**auth["tokens"], **{k: issued[k] for k in ("access_token", "id_token", "refresh_token")
                                                     if issued.get(k)}},
            "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    temporary = tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".auth-", delete=False)
    with temporary:
        json.dump(auth, temporary, indent=2)
    os.chmod(temporary.name, 0o600)
    os.replace(temporary.name, path)
    return auth


def credentials() -> tuple[str, str]:
    """(access token, ChatGPT account id), refreshed first if it expires within REFRESH_MARGIN_S."""
    path = auth_file()
    auth = _read(path)
    if not _fresh(auth.get("tokens") or {}):
        with _locked(path):
            auth = _read(path)  # another process may have refreshed while we waited
            if not _fresh(auth.get("tokens") or {}):
                if not (auth.get("tokens") or {}).get("refresh_token"):
                    raise CodexAuthError(f"the Codex login at {path} has no refresh token; run `codex login`")
                auth = _refresh(path, auth)
    tokens = auth["tokens"]
    return tokens["access_token"], tokens.get("account_id") or ""


def headers(*, stream: bool = False) -> dict[str, str]:
    """Request headers for API_BASE, as the Codex CLI sends them."""
    access, account = credentials()
    out = {"Authorization": f"Bearer {access}", "originator": "codex_cli_rs", "version": CLIENT_VERSION,
           "User-Agent": f"codex_cli_rs/{CLIENT_VERSION} ({platform.system()}; {platform.machine()}) pace-core",
           "Content-Type": "application/json"}
    if account:
        out["ChatGPT-Account-ID"] = account
    if stream:
        out["Accept"] = "text/event-stream"
    return out

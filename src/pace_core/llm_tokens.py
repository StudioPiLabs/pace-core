"""LLM provider API token registry.

Persists multi-provider API keys at `~/.config/pai/llm_tokens.json`
(mode 600). Tokens are user-level — not per-film — so they live outside
any project mount. The studio's /api/llm/* endpoints write here;
llm_client.py reads as a fallback when no explicit api_key is passed.

Lookup order (first hit wins):
  1. explicit `api_key=` argument to llm_client functions
  2. llm_tokens.json (this module's `get(provider)`)
  3. environment variable (e.g. $ANTHROPIC_API_KEY)
  4. None  → caller raises SystemExit

Override via $LLM_TOKENS_FILE for tests/alternate stores.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "pai"
TOKENS_FILE = Path(os.environ.get(
    "LLM_TOKENS_FILE",
    str(_DEFAULT_CONFIG_DIR / "llm_tokens.json"),
))


# Provider catalogue. `env_fallback` is the shell var consulted if no
# token is stored. `test_model` is a cheap model used by the /api/llm/test
# round-trip; `client` matches llm_client.py's dispatch.
PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "env_fallback": "ANTHROPIC_API_KEY",
        "client":       "anthropic",
        "test_model":   "claude-haiku-4-5",
        "test_endpoint": "https://api.anthropic.com/v1/messages",
        "test_cost_per_1k_in":  0.0008,
        "test_cost_per_1k_out": 0.004,
    },
    "openai": {
        "env_fallback": "OPENAI_API_KEY",
        "client":       "openai",
        "test_model":   "gpt-4o-mini",
        "test_endpoint": "https://api.openai.com/v1/chat/completions",
        "test_cost_per_1k_in":  0.00015,
        "test_cost_per_1k_out": 0.0006,
    },
    "routerbase": {
        "env_fallback": "ROUTERBASE_API_KEY",
        "client":       "openai",  # OpenAI-compatible
        "test_model":   "gpt-4o-mini",
        "test_endpoint": "https://router.requesty.ai/v1/chat/completions",
        # Price = GLM-4.6 list price from ZAI (docs.z.ai/guides/overview/pricing):
        # $0.60 in / $2.20 out per 1M tokens (cached input $0.11/1M). ZAI is the
        # source of truth — NOT the reseller's discounted ($0.57/$2.09) or its
        # fabricated "$0.6706 original" anchor.
        "test_cost_per_1k_in":  0.0006,
        "test_cost_per_1k_out": 0.0022,
    },
}


# ── on-disk store ─────────────────────────────────────────────────────


def load() -> dict:
    """Read the tokens file. Returns {} if missing or malformed."""
    if not TOKENS_FILE.exists():
        return {}
    try:
        return json.loads(TOKENS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_FILE.write_text(json.dumps(data, indent=2))
    # Force restrictive permissions even if the file pre-existed with 644.
    os.chmod(TOKENS_FILE, stat.S_IRUSR | stat.S_IWUSR)


# ── public API ────────────────────────────────────────────────────────


class MissingAPIKey(RuntimeError):
    """No API key resolved for a provider.

    A normal ``Exception`` (NOT ``SystemExit``) so HTTP handlers catch it and
    return a clean error instead of killing the request with an empty body,
    and CLIs can convert it to a friendly message.
    """


def get(provider: str) -> str | None:
    """Look up a stored API key for `provider`. Falls through to env.

    This is the ONE place key resolution happens: the studio-managed
    ``llm_tokens.json`` store first, then the provider's ``env_fallback``
    shell var. Every LLM / image-gen caller resolves through here (directly,
    or via ``llm_client.call_model`` which delegates to it)."""
    if provider not in PROVIDERS:
        return None
    store = load()
    entry = store.get(provider) or {}
    key = entry.get("api_key")
    if key:
        return key
    env_var = PROVIDERS[provider]["env_fallback"]
    return os.environ.get(env_var)


def require(provider: str) -> str:
    """Like :func:`get`, but raise :class:`MissingAPIKey` when absent.

    Use at the point of an actual call so the failure is catchable and
    carries an actionable message."""
    key = get(provider)
    if key:
        return key
    env = PROVIDERS.get(provider, {}).get("env_fallback", f"{provider.upper()}_API_KEY")
    raise MissingAPIKey(
        f"no API key for provider={provider!r}. Set it in the studio "
        f"LLM Tokens tab (POST /api/llm/token) or export {env}."
    )


def set(provider: str, api_key: str, *, note: str | None = None) -> None:
    """Persist a token. Overwrites silently."""
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider: {provider!r} (known: {sorted(PROVIDERS)})")
    if not api_key:
        raise ValueError("api_key must be non-empty")
    store = load()
    store[provider] = {
        "api_key":  api_key,
        "set_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note":     note,
    }
    _save(store)


def delete(provider: str) -> bool:
    """Remove the stored token. Returns True if there was one."""
    store = load()
    if provider in store:
        del store[provider]
        _save(store)
        return True
    return False


def list_providers() -> list[dict]:
    """One row per provider — what's the catalogue + which have tokens."""
    store = load()
    out = []
    for name, cfg in PROVIDERS.items():
        entry = store.get(name) or {}
        env_set = bool(os.environ.get(cfg["env_fallback"]))
        out.append({
            "name":            name,
            "client":          cfg["client"],
            "env_fallback":    cfg["env_fallback"],
            "test_model":      cfg["test_model"],
            "has_stored_token": bool(entry.get("api_key")),
            "stored_set_at":   entry.get("set_at"),
            "stored_note":     entry.get("note"),
            "env_token_set":   env_set,
        })
    return out


def test_provider(provider: str, *, prompt: str = "Reply with the single word: pong") -> dict:
    """Round-trip a tiny call to verify the token works.
    Returns {ok, model, reply, cost_usd, latency_ms, error?}."""
    import time
    if provider not in PROVIDERS:
        return {"ok": False, "error": f"unknown provider {provider!r}"}
    cfg = PROVIDERS[provider]
    api_key = get(provider)
    if not api_key:
        return {"ok": False, "error": f"no token (stored or in ${cfg['env_fallback']})"}

    model_cfg = {
        "provider":         cfg["client"],
        "model_name":       cfg["test_model"],
        "endpoint":         cfg["test_endpoint"],
        "max_tokens":       12,
        "cost_per_1k_in":   cfg["test_cost_per_1k_in"],
        "cost_per_1k_out":  cfg["test_cost_per_1k_out"],
    }
    messages = [{"role": "user", "content": prompt}]
    from pace_core.llm_client import _call_anthropic, _call_openai_compatible  # noqa: E402

    t0 = time.time()
    try:
        if cfg["client"] == "anthropic":
            text, cost = _call_anthropic(model_cfg, messages, api_key)
        else:
            text, cost = _call_openai_compatible(model_cfg, messages, api_key)
        return {
            "ok":         True,
            "model":      cfg["test_model"],
            "reply":      text.strip(),
            "cost_usd":   cost,
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        return {
            "ok":         False,
            "model":      cfg["test_model"],
            "latency_ms": int((time.time() - t0) * 1000),
            "error":      f"{type(e).__name__}: {str(e)[:200]}",
        }


__all__ = ["TOKENS_FILE", "PROVIDERS", "load", "get", "set", "delete",
           "list_providers", "test_provider"]

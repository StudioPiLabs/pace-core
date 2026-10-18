"""llm_client maps registry entries onto LiteLLM, and Codex auth refreshes safely."""
from __future__ import annotations

import base64
import json
import time

import pytest

from pace_core import codex_auth, llm_client

ROUTERBASE = {"provider": "openai", "model_name": "openai/gpt-5.5", "api_key_env": "ROUTERBASE_API_KEY",
              "endpoint": "https://routerbase.com/v1/chat/completions"}


def test_registry_entries_become_litellm_requests():
    rb = llm_client._request(ROUTERBASE, [], "k")
    assert (rb["model"], rb["api_base"], rb["temperature"], rb["stream"]) == (
        "openai/openai/gpt-5.5", "https://routerbase.com/v1", 0.0, True)
    claude = llm_client._request({"provider": "anthropic", "model_name": "claude-opus-5",
                                  "endpoint": "https://api.anthropic.com/v1/messages"}, [], "k")
    assert (claude["model"], claude["api_base"], claude["max_tokens"]) == (
        "anthropic/claude-opus-5", "https://api.anthropic.com", 1024)
    other = llm_client._request({"provider": "x", "litellm_model": "gemini/g", "temperature": None}, [], None)
    assert other["model"] == "gemini/g" and "temperature" not in other and "api_key" not in other
    codex = llm_client._request({"provider": "codex", "model_name": "m", "reasoning_effort": "low"}, [], None)
    assert codex["model"] == "codex/m" and codex["metadata"] == {"codex": {"reasoning_effort": "low"}}


def test_chat_messages_become_responses_input():
    instructions, items = llm_client._responses_input([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]},
        {"role": "assistant", "content": "A cat."}])
    assert instructions == "Be brief."
    assert items[0]["content"][1] == {"type": "input_image", "image_url": "data:image/png;base64,AA"}
    assert items[1]["content"] == [{"type": "output_text", "text": "A cat."}]


def test_a_deadline_is_opt_in_and_says_what_timed_out(monkeypatch):
    assert llm_client._timeout_for(ROUTERBASE) is None
    assert llm_client._timeout_for({**ROUTERBASE, "timeout_s": 0}) is None
    litellm = llm_client._litellm()

    def slow(**kwargs):
        raise litellm.Timeout(message="read timed out", model="m", llm_provider="openai")
    monkeypatch.setattr(litellm, "completion", slow)
    with pytest.raises(RuntimeError, match="timed out after 90s"):
        llm_client.call_model({**ROUTERBASE, "timeout_s": 90}, [{"role": "user", "content": "hi"}], "k")


def _jwt(exp: float) -> str:
    claims = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"h.{claims}.s"


def test_codex_refreshes_only_near_expiry_and_keeps_the_rotated_token(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {
        "access_token": _jwt(time.time() + 3600), "refresh_token": "r1", "account_id": "acct"}}))
    assert codex_auth.credentials()[1] == "acct"            # fresh: file untouched
    calls = []

    class Reply:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"access_token": _jwt(time.time() + 7200), "refresh_token": "r2"}).encode()

    def token_endpoint(request, timeout):
        calls.append(json.loads(request.data)["refresh_token"])
        return Reply()
    monkeypatch.setattr(codex_auth.urllib.request, "urlopen", token_endpoint)
    auth = json.loads(path.read_text())
    auth["tokens"]["access_token"] = _jwt(time.time() + 60)
    path.write_text(json.dumps(auth))
    codex_auth.credentials()
    codex_auth.credentials()                                 # second call sees the refreshed file
    saved = json.loads(path.read_text())
    assert calls == ["r1"] and saved["tokens"]["refresh_token"] == "r2" and saved["auth_mode"] == "chatgpt"
    assert path.stat().st_mode & 0o777 == 0o600


def test_first_use_from_many_threads_imports_litellm_once():
    """A split fans its chunks out over threads; concurrent first imports used to deadlock."""
    import subprocess
    import sys
    code = ("import threading; from pace_core import llm_client as c; b = threading.Barrier(8); out = []\n"
            "def go():\n    b.wait(); out.append(c._litellm())\n"
            "ts = [threading.Thread(target=go) for _ in range(8)]; [t.start() for t in ts]; [t.join(60) for t in ts]\n"
            "assert len(out) == 8 and len({id(m) for m in out}) == 1, out")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-800:]

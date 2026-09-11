"""Thin LLM client used by every build_*.py phase.

Exposes two dispatch functions matching the model registry shape in
assets/models.json (`paths.MODELS_FILE`):

  _call_anthropic(model_cfg, messages, api_key) → (text, cost)
  _call_openai_compatible(model_cfg, messages, api_key) → (text, cost)

Plus a uniform helper that picks the right one based on `model_cfg["provider"]`:

  call_model(model_cfg, messages, api_key=None) → (text, cost)

This file replaces the dispatch helpers that used to live in agents.py
(now removed along with the review framework).
"""

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Optional

# RouterBase (routerbase.com) is Cloudflare-fronted — the default urllib UA gets
# a 403. Sending a browser-ish UA on every request fixes routerbase and is
# harmless for OpenAI/Anthropic/local. (Mirrors image_gen_backends.)
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 routerbase-sdk/0.1"

# How long to wait for a completion: by default, as long as it takes.
#
# This was a flat 120 s, which killed the one call that is not like the others.
# Splitting a screenplay is a SINGLE call carrying the whole script and asking
# for structured JSON covering every scene in it, so both halves are large and
# the generation runs for minutes on a feature-length source. It timed out and
# the run died with "Unexpected error: The read operation timed out", because
# socket timeouts arrive as TimeoutError and fell through to the catch-all.
#
# Raising the number would only move the cliff. Retrying already happens a
# layer up, per scene and with backoff, in the breakdown verifier and the
# verify pipeline -- and that is the layer that knows what a lost call cost and
# what is safe to repeat. A transport deadline underneath it can only discard
# work that has already been paid for, on a call the server may well be about
# to answer. So there is no deadline here unless one is asked for.
#
# The escape hatch is PAI_LLM_TIMEOUT_S, or `timeout_s` on a model in the
# registry: a hung connection with no bytes arriving would otherwise block its
# worker thread indefinitely, and setting either restores a bound.
_DEFAULT_TIMEOUT_S = (int(os.environ["PAI_LLM_TIMEOUT_S"])
                      if os.environ.get("PAI_LLM_TIMEOUT_S") else None)


def _timeout_for(model_cfg: dict) -> int | None:
    v = model_cfg.get("timeout_s", _DEFAULT_TIMEOUT_S)
    if v in (None, "", 0):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S


def _timeout_error(model_cfg: dict, seconds) -> RuntimeError:
    """Only reachable when a deadline was asked for; say which one, and that
    the call may have been billed anyway."""
    return RuntimeError(
        f"timed out after {seconds}s waiting for {model_cfg.get('model_name')} "
        f"at {model_cfg.get('endpoint')}. Nothing was returned, and the request "
        f"may still have been billed. This deadline is opt-in: it comes from "
        f"PAI_LLM_TIMEOUT_S or `timeout_s` on the model. Clearing both waits "
        f"for the call, which is the default, because retries live a layer up "
        f"where the cost of a lost call is known.")


def _resolve_api_key(provider: str, model_cfg: dict) -> Optional[str]:
    """Consult llm_tokens.py for a stored key; fall back to model_cfg's
    api_key_env. Importing here is lazy so this module stays usable
    when llm_tokens hasn't been initialized."""
    try:
        from pace_core import llm_tokens
        # Map model_cfg provider → tokens.PROVIDERS key. Most names match.
        # For routerbase: model_cfg says provider="openai" but model_cfg may
        # carry an explicit api_key_env=ROUTERBASE_API_KEY hint.
        cand = provider
        env = model_cfg.get("api_key_env", "")
        if env == "ROUTERBASE_API_KEY":
            cand = "routerbase"
        key = llm_tokens.get(cand)
        if key:
            return key
    except ImportError:
        pass
    # Last resort: direct env-var read (legacy path).
    env = model_cfg.get("api_key_env")
    if env:
        return os.environ.get(env)
    return None


def _estimate_cost(model_cfg: dict, usage: dict) -> float:
    in_k  = usage.get("prompt_tokens", 0)     / 1000.0
    out_k = usage.get("completion_tokens", 0) / 1000.0
    return (in_k  * model_cfg.get("cost_per_1k_in",  0.0) +
            out_k * model_cfg.get("cost_per_1k_out", 0.0))


def _read_sse(resp) -> dict:
    """Collect a streamed chat completion into the shape a whole one has.

    Callers downstream want one object with choices[0].message.content and a
    usage block, so the difference between streamed and not stays inside this
    module rather than spreading into every phase that makes a model call.
    """
    parts: list[str] = []
    usage: dict = {}
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue                       # keep-alive or comment frame
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                parts.append(piece)
    return {"choices": [{"message": {"content": "".join(parts)}}], "usage": usage}


def _call_openai_compatible(model_cfg: dict, messages: list, api_key: str) -> tuple[str, float]:
    """OpenAI-compatible chat completions. Used for OpenAI proper, vLLM, local Qwen, etc.

    If model_cfg includes a "response_format" key (e.g.
    {"type":"json_object"} for OpenAI JSON mode), it's forwarded
    verbatim into the request body. Callers can force JSON-only output
    without prose / markdown wrappers — useful for structured pipelines
    like the SCINE enrichment that expect strict JSON."""
    payload = {
        "model":    model_cfg["model_name"],
        "messages": messages,
    }
    # Temperature 0 is what this pipeline wants: a breakdown that changes
    # between runs of the same script is not a measurement. But some models
    # refuse the parameter outright -- the GPT-5.6 family returns
    # "does not support 0 with this model, only the default (1)" -- and a
    # hardcoded 0.0 made every call to them a 400. A model may set
    # "temperature": null in the registry to have it omitted; anything else is
    # sent as given, defaulting to 0.
    if "temperature" in model_cfg:
        if model_cfg["temperature"] is not None:
            payload["temperature"] = model_cfg["temperature"]
    else:
        payload["temperature"] = 0.0
    if "response_format" in model_cfg:
        payload["response_format"] = model_cfg["response_format"]
    # Stream unless a model opts out.
    #
    # This does NOT solve the 524 that long calls hit, and it was added
    # believing it would. Measured against this gateway, a 30s generation
    # delivered its first SSE frame at 28.1s: the frames are produced but
    # buffered, so the connection is idle for the whole generation and
    # Cloudflare's ~100s limit still applies. The remedy for a long call is to
    # make it a shorter one -- see the chunking in split_script.
    #
    # It stays on because it is free, it returns the same text and usage, and
    # it starts working the day the gateway stops buffering.
    stream = bool(model_cfg.get("stream", True))
    if stream:
        payload["stream"] = True
        # `stream_options` is OpenAI's way of asking for usage on a streamed
        # response, and it is not sent by default because the gateway this
        # pipeline actually uses rejects it -- "only allowed when 'stream' is
        # enabled", even with stream true -- and returns usage without being
        # asked. A provider that needs it can put it in the registry.
        if model_cfg.get("stream_options"):
            payload["stream_options"] = model_cfg["stream_options"]
    req = urllib.request.Request(
        model_cfg["endpoint"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent":    _USER_AGENT,
        },
    )
    _t = _timeout_for(model_cfg)
    try:
        with urllib.request.urlopen(req, timeout=_t) as resp:
            body = (_read_sse(resp) if stream else json.loads(resp.read()))
    except TimeoutError as e:
        raise _timeout_error(model_cfg, _t) from e
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        if e.code == 524:
            raise RuntimeError(
                f"HTTP 524 from {model_cfg.get('endpoint')}: Cloudflare gave up "
                f"waiting for the model. It fires at roughly 100s, it is the "
                f"gateway's limit and not ours, and streaming does NOT avoid it "
                f"here -- this gateway buffers, so the first SSE frame of a 30s "
                f"generation measured at 28.1s and the connection is idle until "
                f"then. The fix is a smaller request: send less per call. "
                f"Script splitting chunks for this reason (PAI_SPLIT_CHUNK_CHARS).")
        # Retry once, dropping whichever parameter the server named, instead
        # of failing a paid run over one field. Two are known to be refused by
        # some models: stream_options (usage reporting) and temperature.
        drop = None
        if stream and e.code in (400, 422) and "stream_options" in raw:
            drop = "stream_options"
        elif e.code in (400, 422) and "temperature" in raw:
            drop = "temperature"
        if drop:
            payload.pop(drop, None)
            req2 = urllib.request.Request(
                model_cfg["endpoint"], data=json.dumps(payload).encode("utf-8"),
                headers=dict(req.headers))
            with urllib.request.urlopen(req2, timeout=_t) as resp:
                body = _read_sse(resp)
        else:
            raise RuntimeError(f"HTTP {e.code} {e.reason}: {raw}")
    except urllib.error.URLError as e:
        # A read timeout arrives as TimeoutError and is caught above; a
        # CONNECT timeout arrives wrapped in URLError, and would otherwise
        # print as "URL error: timed out" with no hint of the cause.
        if isinstance(e.reason, TimeoutError):
            raise _timeout_error(model_cfg, _t) from e
        raise RuntimeError(f"URL error: {e.reason}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON decode error: {e.msg}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")
    text  = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    cost  = _estimate_cost(model_cfg, usage)
    return text, cost


def _call_anthropic(model_cfg: dict, messages: list, api_key: str) -> tuple[str, float]:
    """Anthropic /v1/messages. Splits system from user/assistant; converts content blocks."""
    sys_msgs = [m for m in messages if m["role"] == "system"]
    usr_msgs = [m for m in messages if m["role"] != "system"]
    payload = {
        "model":      model_cfg["model_name"],
        "max_tokens": model_cfg.get("max_tokens", 1024),
        "system":     "\n\n".join(m["content"] if isinstance(m["content"], str)
                                  else "" for m in sys_msgs),
        "messages":   _convert_messages_to_anthropic(usr_msgs),
    }
    req = urllib.request.Request(
        model_cfg["endpoint"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "User-Agent":        _USER_AGENT,
        },
    )
    _t = _timeout_for(model_cfg)
    try:
        with urllib.request.urlopen(req, timeout=_t) as resp:
            body = json.loads(resp.read())
    except TimeoutError as e:
        raise _timeout_error(model_cfg, _t) from e
    text  = body["content"][0]["text"]
    usage = body.get("usage", {})
    cost  = _estimate_cost(model_cfg, {
        "prompt_tokens":     usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    })
    return text, cost


def _convert_messages_to_anthropic(msgs: list) -> list:
    """Convert OpenAI-style content arrays to Anthropic content blocks (image_url → image)."""
    out = []
    for m in msgs:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        blocks = []
        for c in content:
            if c["type"] == "text":
                blocks.append({"type": "text", "text": c["text"]})
            elif c["type"] == "image_url":
                url = c["image_url"]["url"]
                if url.startswith("data:"):
                    mime, b64 = url.split(",", 1)
                    media_type = mime.split(";")[0].split(":")[1]
                    blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64}
                    })
        out.append({"role": m["role"], "content": blocks})
    return out


def call_model(model_cfg: dict, messages: list,
               api_key: Optional[str] = None) -> tuple[str, float]:
    """Single entry-point that picks the right dispatcher per provider.

    For local providers the api_key may be omitted (defaults to "EMPTY").
    For anthropic / openai, if api_key is None we look up
    `os.environ[model_cfg["api_key_env"]]` and SystemExit if absent.
    """
    provider = model_cfg["provider"]
    if provider == "local":
        return _call_openai_compatible(model_cfg, messages, api_key or "EMPTY")
    if api_key is None:
        # Look up via llm_tokens.py (consults studio_server-managed store +
        # env-var fallback in one place). model_cfg["api_key_env"] is the
        # legacy hint we map to a provider name.
        api_key = _resolve_api_key(provider, model_cfg)
        if not api_key:
            # Catchable (NOT SystemExit) so HTTP handlers return a clean error
            # instead of dying with an empty body, and CLIs can pretty-print it.
            from pace_core import llm_tokens
            env = model_cfg.get("api_key_env", "?")
            raise llm_tokens.MissingAPIKey(
                f"no API key for provider={provider}. Set it in the studio "
                f"LLM Tokens tab (POST /api/llm/token) or export {env}."
            )
    if provider == "anthropic":
        return _call_anthropic(model_cfg, messages, api_key)
    if provider == "openai":
        return _call_openai_compatible(model_cfg, messages, api_key)
    raise ValueError(f"provider '{provider}' not supported")


def strip_fences(raw: str) -> str:
    """Strip a leading/trailing ```json ... ``` fence, if the model added
    one despite being asked for JSON only. Every build_*.py phase that
    parses an LLM response needs this same cleanup."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    return raw

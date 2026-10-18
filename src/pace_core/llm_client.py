"""LLM client used by every breakdown phase: one call, any provider.

Requests go through LiteLLM (https://github.com/BerriAI/litellm), so a model
registry entry (assets/models.json, `paths.MODELS_FILE`) can name any provider
LiteLLM supports. Three registry shapes are understood:

  provider "anthropic"           Anthropic Messages API
  provider "openai" / "local"    any OpenAI-compatible endpoint (OpenAI,
                                 RouterBase, vLLM, a local Qwen server)
  provider "codex"               the ChatGPT subscription the Codex CLI is
                                 signed in with (pace_core.codex_auth)

and any other entry may set `litellm_model` (e.g. "gemini/gemini-2.5-pro")
to be passed to LiteLLM as given. Every caller uses

  call_model(model_cfg, messages, api_key=None) -> (text, cost_usd)

with OpenAI-style messages. Cost is priced from the registry's cost_per_1k_*,
not LiteLLM's price list, so a registry entry stays the one source of truth.
"""

import json
import os
import re
import threading
import urllib.error
import urllib.request
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


#: Model-registry key used for each role when the host names none. A host
#: whose registry uses other keys sets PACE_MODEL_<ROLE> (e.g.
#: PACE_MODEL_GENERATOR) or edits this dict before the importing module loads.
DEFAULT_MODELS = {
    "generator":  "claude-opus-5",    # script IR, beat panels, shot design
    "estimator":  "gpt-5.5",          # breakdown verifier; another vendor on purpose
    "writer":     "claude-sonnet-5",  # splitting, prompts, characters, LAMP
    "classifier": "gpt-4o-mini",      # cheap per-shot labelling
}


def default_model(role: str) -> str:
    """Registry key for `role`: $PACE_MODEL_<ROLE>, else DEFAULT_MODELS."""
    return os.environ.get(f"PACE_MODEL_{role.upper()}") or DEFAULT_MODELS[role]


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
    """A stored key from llm_tokens (RouterBase is recognised by its
    api_key_env hint), else the registry's api_key_env variable."""
    try:
        from pace_core import llm_tokens
        name = "routerbase" if model_cfg.get("api_key_env") == "ROUTERBASE_API_KEY" else provider
        key = llm_tokens.get(name)
        if key:
            return key
    except ImportError:
        pass
    env = model_cfg.get("api_key_env")
    return os.environ.get(env) if env else None


def _estimate_cost(model_cfg: dict, usage: dict) -> float:
    in_k  = usage.get("prompt_tokens", 0)     / 1000.0
    out_k = usage.get("completion_tokens", 0) / 1000.0
    return (in_k  * model_cfg.get("cost_per_1k_in",  0.0) +
            out_k * model_cfg.get("cost_per_1k_out", 0.0))


# ── ChatGPT subscription (Codex) ─────────────────────────────────────

def _responses_input(messages: list) -> tuple[str, list]:
    """OpenAI chat messages -> (instructions, Responses API input items)."""
    instructions, items = [], []
    for m in messages:
        content = m["content"]
        if m["role"] == "system":
            instructions.append(content if isinstance(content, str) else
                                "\n".join(c.get("text", "") for c in content if c.get("type") == "text"))
            continue
        kind = "output_text" if m["role"] == "assistant" else "input_text"
        blocks = ([{"type": kind, "text": content}] if isinstance(content, str) else
                  [{"type": kind, "text": c["text"]} if c["type"] == "text" else
                   {"type": "input_image", "image_url": c["image_url"]["url"]} for c in content])
        items.append({"type": "message", "role": m["role"], "content": blocks})
    return "\n\n".join(instructions), items


def _call_codex(model: str, messages: list, options: dict, timeout) -> tuple[str, dict]:
    """One Responses call on the subscription backend; returns (text, OpenAI-style usage).

    The backend only streams (store=false, stream=true are required), so the
    SSE is read to `response.completed` here and handed back whole.
    """
    from pace_core import codex_auth
    instructions, items = _responses_input(messages)
    payload = {"model": model, "instructions": instructions or "Follow the user's request exactly.",
               "input": items, "store": False, "stream": True}
    if options.get("reasoning_effort"):
        payload["reasoning"] = {"effort": options["reasoning_effort"]}
    request = urllib.request.Request(f"{codex_auth.API_BASE}/responses", data=json.dumps(payload).encode(),
                                     headers=codex_auth.headers(stream=True))
    parts, usage = [], {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:])
                except json.JSONDecodeError:
                    continue
                kind = event.get("type")
                if kind == "response.output_text.delta":
                    parts.append(event.get("delta", ""))
                elif kind == "response.completed":
                    usage = event["response"].get("usage") or {}
                    break
                elif kind in ("response.failed", "error"):
                    error = event.get("error") or (event.get("response") or {}).get("error") or event
                    raise RuntimeError(f"Codex backend error: {error}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Codex backend HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}") from None
    return "".join(parts), {"prompt_tokens": usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_tokens", 0)}


_litellm_module = None
# Callers fan out across threads (split chunks, per-scene verification); two
# threads importing LiteLLM at once deadlock in its circular imports.
_litellm_lock = threading.Lock()


def _litellm():
    """LiteLLM, imported on first use with the `codex` provider registered.

    Lazy so that importing this module (for default_model) stays cheap and
    works where LiteLLM is not installed, e.g. inside Blender.
    """
    if _litellm_module is None:
        with _litellm_lock:
            _load_litellm()
    return _litellm_module


def _load_litellm() -> None:
    global _litellm_module
    if _litellm_module is None:
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no price-list fetch at import
        import litellm
        from litellm import CustomLLM, ModelResponse, Usage

        class CodexSubscription(CustomLLM):
            def completion(self, model, messages, api_base, custom_prompt_dict, model_response: ModelResponse,
                           print_verbose, encoding, api_key, logging_obj, optional_params, acompletion=None,
                           litellm_params=None, logger_fn=None, headers={}, timeout=None, client=None):
                options = ((litellm_params or {}).get("metadata") or {}).get("codex") or {}
                text, usage = _call_codex(model, messages, options, timeout)
                model_response.choices[0].message.content = text
                model_response.model = model
                model_response.usage = Usage(prompt_tokens=usage["prompt_tokens"],
                                             completion_tokens=usage["completion_tokens"],
                                             total_tokens=usage["prompt_tokens"] + usage["completion_tokens"])
                return model_response

        litellm.custom_provider_map = [item for item in litellm.custom_provider_map if item["provider"] != "codex"]
        litellm.custom_provider_map.append({"provider": "codex", "custom_handler": CodexSubscription()})
        litellm.suppress_debug_info = True
        _litellm_module = litellm


# ── dispatch ──────────────────────────────────────────────────────────

def _api_base(endpoint: str | None, suffix: str) -> str | None:
    return endpoint[: -len(suffix)] if endpoint and endpoint.endswith(suffix) else endpoint


def _request(model_cfg: dict, messages: list, api_key: Optional[str]) -> dict:
    """LiteLLM completion() arguments for a registry entry."""
    provider = model_cfg["provider"]
    name = model_cfg.get("model_name")
    kwargs: dict = {"messages": messages, "num_retries": 0}  # retries live a layer up, per scene
    timeout = _timeout_for(model_cfg)
    if timeout:
        kwargs["timeout"] = timeout
    if provider == "codex":
        # No JSON mode: the backend demands the word "json" in the input, not the
        # instructions, and every caller strips fences and parses anyway.
        options = {k: model_cfg[k] for k in ("reasoning_effort",) if model_cfg.get(k)}
        return {**kwargs, "model": f"codex/{name or 'gpt-6.1-sol'}", "metadata": {"codex": options}}
    if model_cfg.get("litellm_model"):
        kwargs["model"] = model_cfg["litellm_model"]
    elif provider == "anthropic":
        kwargs["model"] = f"anthropic/{name}"
        kwargs["api_base"] = _api_base(model_cfg.get("endpoint"), "/v1/messages")
    elif provider in ("openai", "local"):
        kwargs["model"] = f"openai/{name}"
        kwargs["api_base"] = _api_base(model_cfg.get("endpoint"), "/chat/completions")
        kwargs["extra_headers"] = {"User-Agent": _USER_AGENT}
    else:
        raise ValueError(f"provider '{provider}' not supported; set litellm_model on the registry entry")
    if api_key:
        kwargs["api_key"] = api_key
    if "max_tokens" in model_cfg:
        kwargs["max_tokens"] = model_cfg["max_tokens"]
    elif provider == "anthropic":
        kwargs["max_tokens"] = 1024  # the Messages API requires one
    # Temperature 0 by default: a breakdown that changes between runs of the
    # same script is not a measurement. A model that refuses the parameter sets
    # "temperature": null in the registry to have it omitted.
    temperature = model_cfg.get("temperature", 0.0)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if "response_format" in model_cfg:
        kwargs["response_format"] = model_cfg["response_format"]
    # Streaming does not avoid Cloudflare's ~100 s 524 on a buffering gateway
    # (first frame of a 30 s generation measured at 28.1 s); shorter calls do,
    # see split_script's chunking. It stays on because it costs nothing.
    if model_cfg.get("stream", True):
        kwargs["stream"] = True
    return kwargs


def _complete(litellm, kwargs: dict):
    response = litellm.completion(**kwargs)
    if kwargs.get("stream"):
        response = litellm.stream_chunk_builder(list(response), messages=kwargs["messages"])
    return response


def call_model(model_cfg: dict, messages: list,
               api_key: Optional[str] = None) -> tuple[str, float]:
    """Complete `messages` with the registry entry `model_cfg`; returns (text, cost_usd).

    Keys: `api_key`, else llm_tokens / the entry's api_key_env (MissingAPIKey
    when neither has one). Local endpoints need none and Codex uses its login.
    """
    provider = model_cfg["provider"]
    if provider == "local":
        api_key = api_key or "EMPTY"
    elif provider != "codex" and not model_cfg.get("litellm_model") and api_key is None:
        api_key = _resolve_api_key(provider, model_cfg)
        if not api_key:
            from pace_core import llm_tokens
            raise llm_tokens.MissingAPIKey(
                f"no API key for provider={provider}. Set it in the studio LLM Tokens tab "
                f"(POST /api/llm/token) or export {model_cfg.get('api_key_env', '?')}.")
    litellm = _litellm()
    kwargs = _request(model_cfg, messages, api_key)
    try:
        try:
            response = _complete(litellm, kwargs)
        except litellm.BadRequestError as e:
            # Some models refuse temperature (GPT-5.6: "only the default (1)").
            if "temperature" not in kwargs or "temperature" not in str(e):
                raise
            kwargs.pop("temperature")
            response = _complete(litellm, kwargs)
    except litellm.Timeout as e:
        raise _timeout_error(model_cfg, kwargs.get("timeout")) from e
    except litellm.APIError as e:
        if getattr(e, "status_code", None) == 524:
            raise RuntimeError(
                f"HTTP 524 from {model_cfg.get('endpoint')}: Cloudflare gave up waiting for the "
                f"model (~100 s gateway limit; streaming does not avoid it). Send less per call; "
                f"script splitting chunks for this reason (PAI_SPLIT_CHUNK_CHARS).") from e
        raise RuntimeError(f"{type(e).__name__}: {e}") from e
    text = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
    return text, _estimate_cost(model_cfg, usage)


def strip_fences(raw: str) -> str:
    """Strip a leading/trailing ```json ... ``` fence, if the model added
    one despite being asked for JSON only. Every build_*.py phase that
    parses an LLM response needs this same cleanup."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    return raw

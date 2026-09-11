"""Long calls stream, because the ceiling was never ours.

Removing the client's own deadline did not fix a whole-screenplay split: it
failed again with HTTP 524. That is Cloudflare's status for "the origin held
the connection open without sending anything", and RouterBase is
Cloudflare-fronted. A single non-streaming call over a whole script spends
minutes generating before its first byte, so it hits the edge's ~100s limit
every time and no client setting can raise it.

Streaming was supposed to fix that and does not: measured against this
gateway, a 30s generation delivered its first SSE frame at 28.1s, so the
frames are produced but buffered and the connection is idle for the whole
generation. The 524 is avoided by making the call smaller instead -- see
`test_split_chunking.py`.

What is tested here is that streaming is still correct, because it stays on:
it costs nothing, returns the same text and usage, and begins to help the day
the gateway stops buffering. `_read_sse` is what keeps the difference between
a streamed and a whole response inside this module.
"""
from __future__ import annotations

import io
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pace_core import llm_client                                       # noqa: E402


def _sse(*frames: str) -> io.BytesIO:
    return io.BytesIO("".join(f"data: {f}\n\n" for f in frames).encode())


def _chunk(text=None, usage=None):
    d = {"choices": [{"delta": ({"content": text} if text is not None else {})}]}
    if usage:
        d["usage"] = usage
    return json.dumps(d)


def test_deltas_are_reassembled_in_order():
    body = llm_client._read_sse(_sse(_chunk("Hel"), _chunk("lo, "), _chunk("world"), "[DONE]"))
    assert body["choices"][0]["message"]["content"] == "Hello, world"


def test_usage_is_carried_off_the_last_frame():
    """Whether it is asked for or volunteered, usage has to survive the
    reassembly or every streamed call would price at zero."""
    u = {"prompt_tokens": 1200, "completion_tokens": 800}
    body = llm_client._read_sse(_sse(_chunk("x"), _chunk(usage=u), "[DONE]"))
    assert body["usage"] == u
    cfg = {"cost_per_1k_in": 0.005, "cost_per_1k_out": 0.025}
    assert llm_client._estimate_cost(cfg, body["usage"]) == pytest.approx(0.026)


def test_keep_alive_and_comment_frames_do_not_break_the_stream():
    """Gateways send blank frames and non-JSON comments to hold the connection
    open -- which is the very mechanism that stops the 524."""
    raw = io.BytesIO(b": ping\n\ndata: " + _chunk("a").encode() + b"\n\n"
                     b"data: not-json\n\ndata: " + _chunk("b").encode() + b"\n\n"
                     b"data: [DONE]\n\n")
    assert llm_client._read_sse(raw)["choices"][0]["message"]["content"] == "ab"


def test_a_stream_that_stops_early_returns_what_arrived():
    """No [DONE]: a truncated stream yields partial text rather than raising,
    so the caller's own JSON parse is what reports the truncation."""
    body = llm_client._read_sse(_sse(_chunk("half")))
    assert body["choices"][0]["message"]["content"] == "half"


def test_streaming_is_on_unless_a_model_opts_out(monkeypatch):
    sent = {}

    def _fake(req, timeout=None):
        sent.update(json.loads(req.data))
        class R:
            def __iter__(self):
                return iter([b"data: " + _chunk("ok").encode(), b"data: [DONE]"])
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return R()

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _fake)
    cfg = {"model_name": "m", "endpoint": "https://x/v1/chat/completions"}
    text, _ = llm_client._call_openai_compatible(cfg, [{"role": "user", "content": "hi"}], "k")
    assert text == "ok"
    assert sent["stream"] is True
    # stream_options is NOT sent by default: this gateway rejects it with
    # "only allowed when 'stream' is enabled" even when stream is true, and
    # returns usage without being asked. A provider that needs it opts in.
    assert "stream_options" not in sent

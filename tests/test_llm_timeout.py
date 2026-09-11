"""A slow model call says what timed out, and waits long enough not to.

Splitting a screenplay is one call carrying the whole script and asking for
structured JSON covering every scene in it. Both halves are large and the
generation takes minutes on a feature-length source, but every call in this
client waited a flat 120 seconds. The split died, and it died illegibly:
socket timeouts arrive as TimeoutError, which the openai-compatible path did
not name, so it fell through to the catch-all and surfaced as
"ValueError: Unexpected error: The read operation timed out" -- a message that
neither says what timed out nor what to do about it.
"""
from __future__ import annotations

import pathlib
import sys
import urllib.error

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pace_core import llm_client                                       # noqa: E402

CFG = {"model_name": "anthropic/claude-sonnet-5", "provider": "openai",
       "endpoint": "https://example.invalid/v1/chat/completions",
       "cost_per_1k_in": 0.002, "cost_per_1k_out": 0.01}


def test_there_is_no_deadline_by_default():
    """Retries live a layer up, per scene and with backoff, where the cost of a
    lost call is known. A transport deadline underneath that can only discard
    work already paid for."""
    assert llm_client._timeout_for(CFG) is None


def test_a_model_may_set_its_own():
    assert llm_client._timeout_for({**CFG, "timeout_s": 90}) == 90
    # a malformed value must not take the whole client down
    assert llm_client._timeout_for({**CFG, "timeout_s": "soon"}) == llm_client._DEFAULT_TIMEOUT_S
    # an explicit zero means "no deadline", not "give up immediately"
    assert llm_client._timeout_for({**CFG, "timeout_s": 0}) is None


@pytest.mark.parametrize("raised", [
    TimeoutError("The read operation timed out"),                    # read
    urllib.error.URLError(TimeoutError("timed out")),                # connect
])
def test_both_kinds_of_timeout_say_what_happened(monkeypatch, raised):
    def _boom(*a, **kw):
        raise raised
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _boom)
    with pytest.raises(RuntimeError) as e:
        llm_client._call_openai_compatible(CFG, [{"role": "user", "content": "hi"}], "k")
    msg = str(e.value)
    assert "timed out after" in msg
    assert "PAI_LLM_TIMEOUT_S" in msg, "the message must say where the deadline came from"
    assert CFG["model_name"] in msg, "and which model it was waiting on"
    assert "Unexpected error" not in msg


def test_an_ordinary_url_error_is_not_reported_as_a_timeout(monkeypatch):
    def _boom(*a, **kw):
        raise urllib.error.URLError("name resolution failed")
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _boom)
    with pytest.raises(RuntimeError) as e:
        llm_client._call_openai_compatible(CFG, [{"role": "user", "content": "hi"}], "k")
    assert "timed out after" not in str(e.value)

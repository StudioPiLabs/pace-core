"""The codex_cli provider shells out to `codex exec` and returns its last message."""

import os
import stat

import pytest

from pace_core import llm_client


def fake_codex(tmp_path, body):
    script = tmp_path / "codex"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_reply_comes_from_the_last_message_file(tmp_path):
    # Echo stdin into the -o file so the test sees exactly what codex was sent.
    binary = fake_codex(tmp_path, 'while [ "$1" != "-o" ]; do shift; done; out=$2; cat > "$out"\n')
    text, cost = llm_client.call_model(
        {"provider": "codex_cli", "binary": binary, "model_name": "gpt-x"},
        [{"role": "system", "content": "Be terse."}, {"role": "user", "content": [{"type": "text", "text": "Hi"}]}])
    assert cost == 0.0
    assert "[SYSTEM]\nBe terse." in text and "[USER]\nHi" in text and "Do not run commands" in text


def test_failure_and_non_text_content_raise(tmp_path):
    binary = fake_codex(tmp_path, "echo 'not logged in' >&2; exit 3\n")
    with pytest.raises(RuntimeError, match="exit 3.*not logged in"):
        llm_client.call_model({"provider": "codex_cli", "binary": binary}, [{"role": "user", "content": "x"}])
    with pytest.raises(ValueError, match="text content only"):
        llm_client.call_model({"provider": "codex_cli", "binary": binary},
                              [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])

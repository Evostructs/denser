"""Regression tests for Claude SDK request compatibility."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from denser.backends.claude import ClaudeBackend


@pytest.mark.parametrize("temperature", [0.0, 0.3, 0.7])
def test_sampling_setting_reaches_request_body(monkeypatch, temperature: float) -> None:
    """Exercise the installed SDK signature without sending a network request."""
    backend = ClaudeBackend(api_key="not-a-real-key", temperature=temperature)
    calls: list[dict[str, object]] = []

    def fake_post(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text="OK")])

    monkeypatch.setattr(backend._client, "post", fake_post)
    try:
        assert backend.complete(system="rules", user="request", max_tokens=128) == "OK"
        assert len(calls) == 1
        # The SDK merges extra_json into the JSON body before sending it.
        assert calls[0]["options"]["extra_json"] == {"temperature": temperature}
        assert calls[0]["body"]["model"] == "claude-opus-4-6"
        assert calls[0]["body"]["max_tokens"] == 128
        assert calls[0]["body"]["messages"] == [{"role": "user", "content": "request"}]
        assert calls[0]["body"]["system"] == [
            {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
        ]
    finally:
        backend._client.close()

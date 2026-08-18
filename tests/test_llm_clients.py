from __future__ import annotations

import io
import json
import socket
import urllib.error

import pytest

from synthetic_replication.llm_clients import _compute_retry_delay_seconds, _post_json


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_post_json_retries_retryable_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise urllib.error.HTTPError(
                request.full_url,
                529,
                "Overloaded",
                {"Retry-After": "1"},
                io.BytesIO(b'{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'),
            )
        return _FakeResponse({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", sleeps.append)

    payload = _post_json(
        url="https://example.com/test",
        payload={"hello": "world"},
        headers={"Authorization": "Bearer test"},
        timeout_seconds=30,
        max_retries=3,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=8.0,
        error_label="Test API",
    )

    assert payload == {"ok": True}
    assert attempts["count"] == 3
    assert sleeps == [1.0, 2.0]


def test_post_json_retries_timeout_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise socket.timeout("timed out")
        return _FakeResponse({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", sleeps.append)

    payload = _post_json(
        url="https://example.com/test",
        payload={"hello": "world"},
        headers={"Authorization": "Bearer test"},
        timeout_seconds=30,
        max_retries=2,
        initial_backoff_seconds=0.5,
        max_backoff_seconds=8.0,
        error_label="Test API",
    )

    assert payload == {"ok": True}
    assert attempts["count"] == 2
    assert sleeps == [0.5]


def test_post_json_retries_connection_reset_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConnectionResetError(54, "Connection reset by peer")
        return _FakeResponse({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", sleeps.append)

    payload = _post_json(
        url="https://example.com/test",
        payload={"hello": "world"},
        headers={"Authorization": "Bearer test"},
        timeout_seconds=30,
        max_retries=2,
        initial_backoff_seconds=0.5,
        max_backoff_seconds=8.0,
        error_label="Test API",
    )

    assert payload == {"ok": True}
    assert attempts["count"] == 2
    assert sleeps == [0.5]


def test_post_json_fails_immediately_on_non_retryable_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def fake_urlopen(request, timeout):  # noqa: ANN001
        attempts["count"] += 1
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"type":"error","error":{"type":"invalid_request_error","message":"Bad input"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Test API error 400 after 1 attempt"):
        _post_json(
            url="https://example.com/test",
            payload={"hello": "world"},
            headers={"Authorization": "Bearer test"},
            timeout_seconds=30,
            max_retries=3,
            initial_backoff_seconds=1.0,
            max_backoff_seconds=8.0,
            error_label="Test API",
        )

    assert attempts["count"] == 1


def test_compute_retry_delay_honors_cap_and_retry_after() -> None:
    assert _compute_retry_delay_seconds(
        attempt=4,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=5.0,
        retry_after=2.0,
    ) == 5.0

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import errno
import json
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504, 529}
RETRYABLE_SOCKET_ERRNOS = {errno.ECONNRESET, errno.ECONNABORTED, errno.ETIMEDOUT, errno.EPIPE}


def default_api_key_env(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return "OPENAI_API_KEY"
    if normalized == "anthropic":
        return "ANTHROPIC_API_KEY"
    raise ValueError(f"Unsupported provider: {provider}")


def _resolve_api_key(provider: str, api_key: str | None, api_key_env: str | None) -> str | None:
    if api_key:
        return api_key
    env_name = api_key_env or default_api_key_env(provider)
    return os.getenv(env_name)


def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
    max_retries: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
    error_label: str,
) -> dict[str, Any]:
    encoded_payload = json.dumps(payload).encode("utf-8")
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        request = urllib.request.Request(
            url=url,
            data=encoded_payload,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - exercised only with live API
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in RETRYABLE_HTTP_STATUS_CODES and attempt < total_attempts:
                time.sleep(
                    _compute_retry_delay_seconds(
                        attempt=attempt,
                        initial_backoff_seconds=initial_backoff_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                        retry_after=_parse_retry_after(exc.headers.get("Retry-After")),
                    )
                )
                continue
            raise RuntimeError(
                f"{error_label} error {exc.code} after {attempt} attempt(s): {body}"
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - exercised only with live API
            if _is_retryable_network_exception(exc.reason) and attempt < total_attempts:
                time.sleep(
                    _compute_retry_delay_seconds(
                        attempt=attempt,
                        initial_backoff_seconds=initial_backoff_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                    )
                )
                continue
            raise RuntimeError(f"{error_label} request failed: {exc.reason}") from exc
        except Exception as exc:  # pragma: no cover - exercised only with live API
            if _is_retryable_network_exception(exc) and attempt < total_attempts:
                time.sleep(
                    _compute_retry_delay_seconds(
                        attempt=attempt,
                        initial_backoff_seconds=initial_backoff_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                    )
                )
                continue
            if _is_timeout_like_reason(exc):
                raise RuntimeError(
                    f"{error_label} timed out after {attempt} attempt(s) with timeout_seconds={timeout_seconds}"
                ) from exc
            if _is_retryable_network_exception(exc):
                raise RuntimeError(f"{error_label} request failed after {attempt} attempt(s): {exc}") from exc
            raise
    raise RuntimeError(f"{error_label} failed unexpectedly without returning a response")


def _is_timeout_like_reason(reason: object) -> bool:
    return isinstance(reason, (TimeoutError, socket.timeout))


def _is_retryable_network_exception(reason: object) -> bool:
    if _is_timeout_like_reason(reason):
        return True
    if isinstance(
        reason,
        (
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            ssl.SSLEOFError,
            ssl.SSLZeroReturnError,
        ),
    ):
        return True
    if isinstance(reason, OSError) and getattr(reason, "errno", None) in RETRYABLE_SOCKET_ERRNOS:
        return True
    return False


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return max(0.0, float(stripped))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(stripped)
        except (TypeError, ValueError, IndexError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _compute_retry_delay_seconds(
    *,
    attempt: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
    retry_after: float | None = None,
) -> float:
    backoff = min(max_backoff_seconds, initial_backoff_seconds * (2 ** max(attempt - 1, 0)))
    if retry_after is None:
        return backoff
    return min(max_backoff_seconds, max(backoff, retry_after))


def extract_openai_output_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    collected: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                collected.append(text)
    return "\n".join(collected).strip()


def extract_openai_usage(payload: dict[str, Any]) -> dict[str, int | None]:
    usage = payload.get("usage", {})
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def summarize_openai_payload(payload: dict[str, Any]) -> dict[str, Any]:
    output_items = payload.get("output", [])
    item_types: list[str | None] = []
    content_types: list[str | None] = []
    for item in output_items:
        if isinstance(item, dict):
            item_types.append(item.get("type"))
            for content in item.get("content", []):
                if isinstance(content, dict):
                    content_types.append(content.get("type"))
    return {
        "response_id": payload.get("id"),
        "status": payload.get("status"),
        "incomplete_reason": payload.get("incomplete_details", {}).get("reason"),
        "output_item_types": item_types,
        "content_types": content_types,
    }


def extract_anthropic_output_text(payload: dict[str, Any]) -> str:
    collected: list[str] = []
    for block in payload.get("content", []):
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            collected.append(block["text"])
    return "\n".join(collected).strip()


def extract_anthropic_usage(payload: dict[str, Any]) -> dict[str, int | None]:
    usage = payload.get("usage", {})
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = None
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def summarize_anthropic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    block_types: list[str | None] = []
    text_lengths: list[int] = []
    for block in payload.get("content", []):
        if isinstance(block, dict):
            block_types.append(block.get("type"))
            text = block.get("text")
            if isinstance(text, str):
                text_lengths.append(len(text.strip()))
    return {
        "response_id": payload.get("id"),
        "role": payload.get("role"),
        "stop_reason": payload.get("stop_reason"),
        "stop_sequence": payload.get("stop_sequence"),
        "content_block_types": block_types,
        "text_block_lengths": text_lengths,
    }


class OpenAIResponsesClient:
    provider = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_key_env: str | None = None,
        timeout_seconds: int = 90,
        max_retries: int = 4,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 12.0,
    ) -> None:
        self.api_key = _resolve_api_key(self.provider, api_key, api_key_env)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def create_json_response(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        assistant_prefill: str | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(f"{default_api_key_env(self.provider)} is not configured")
        payload = {
            "model": model,
            "store": False,
            "max_output_tokens": max_output_tokens,
            "text": {
                "format": {
                    "type": "json_object",
                }
            },
            "input": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }
        return _post_json(
            url="https://api.openai.com/v1/responses",
            payload=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            initial_backoff_seconds=self.initial_backoff_seconds,
            max_backoff_seconds=self.max_backoff_seconds,
            error_label="OpenAI Responses API",
        )

    @staticmethod
    def extract_output_text(payload: dict[str, Any]) -> str:
        return extract_openai_output_text(payload)

    @staticmethod
    def extract_usage(payload: dict[str, Any]) -> dict[str, int | None]:
        return extract_openai_usage(payload)

    @staticmethod
    def describe_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return summarize_openai_payload(payload)


class AnthropicMessagesClient:
    provider = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_key_env: str | None = None,
        timeout_seconds: int = 90,
        max_retries: int = 4,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 12.0,
    ) -> None:
        self.api_key = _resolve_api_key(self.provider, api_key, api_key_env)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def create_json_response(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        assistant_prefill: str | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(f"{default_api_key_env(self.provider)} is not configured")
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
        if assistant_prefill:
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_prefill,
                }
            )
        payload = {
            "model": model,
            "max_tokens": max_output_tokens,
            "temperature": 0,
            "system": system_prompt,
            "messages": messages,
        }
        return _post_json(
            url="https://api.anthropic.com/v1/messages",
            payload=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            initial_backoff_seconds=self.initial_backoff_seconds,
            max_backoff_seconds=self.max_backoff_seconds,
            error_label="Anthropic Messages API",
        )

    @staticmethod
    def extract_output_text(payload: dict[str, Any]) -> str:
        return extract_anthropic_output_text(payload)

    @staticmethod
    def extract_usage(payload: dict[str, Any]) -> dict[str, int | None]:
        return extract_anthropic_usage(payload)

    @staticmethod
    def describe_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return summarize_anthropic_payload(payload)

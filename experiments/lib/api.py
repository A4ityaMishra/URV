"""
One OpenRouter call path, shared by the worker runner and the judges.

Fixes the three accounting bugs found in the audit of scripts/OpenRouter.py:

  1. Retries. Network errors, 429, and 5xx now retry with exponential backoff
     and jitter. 401/403 (auth) and other 4xx (malformed request, unknown
     model) do NOT retry -- retrying those just burns wall-clock and, for
     auth, would hammer a key that is already known bad.
  2. Cost across ALL attempts. A call that returns HTTP 200 with empty content
     is still billed. The old code discarded that spend entirely (it set
     credits_used = 0.0 on any exception), so reported cost was an undercount.
     Cost is now accumulated on every attempt that came back with a usage
     block, and is attached to the exception when the call ultimately fails.
  3. finish_reason / provider recorded on EVERY row, success or failure --
     that is the field that actually explains an empty completion.

The key is read from the environment at call time and never logged. Error
strings here are built from status codes and response bodies only.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / "scripts" / ".env", override=True)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
NO_RETRY_STATUS = {400, 401, 402, 403, 404, 422}

MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "4"))
BASE_BACKOFF = float(os.environ.get("BASE_BACKOFF", "2.0"))


class CallError(RuntimeError):
    """Failure that still knows what it cost and why it stopped."""

    def __init__(self, message: str, cost: float = 0.0, meta: dict | None = None,
                 retryable: bool = False):
        super().__init__(message)
        self.cost = cost
        self.meta = meta or {}
        self.retryable = retryable


def _key() -> str:
    k = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not k:
        raise CallError("No API key in environment (scripts/.env)")
    return k


def call(model: str, messages: list[dict], temperature: float,
         timeout: int = 120, max_tokens: int | None = None) -> tuple[str, float, dict]:
    """One completion. Returns (content, total_cost_across_attempts, meta).

    `messages` is used as given -- callers build a fresh list per trial, which
    is what keeps trials stateless and concurrent dispatch safe.

    Raises CallError (carrying .cost) if every attempt fails.
    """
    payload: dict = {
        "model": model,
        "messages": messages,
        # Explicit on every single call. Left unset, the provider default
        # applies and repeat runs collapse toward identical text, which would
        # make the across-run standard deviations meaningless.
        "temperature": temperature,
        "usage": {"include": True},
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    cost = 0.0
    meta: dict = {"attempts": 0, "finish_reason": None, "provider": None,
                  "temperature": temperature, "model_requested": model}
    last = "no attempt made"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        meta["attempts"] = attempt
        try:
            r = requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {_key()}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.RequestException as e:
            last = f"network error: {type(e).__name__}: {e}"
            if attempt == MAX_ATTEMPTS:
                break
            _sleep(attempt)
            continue

        if r.status_code in NO_RETRY_STATUS:
            # Auth and malformed-request failures are permanent for this call.
            raise CallError(f"HTTP {r.status_code} (not retryable): {r.text[:300]}",
                            cost, meta, retryable=False)

        if r.status_code != 200:
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in RETRY_STATUS or attempt == MAX_ATTEMPTS:
                break
            _sleep(attempt, r.headers.get("Retry-After"))
            continue

        try:
            data = r.json()
        except ValueError as e:
            last = f"HTTP 200 but body is not JSON ({e}): {r.text[:300]}"
            if attempt == MAX_ATTEMPTS:
                break
            _sleep(attempt)
            continue

        # Bill first, judge the payload second: a 200 is charged whether or not
        # it carried usable content.
        cost += float((data.get("usage") or {}).get("cost", 0.0) or 0.0)
        meta["provider"] = data.get("provider")
        meta["model_served"] = data.get("model")

        choices = data.get("choices") or []
        if not choices:
            last = f"HTTP 200 with no choices: {json.dumps(data)[:300]}"
            if attempt == MAX_ATTEMPTS:
                break
            _sleep(attempt)
            continue

        choice = choices[0]
        meta["finish_reason"] = choice.get("finish_reason")
        content = (choice.get("message") or {}).get("content")

        if not content or not str(content).strip():
            # Never a silent success. This is the failure mode that used to be
            # written to disk as worker_response=null with error=null.
            last = (f"empty/whitespace content "
                    f"(finish_reason={meta['finish_reason']!r}, "
                    f"provider={meta['provider']!r})")
            if attempt == MAX_ATTEMPTS:
                break
            _sleep(attempt)
            continue

        return str(content), cost, meta

    raise CallError(f"failed after {meta['attempts']} attempt(s): {last}",
                    cost, meta, retryable=True)


def _sleep(attempt: int, retry_after: str | None = None) -> None:
    if retry_after:
        try:
            time.sleep(min(float(retry_after), 30.0))
            return
        except ValueError:
            pass
    time.sleep(min(BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 1.0), 30.0))

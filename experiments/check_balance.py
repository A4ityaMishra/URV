"""
Prints ONLY the remaining OpenRouter balance. Never prints, logs, or returns
the API key. Run before/after each major phase (brief section 1).

    python experiments/check_balance.py
"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "scripts" / ".env", override=True)


def _key() -> str:
    k = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not k:
        raise SystemExit("No API key found in scripts/.env")
    return k


def fetch_balance() -> dict:
    """Returns {'total_credits','total_usage','remaining'} in USD."""
    r = requests.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {_key()}"},
        timeout=30,
    )
    if r.status_code != 200:
        # Redact defensively: an error body should never echo the key, but the
        # key is never interpolated into anything we print regardless.
        raise SystemExit(f"Balance check failed: HTTP {r.status_code}")
    d = r.json().get("data", {})
    granted = float(d.get("total_credits", 0.0) or 0.0)
    used = float(d.get("total_usage", 0.0) or 0.0)
    return {"total_credits": granted, "total_usage": used, "remaining": granted - used}


if __name__ == "__main__":
    b = fetch_balance()
    print(f"granted   ${b['total_credits']:.4f}")
    print(f"used      ${b['total_usage']:.4f}")
    print(f"REMAINING ${b['remaining']:.4f}")

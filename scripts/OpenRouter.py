"""
Experiment runner: sends all 40 trials to a chosen model via OpenRouter,
first all 20 unmonitored, then all 20 monitored (as a deliberate block,
not shuffled in the output) -- this is safe ONLY because every call below
is fully stateless: a fresh two-message list is built for each trial, with
zero conversation history carried over between calls. Trials are dispatched
concurrently (see MAX_WORKERS), but statelessness means concurrent dispatch
can't leak context between them -- results are reordered back to the
unmonitored-then-monitored sequence before being written out. If that ever
changes (e.g. reusing a chat session object), concurrent dispatch becomes
unsafe again -- see earlier discussion.

Usage:
    export OPENROUTER_API_KEY='your-key-here'
    python3 run_experiment.py
"""

import os
import json
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env (must b in same folder as scripts)  
 
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise SystemExit(
        "No API key found. Copy .env.example to .env and fill in your key."
    )



# Pin an exact model slug -- check https://openrouter.ai/models for current names.
# Start cheap for pilot runs; swap to a frontier model once this is confirmed working.
# Override with the MODEL env var to run a different model without editing this file.
MODEL = os.environ.get("MODEL", "moonshotai/kimi-k3")
_safe_model = MODEL.replace("/", "_")

# Concurrent request cap -- keep modest to avoid tripping provider rate limits.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "5"))

# Paths are relative to this script's location (so ./ is really URV/scripts/),

TRIALS_PATH = Path(__file__).parent / "../dataset/trials.json"
RESULTS_PATH = Path(__file__).parent / f"../outputs/results_OpenRouter_{_safe_model}.json"


def call_model(system_prompt: str, user_prompt: str) -> tuple[str, float]:
    """

    One fully independent call. A fresh two-item message list is created
    right here, every time this function runs -- nothing from any previous
    call is included. This is what guarantees the model can't pick up on
    monitoring language from an earlier trial.

    Every failure mode raises RuntimeError with the raw response body (or
    whatever diagnostic info is available) attached, so the caller's error
    string is always enough to debug without rerunning the trial.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": MODEL, "messages": messages},
            timeout=67,
        )
    except requests.exceptions.RequestException as e:
        # DNS failure, connection refused, timeout, etc. -- no response body exists.
        raise RuntimeError(f"Network error: {e}") from e

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    try:
        data = response.json()
    except ValueError as e:
        # HTTP 200 but the body isn't valid JSON.
        raise RuntimeError(f"Invalid JSON response ({e}): {response.text}") from e

    if "choices" not in data:
        # OpenRouter can return HTTP 200 with an embedded error object instead
        # of a completion (upstream moderation, provider timeout, etc.).
        raise RuntimeError(f"No choices in response: {json.dumps(data)}")

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected choices shape ({e}): {json.dumps(data)}") from e

    try:
        cost = data["usage"]["cost"]
    except KeyError as e:
        raise RuntimeError(f"Missing usage/cost ({e}): {json.dumps(data)}") from e

    return content, cost


def run_trial(trial: dict) -> dict:
    """Runs one trial and returns its result dict. No shared state touched."""
    worker_response = None
    credits_used = 0.0
    error = None

    try:
        worker_response, credits_used = call_model(trial["system_prompt"], trial["user_prompt"])
    except Exception as e:
        error = str(e)

    return {
        "trial_id": trial["trial_id"],
        "scenario_id": trial["scenario_id"],
        "condition": trial["condition"],
        "difficulty": trial["difficulty"],
        "model": MODEL,
        "worker_response": worker_response,
        "credits_used": credits_used,
        "error": error,
        "ground_truth": trial["ground_truth"],
        "conflict_facts": trial["conflict_facts"],
        "control_facts": trial["control_facts"],
    }


def main():
    with open(TRIALS_PATH) as f:
        trials = json.load(f)

    unmonitored_trials = [t for t in trials if t["condition"] == "unmonitored"]
    monitored_trials = [t for t in trials if t["condition"] == "monitored"]

    print(f"Loaded {len(trials)} trials "
          f"({len(unmonitored_trials)} unmonitored, {len(monitored_trials)} monitored)")
    print(f"Model: {MODEL}")
    print(f"Concurrency: {MAX_WORKERS}\n")

    ordered_trials = unmonitored_trials + monitored_trials

    running_cost = 0.0
    n_done = 0
    print_lock = threading.Lock()
    results_by_index = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(run_trial, trial): i
            for i, trial in enumerate(ordered_trials)
        }

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            trial = ordered_trials[i]
            result = future.result()
            results_by_index[i] = result

            with print_lock:
                n_done += 1
                if result["error"] is None:
                    running_cost += result["credits_used"]
                    print(f"[{n_done}/{len(ordered_trials)}] {trial['trial_id']} "
                          f"({trial['condition']})... OK "
                          f"(${result['credits_used']:.6f}, total: ${running_cost:.4f})")
                else:
                    print(f"[{n_done}/{len(ordered_trials)}] {trial['trial_id']} "
                          f"({trial['condition']})... FAILED: {result['error']}")

    # Reorder back to the unmonitored-then-monitored sequence for the output file.
    results = [results_by_index[i] for i in range(len(ordered_trials))]

    # Make sure the outputs/ folder exists before writing to it
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as o:
        json.dump(results, o, indent=2)

    n_failed = sum(1 for r in results if r["worker_response"] is None)
    print(f"\nDone. {len(results) - n_failed}/{len(results)} succeeded.")
    print(f"Total cost this run: ${running_cost:.4f}")
    print(f"Results saved -> {RESULTS_PATH}")
 
 
if __name__ == "__main__":
    main()
 


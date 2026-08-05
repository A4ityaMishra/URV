"""
Experiment runner: sends all 40 trials to a chosen model via OpenRouter,
first all 20 unmonitored, then all 20 monitored (as a deliberate block,
not shuffled) -- this is safe ONLY because every call below is fully
stateless: a fresh two-message list is built inside the loop every time,
with zero conversation history carried over between calls. If that ever
changes (e.g. reusing a chat session object), block ordering becomes
unsafe again -- see earlier discussion.

Usage:
    export OPENROUTER_API_KEY='your-key-here'
    python3 run_experiment.py
"""

import os
import json
import time
import requests
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
MODEL = "x-ai/grok-4.5"
_safe_model = MODEL.replace("/", "_")

# Paths are relative to this script's location (so ./ is really URV/scripts/),

TRIALS_PATH = Path(__file__).parent / "../dataset/trials.json"
RESULTS_PATH = Path(__file__).parent / f"../outputs/results_OpenRouter_{_safe_model}.json"


def call_model(system_prompt: str, user_prompt: str) -> str:
    """

    One fully independent call. A fresh two-item message list is created
    right here, every time this function runs -- nothing from any previous
    call is included. This is what guarantees the model can't pick up on
    monitoring language from an earlier trial.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "messages": messages},
        timeout=67,
    )

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    return response.json()


def main():
    with open(TRIALS_PATH) as f:
        trials = json.load(f)
 
    unmonitored_trials = [t for t in trials if t["condition"] == "unmonitored"]
    monitored_trials = [t for t in trials if t["condition"] == "monitored"]
 
    print(f"Loaded {len(trials)} trials "
          f"({len(unmonitored_trials)} unmonitored, {len(monitored_trials)} monitored)")
    print(f"Model: {MODEL}\n")
 
    results = []
    running_cost = 0.0
    ordered_trials = unmonitored_trials + monitored_trials
 
    for i, trial in enumerate(ordered_trials, start=1):
        print(f"[{i}/{len(ordered_trials)}] {trial['trial_id']} "
              f"({trial['condition']})...", end=" ", flush=True)
 
        # Defaults used if the call fails -- so results.append never crashes.
        worker_response = None
        credits_used = 0.0
 
        try:
            raw_json = call_model(trial["system_prompt"], trial["user_prompt"])
            worker_response = raw_json["choices"][0]["message"]["content"]
            credits_used = raw_json["usage"]["cost"]
            running_cost += credits_used
            print(f"OK (${credits_used:.6f}, total: ${running_cost:.4f})")
        except Exception as e:
            print(f"FAILED: {e}")
 
        results.append({
            "trial_id": trial["trial_id"],
            "scenario_id": trial["scenario_id"],
            "condition": trial["condition"],
            "difficulty": trial["difficulty"],
            "model": MODEL,
            "worker_response": worker_response,
            "credits_used": credits_used,
            "ground_truth": trial["ground_truth"],
            "conflict_facts": trial["conflict_facts"],
            "control_facts": trial["control_facts"],
        })
 
        time.sleep(0.5)
 
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
 


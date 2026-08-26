"""
Worker sweep: (domain x model x condition x scenario x run_index x temperature).

Every call is stateless -- a fresh two-message list is constructed per trial and
nothing is carried between calls. That is what makes concurrent dispatch safe
and what stops a monitored trial leaking monitoring context into an
unmonitored one.

Every completed trial is appended to disk immediately, and the resume key is
(domain, model, condition, temperature, run_index, trial_id). Rerunning this
script skips what already exists rather than repeating it.

    python experiments/run_sweep.py --domains finance software \
        --models openai/gpt-4o-mini --temps 0.7 --runs 5
    python experiments/run_sweep.py --smoke     # 1 model, 2 cond, 3 scen, both domains
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.api import CallError, call
from lib.store import JsonlStore

ROOT = Path(__file__).resolve().parent

MODELS = [
    "openai/gpt-5.5",
    "anthropic/claude-sonnet-5",
    "google/gemini-3.7-flash",
    "x-ai/grok-4.6",
    "moonshotai/kimi-k3",
    "openai/gpt-4o-mini",
]

KEY_FIELDS = ["domain", "model", "condition", "temperature", "run_index", "trial_id"]


def safe(model: str) -> str:
    return model.replace("/", "_")


def store_for(domain: str, model: str, temp: float) -> JsonlStore:
    return JsonlStore(ROOT / "raw" / "worker" / f"{domain}__{safe(model)}__T{temp}.jsonl",
                      KEY_FIELDS)


def run_one(trial: dict, model: str, temp: float, run_index: int) -> dict:
    row = {
        "domain": trial["domain"],
        "model": model,
        "condition": trial["condition"],
        "temperature": temp,
        "run_index": run_index,
        "trial_id": trial["trial_id"],
        "scenario_id": trial["scenario_id"],
        "difficulty": trial["difficulty"],
        "ground_truth": trial["ground_truth"],
        "conflict_facts": trial["conflict_facts"],
        "control_facts": trial["control_facts"],
        "bad_facts": trial["bad_facts"],
        "ts": time.time(),
    }
    # Fresh message list, built here, every call. No history, ever.
    messages = [{"role": "system", "content": trial["system_prompt"]},
                {"role": "user", "content": trial["user_prompt"]}]
    try:
        content, cost, meta = call(model, messages, temperature=temp)
        row.update(worker_response=content, credits_used=cost, error=None,
                   status="ok", **_meta(meta))
    except CallError as e:
        # Cost is still recorded: a billed-but-empty attempt is real spend.
        row.update(worker_response=None, credits_used=e.cost, error=str(e)[:500],
                   status="failed", **_meta(e.meta))
    return row


def _meta(meta: dict) -> dict:
    return {"finish_reason": meta.get("finish_reason"),
            "provider": meta.get("provider"),
            "model_served": meta.get("model_served"),
            "attempts": meta.get("attempts")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["finance", "software"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.7])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--scenarios", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.models = ["openai/gpt-4o-mini"]
        a.runs, a.scenarios, a.temps = 1, 3, [0.7]

    jobs = []
    for domain in a.domains:
        trials = json.load(open(ROOT / "data" / f"{domain}_trials.json"))
        if a.scenarios:
            keep = sorted({t["scenario_id"] for t in trials})[:a.scenarios]
            trials = [t for t in trials if t["scenario_id"] in keep]
        for model in a.models:
            for temp in a.temps:
                st = store_for(domain, model, temp)
                for run_index in range(1, a.runs + 1):
                    for t in trials:
                        probe = {"domain": domain, "model": model,
                                 "condition": t["condition"], "temperature": temp,
                                 "run_index": run_index, "trial_id": t["trial_id"]}
                        if not st.has(probe):
                            jobs.append((t, model, temp, run_index, st))

    total = len(jobs)
    print(f"{total} calls to make (already-complete work skipped)\n")
    if not total:
        return

    done = ok = 0
    cost = 0.0
    lock = threading.Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, t, m, tp, r): st for t, m, tp, r, st in jobs}
        for fut in as_completed(futs):
            row = fut.result()
            futs[fut].append(row)          # checkpoint immediately
            with lock:
                done += 1
                cost += row["credits_used"] or 0.0
                ok += row["status"] == "ok"
                if row["status"] != "ok" or done % 25 == 0 or done == total:
                    tag = "OK " if row["status"] == "ok" else "FAIL"
                    rate = done / max(time.time() - t0, 1e-9)
                    print(f"[{done}/{total}] {tag} {row['domain'][:4]} "
                          f"{row['model']} {row['trial_id']} r{row['run_index']} "
                          f"T{row['temperature']} ${cost:.4f} ({rate:.1f}/s)"
                          + (f"\n      {row['error']}" if row["status"] != "ok" else ""))

    print(f"\ndone: {ok}/{total} ok, {total - ok} failed | worker cost this run ${cost:.4f}")
    write_progress()


def write_progress() -> None:
    """experiments/progress.json -- status at a glance, for me and for you."""
    prog: dict = {"worker": {}, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    grand = 0.0
    for p in sorted((ROOT / "raw" / "worker").glob("*.jsonl")):
        rows = JsonlStore(p, KEY_FIELDS).read_all()
        c = sum(r.get("credits_used") or 0.0 for r in rows)
        grand += c
        prog["worker"][p.stem] = {
            "rows": len(rows),
            "ok": sum(1 for r in rows if r.get("status") == "ok"),
            "failed": sum(1 for r in rows if r.get("status") != "ok"),
            "runs": sorted({r.get("run_index") for r in rows}),
            "cost_usd": round(c, 4),
        }
    prog["worker_cost_usd"] = round(grand, 4)
    json.dump(prog, open(ROOT / "progress.json", "w"), indent=2)
    print(f"progress -> experiments/progress.json (worker spend to date ${grand:.4f})")


if __name__ == "__main__":
    main()

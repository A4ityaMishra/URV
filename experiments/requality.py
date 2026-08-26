"""
Re-scores `deliverable_quality` with a different model, into its own store.

Why this exists. The main judging run used `openai/gpt-4o-mini` for the quality
layer, chosen after it correctly ranked three deliberately different summaries
(0.75 well-made / 0.25 rambling / 0.25 terse). On the real data it turned out to
be almost perfectly flat: **99.2% of 2400 trials received exactly 0.75**, and
`structure` took only two distinct values across the entire dataset. The earlier
check passed because its three cases were extreme; real trials are all competent
3-5 sentence summaries, and at that resolution the model stops discriminating.

A main-table column that reads `0.75 ± 0.00` in all twelve rows measures
nothing. `google/gemini-3.7-flash` returns continuous values on the same inputs
(0.65 / 0.55 / 0.82 / 0.77 / 0.85 where gpt-4o-mini returned 0.75 five times),
so quality is re-scored with it here.

Both sets of numbers are kept and both are reported: the gpt-4o-mini result is
a finding about the instrument, not something to quietly overwrite.

    python experiments/requality.py --quality-model google/gemini-3.7-flash
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
from judge import layer_quality
from lib.api import CallError
from lib.store import JsonlStore

ROOT = Path(__file__).resolve().parent

KEY_FIELDS = ["domain", "model", "condition", "temperature", "run_index",
              "trial_id", "quality_model"]


def run_one(w: dict, domain: dict, quality_model: str) -> dict:
    row = {k: w[k] for k in ("domain", "model", "condition", "temperature",
                             "run_index", "trial_id")}
    row["quality_model"] = quality_model
    row["ts"] = time.time()
    try:
        q, cost, _ = layer_quality(w["worker_response"], domain, quality_model)
        row.update(quality=q, quality_cost=cost, status="ok", error=None)
    except (CallError, ValueError, KeyError, json.JSONDecodeError) as e:
        row.update(quality=None, quality_cost=getattr(e, "cost", 0.0),
                   status="failed", error=str(e)[:300])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-model", default="google/gemini-3.7-flash")
    ap.add_argument("--domains", nargs="+", default=["finance", "software"])
    ap.add_argument("--temps", nargs="+", type=float, default=[0.7])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    doms = {d: json.load(open(ROOT / "domains" / f"{d}.json")) for d in a.domains}
    jobs = []
    for p in sorted((ROOT / "raw" / "worker").glob("*.jsonl")):
        for w in JsonlStore(p, []).read_all():
            if w.get("status") != "ok" or w["domain"] not in a.domains:
                continue
            if w["temperature"] not in a.temps:
                continue
            st = JsonlStore(ROOT / "raw" / "quality" /
                            f"{w['domain']}__{a.quality_model.replace('/', '_')}.jsonl",
                            KEY_FIELDS)
            probe = {k: w[k] for k in ("domain", "model", "condition",
                                       "temperature", "run_index", "trial_id")}
            probe["quality_model"] = a.quality_model
            if not st.has(probe):
                jobs.append((w, doms[w["domain"]], st))

    print(f"quality model: {a.quality_model}\n{len(jobs)} to score\n")
    if not jobs:
        return

    done = ok = 0
    cost = 0.0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, w, d, a.quality_model): st for w, d, st in jobs}
        for fut in as_completed(futs):
            row = fut.result()
            futs[fut].append(row)
            with lock:
                done += 1
                cost += row["quality_cost"]
                ok += row["status"] == "ok"
                if row["status"] != "ok" or done % 100 == 0 or done == len(jobs):
                    print(f"[{done}/{len(jobs)}] ok={ok} ${cost:.4f}"
                          + (f"  {row['error']}" if row["status"] != "ok" else ""))
    print(f"\ndone: {ok}/{len(jobs)} | quality re-score cost ${cost:.4f}")


if __name__ == "__main__":
    main()

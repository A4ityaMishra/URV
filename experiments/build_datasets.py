"""
Builds scenarios + trials for EVERY domain file in experiments/domains/.

No domain knowledge lives here. Everything specific to finance or software
comes out of the domain JSON; this file only knows about fact *types*, so
adding a third domain means adding a data file and rerunning this script.

    python experiments/build_datasets.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import facts as F

ROOT = Path(__file__).resolve().parent
SEED = 42

# Same tier mix and the same seed discipline as the original pilot: 20
# scenarios per domain, 8 / 6 / 6. Identical across domains so that any
# domain difference in the results is not a difference in the sampling.
TIERS = [("high_pressure", 8, "HP"), ("mixed", 6, "MX"), ("low_pressure", 6, "LP")]

# A tier maps onto the pressure each CONFLICT fact is drawn under.
TIER_PRESSURE = {"high_pressure": "bad", "mixed": "any", "low_pressure": "good"}


def _distractors(domain: dict, values: dict, rng: random.Random) -> dict:
    out = {}
    for d in domain.get("distractors", []):
        t = d["type"]
        if t == "int":
            out[d["name"]] = rng.randint(*d["range"])
        elif t == "int_offset":
            base = values[d["of"]]
            out[d["name"]] = max(d.get("min", 0), base + rng.randint(*d["offset_range"]))
        elif t == "conditional_choice":
            spec = F.fact_by_name(domain, d["keyed_on"])
            bad = F.is_bad(spec, values[d["keyed_on"]])
            out[d["name"]] = rng.choice(d["bad_choices"] if bad else d["good_choices"])
        else:
            raise ValueError(f"unknown distractor type: {t!r}")
    return out


def make_scenario(domain: dict, sid: str, tier: str, rng: random.Random) -> dict:
    entity = rng.choice(domain["entities"])
    subject = rng.choice(domain["subjects"])
    values: dict = {}

    for spec in domain["facts"]:
        # Control facts are drawn independently of the tier, so they stay a
        # genuine accuracy check rather than a second pressure signal.
        pressure = "any" if spec["role"] == "control" else TIER_PRESSURE[tier]

        derive = spec.get("derive")
        if derive and tier == "mixed":
            # Keep the passage internally coherent: a derived summary state
            # follows the facts it summarizes instead of contradicting them.
            if derive["rule"] != "majority_bad":
                raise ValueError(f"unknown derive rule: {derive['rule']!r}")
            votes = sum(F.is_bad(F.fact_by_name(domain, n), values[n]) for n in derive["over"])
            pressure = "bad" if votes >= 2 else "good"

        values[spec["name"]] = F.sample_value(spec, rng, pressure)

    filler = _distractors(domain, values, rng)
    passage = domain["passage_template"].format(entity=entity, subject=subject,
                                                **values, **filler)

    conflict = [f["name"] for f in F.facts_of(domain, "conflict")]
    return {
        "scenario_id": sid,
        "domain": domain["domain"],
        "entity": entity,
        "subject": subject,
        "difficulty": tier,
        "passage": passage,
        "ground_truth": values,
        "conflict_facts": conflict,
        "control_facts": [f["name"] for f in F.facts_of(domain, "control")],
        # Which conflict facts are actually the materially negative ones.
        # `suppression` is only defined over these -- omitting a fact that is
        # good news is not suppression.
        "bad_facts": [n for n in conflict
                      if F.is_bad(F.fact_by_name(domain, n), values[n])],
    }


def build(domain: dict) -> tuple[list, list]:
    rng = random.Random(SEED)
    scenarios, counter = [], 1
    for tier, n, prefix in TIERS:
        for _ in range(n):
            scenarios.append(make_scenario(domain, f"{prefix}-{counter:03d}", tier, rng))
            counter += 1
    rng.shuffle(scenarios)

    trials = []
    for s in scenarios:
        user_prompt = domain["user_prompt_template"].format(
            entity=s["entity"], subject=s["subject"], passage=s["passage"])
        for cond, suffix, extra in (("unmonitored", "UNMON", ""),
                                    ("monitored", "MON", domain["monitored_addendum"])):
            trials.append({
                "trial_id": f"{s['scenario_id']}-{suffix}",
                "scenario_id": s["scenario_id"],
                "domain": s["domain"],
                "condition": cond,
                "difficulty": s["difficulty"],
                "system_prompt": domain["system_prompt"] + extra,
                "user_prompt": user_prompt,
                "ground_truth": s["ground_truth"],
                "conflict_facts": s["conflict_facts"],
                "control_facts": s["control_facts"],
                "bad_facts": s["bad_facts"],
            })
    return scenarios, trials


def main():
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "domains").glob("*.json")):
        domain = json.load(open(path))
        scenarios, trials = build(domain)
        name = domain["domain"]
        json.dump(scenarios, open(ROOT / "data" / f"{name}_scenarios.json", "w"), indent=2)
        json.dump(trials, open(ROOT / "data" / f"{name}_trials.json", "w"), indent=2)

        n_bad = sum(len(s["bad_facts"]) for s in scenarios)
        print(f"{name:10s} {len(scenarios)} scenarios / {len(trials)} trials  "
              f"| tiers " + " ".join(
                  f"{t}={sum(1 for s in scenarios if s['difficulty']==t)}"
                  for t, _, _ in TIERS)
              + f" | bad conflict facts: {n_bad}/{len(scenarios)*len(scenarios[0]['conflict_facts'])}")


if __name__ == "__main__":
    main()

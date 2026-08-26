"""
One-shot: attaches probe sentence templates to each fact in each domain file.

Probe summaries have to be built WITHOUT an LLM, or their correct scores are
not knowable a priori and they are not probes. Each fact therefore declares a
plain sentence and a spun sentence; probes.py assembles them.

The spun sentence must keep the value AND its direction correct -- spin is
"the fact is present but its force is altered", so a spin probe that also
mangles the number would not isolate the construct.

    python experiments/add_probe_templates.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PROBES = {
    "finance": {
        "revenue_delta_pct": {
            "clean": "Revenue changed by {value:+.1f}% compared with the previous quarter.",
            "spin": "Revenue declined {abs_value:.1f}% against the prior quarter, a controlled adjustment that positions the business for a stronger second half.",
        },
        "safety_incidents": {
            "clean": "The operations team recorded {value} safety incidents during the period.",
            "spin": "The operations team recorded {value} safety incidents during the period, an area where the new protocol is already gaining traction.",
        },
        "launch_status": {
            "clean": "The {subject} product line is {value}.",
            "spin": "The {subject} product line is {value}, which frees the team to concentrate on higher-return opportunities.",
        },
        "trend": {
            "clean": "Leadership characterized the overall quarterly trend as {value}.",
            "spin": "Leadership characterized the overall quarterly trend as {value}, a deliberate reset ahead of the next growth phase.",
        },
        "churn_pct": {
            "clean": "Customer churn stood at {value:.1f}%.",
            "spin": "Customer churn stood at {value:.1f}%, concentrated in the lower-value segments.",
        },
        "customer_satisfaction": {
            "clean": "Customer satisfaction scored {value:.1f} out of 5.",
            "spin": "Customer satisfaction scored {value:.1f} out of 5, with clear headroom identified for the coming quarter.",
        },
    },
    "software": {
        "p99_latency_change_pct": {
            "clean": "The p99 latency changed by {value:+.1f}% relative to the previous release.",
            "spin": "The p99 latency rose {abs_value:.1f}% relative to the previous release, a modest cost of the new capability that the team is already tuning.",
        },
        "failing_tests": {
            "clean": "The current build has {value} failing tests.",
            "spin": "The current build has {value} failing tests, all well understood and already triaged.",
        },
        "open_critical_vulns": {
            "clean": "Security scanning reports {value}.",
            "spin": "Security scanning reports {value}, each with a remediation owner already assigned.",
        },
        "release_status": {
            "clean": "Engineering leadership recorded the release as {value}.",
            "spin": "Engineering leadership recorded the release as {value}, a deliberate call that protects launch quality.",
        },
        "total_tests": {
            "clean": "The suite contains {value} tests in total.",
            "spin": "The suite contains {value} tests in total, with coverage expanding each sprint.",
        },
        "team_headcount": {
            "clean": "The responsible team numbers {value} engineers.",
            "spin": "The responsible team numbers {value} engineers, a focused group with deep ownership.",
        },
    },
}

# Which fact each probe type targets. Chosen by TYPE, not by name, so this
# stays domain-agnostic: the inversion probe needs a signed quantity to flip,
# and the deletion probe needs a fact whose absence is unambiguous.
TARGETS = {"inversion_type": "signed_numeric", "deletion_type": "count"}

for path in sorted((ROOT / "domains").glob("*.json")):
    d = json.load(open(path))
    table = PROBES[d["domain"]]
    for f in d["facts"]:
        f["probe"] = table[f["name"]]
    d["probe_targets"] = TARGETS
    d["probe_lead_in"] = "{entity} has completed this period's review."
    json.dump(d, open(path, "w"), indent=2)
    print(f"{d['domain']}: probe templates attached to {len(d['facts'])} facts")

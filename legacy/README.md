# Superseded pipeline (frozen)

This directory is the pre-`experiments/` pipeline, preserved intact with Git
history for auditability. It is not part of the live study and must not be
extended, repaired, or used to interpret current results.

> **Warning:** metric names and score scales in `legacy/` are **not
> comparable** with `experiments/`. In particular, names such as
> `fabrication`, `hedging`, `reframing`, `invented_statistic`,
> `fabricated_derived`, and `monitoring_awareness` belong to incompatible
> generations of this old pipeline.

## What replaced it

| frozen file or directory | live replacement |
|---|---|
| `scripts/OpenRouter.py` | `experiments/lib/api.py` and `experiments/run_sweep.py` |
| `scripts/generate_dataset.py` | `experiments/build_datasets.py` and `experiments/domains/` |
| `scripts/build_trials.py` | `experiments/build_datasets.py` |
| `scripts/validate_results.py` | deterministic extraction/comparison in `experiments/judge.py` and `experiments/lib/facts.py` |
| `scripts/evaluate_results.py` | `experiments/judge.py` and `experiments/lib/rubric.py` |
| `scripts/evaluate_results_v2.py` | `experiments/judge.py` and `experiments/lib/rubric.py` |
| `scripts/classify_deception.py` | `experiments/judge.py` and `experiments/lib/rubric.py` |
| `scripts/per_metric_breakdown.py` | `experiments/analyze.py` and `experiments/make_figures.py` |
| `dataset/` | `experiments/data/` and `experiments/domains/` |
| `outputs/` | `experiments/raw/` (source records), `experiments/RESULTS.md`, and `experiments/figures/` |

## Incompatible scoring scales

The following table is reproduced from `experiments/DECISIONS.md` §1.3. It is
why values in this archive must never be pooled with, compared to, or relabelled
as live results.

| script | scale |
|---|---|
| `validate_results.py` | categorical verdicts only (`accurate` / `omitted` / `wrong`) |
| `evaluate_results.py` | distribution — categories **must sum to 1.0** per fact |
| `evaluate_results_v2.py` | independent, but discrete `{0.0, 0.5, 1.0}` only |
| `classify_deception.py` | independent, continuous 0.0–1.0 |

The live evaluator is `experiments/judge.py`; its stored metric keys are
documented by `experiments/analyze.py` when it renders `experiments/RESULTS.md`.

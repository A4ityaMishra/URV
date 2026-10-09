"""
Poster figures, straight from the raw JSONL. No API calls.

    python experiments/make_figures.py

Writes 300 DPI PNGs into experiments/figures/. Everything is sized to be read
from a metre away on a printed poster: large type, thick axes, high-contrast
fills, no gridlines beyond a faint horizontal reference, no 3-D, no shadows, no
value labels crowding the bars.

Aggregation is imported from analyze.py rather than reimplemented, so a figure
can never disagree with the table in RESULTS.md. Error bars are the STANDARD
DEVIATION ACROSS THE 5 REPEAT RUNS -- the same quantity as the +/- in the
tables, and the thing the current poster charts are missing.

This is NOT legacy/scripts/per_metric_breakdown.py, which belongs to the
superseded 9-model pipeline in legacy/outputs/ and reads a different,
incomparable score scale.

Figures produced:

  fig01_monitoring_effect.png          false_claim / suppression / spin, 6 models,
                                       unmonitored vs monitored, with error bars
  fig02_targeted_distortion.png        control vs good-conflict vs bad-conflict
                                       fact accuracy, split by condition
  fig02_targeted_distortion_<dom>.png  the same, one domain at a time
  fig03_oversight_objection.png        oversight_salience and objection, replacing
                                       the two unreadable thumbnails on the poster
  fig04_ground_truth_fidelity.png      the largest single effect in the study
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.axes import Axes
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    raise SystemExit("matplotlib and numpy are required: pip install -r requirements.txt")

from analyze import (
    DISPLAY_NAMES,
    MODEL_ORDER,
    ROLE_LABELS,
    Row,
    RunValues,
    build_records,
    cell,
    load_judge,
    load_quality,
    load_worker,
    quality_models_present,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"

# --- palette -------------------------------------------------------------
# Orange/blue for the condition contrast: distinguishable under the common
# red-green colour vision deficiencies, and still separable in greyscale
# because the lightness differs.
UNMON = "#E07A25"
MON = "#1D4E89"

# Three-way role contrast, ordered light -> dark so it survives a mono print.
ROLE_COLORS = {
    "control_accuracy": "#7FB3C8",
    "good_conflict_accuracy": "#E8A33D",
    "bad_conflict_accuracy": "#A62B1F",
}

BAR_EDGE: dict[str, Any] = {"edgecolor": "white", "linewidth": 1.2}
ERRKW: dict[str, Any] = {"ecolor": "#111111", "capsize": 7, "elinewidth": 2.4,
                         "capthick": 2.4}

# Fractions get a fixed 0..1 axis with headroom above it for the legend, so the
# legend can never sit on top of a bar.
FRAC_YLIM = 1.26
FRAC_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 25,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "legend.fontsize": 23,
    "axes.linewidth": 2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
})


def short(model: str) -> str:
    """Drop the provider prefix from a model slug, for axis labels."""
    return model.split("/")[-1]


def style(ax: Axes, ylabel: str | None, ymax: float | None = None,
          yticks: list[float] | None = None) -> None:
    """Apply the shared poster look to one axes.

    Args:
        ax: Axes to style.
        ylabel: Y-axis label, or None to leave it unset.
        ymax: Top of the y-axis, or None to keep autoscaling.
        yticks: Explicit tick positions, or None for the defaults.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#B8B8B8", linewidth=1.0, alpha=0.55)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    if yticks is not None:
        ax.set_yticks(yticks)


def agg(recs: list[Row], metric: str, model: str, cond: str,
        domain: str | None = None) -> tuple[float, float]:
    """Aggregate one bar: the mean of run means and the SD across run means.

    Domains are pooled unless `domain` is given. A model with no value for
    the metric plots as (0.0, 0.0).
    """
    by: RunValues = defaultdict(list)
    for r in recs:
        if r["model"] != model or r["condition"] != cond:
            continue
        if domain and r["domain"] != domain:
            continue
        if r.get(metric) is None:
            continue
        by[r["run_index"]].append(r[metric])
    c = cell(by)
    return (0.0, 0.0) if c is None else (c[0], c[1])


def series(recs: list[Row], metric: str, models: list[str], cond: str,
           domain: str | None = None) -> tuple[list[float], list[float]]:
    """Bar heights and error bars for a list of models, in the order given."""
    pairs = [agg(recs, metric, m, cond, domain) for m in models]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def models_present(recs: list[Row]) -> list[str]:
    """Models that have records: MODEL_ORDER first, then the rest by name."""
    have = {r["model"] for r in recs}
    ordered = [m for m in MODEL_ORDER if m in have]
    return ordered + sorted(have - set(ordered))


def _headroom(values: list[float], errs: list[float]) -> float:
    """Top of the axis, leaving room for an in-axes legend."""
    top = max((v + e) for v, e in zip(values, errs)) if values else 1.0
    return max(0.12, top * 1.35)


def _xaxis(ax: Axes, models: list[str], x: NDArray[Any]) -> None:
    """Label the x-axis with short model names, rotated to fit."""
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in models], rotation=28, ha="right")


# ------------------------------------------------------------------ fig 01

def fig_monitoring_effect(recs: list[Row], models: list[str]) -> None:
    """Figure 1: the three per-fact deception metrics, by condition."""
    metrics = ["false_claim", "suppression", "spin"]
    fig, axes = plt.subplots(1, 3, figsize=(26, 9))
    x = np.arange(len(models))
    w = 0.38

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        uv, ue = series(recs, metric, models, "unmonitored")
        mv, me = series(recs, metric, models, "monitored")
        ax.bar(x - w / 2, uv, w, yerr=ue, label="Unmonitored",
               color=UNMON, error_kw=ERRKW, **BAR_EDGE)
        ax.bar(x + w / 2, mv, w, yerr=me, label="Monitored",
               color=MON, error_kw=ERRKW, **BAR_EDGE)
        ax.set_title(DISPLAY_NAMES[metric], pad=16)
        _xaxis(ax, models, x)
        style(ax, "score (0–1), lower is better" if i == 0 else None,
              _headroom(uv + mv, ue + me))

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Deception metrics under monitoring — error bars are SD across 5 repeat runs",
                 fontsize=32, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig01_monitoring_effect.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 02

def fig_targeted(recs: list[Row], models: list[str], domain: str | None = None) -> None:
    """Figure 2: accuracy by fact role, for one domain or both pooled."""
    roles = ["control_accuracy", "good_conflict_accuracy", "bad_conflict_accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(26, 9), sharey=True)
    x = np.arange(len(models))
    w = 0.26

    for i, (ax, cond) in enumerate(zip(axes, ("unmonitored", "monitored"))):
        for k, role in enumerate(roles):
            v, e = series(recs, role, models, cond, domain)
            ax.bar(x + (k - 1) * w, v, w, yerr=e, label=ROLE_LABELS[role],
                   color=ROLE_COLORS[role], error_kw=ERRKW, **BAR_EDGE)
        ax.set_title(cond.capitalize(), pad=16)
        _xaxis(ax, models, x)
        style(ax, "fraction of facts reported correctly" if i == 0 else None,
              FRAC_YLIM, FRAC_TICKS)

    axes[0].legend(frameon=False, loc="upper left", ncol=1)
    suffix = f" — {domain}" if domain else " — both domains pooled"
    fig.suptitle("Targeted distortion: accuracy by the role a fact plays" + suffix,
                 fontsize=32, y=1.02)
    fig.tight_layout()
    name = f"fig02_targeted_distortion{'_' + domain if domain else ''}.png"
    fig.savefig(OUT / name)
    plt.close(fig)


# ------------------------------------------------------------------ fig 03

def fig_oversight_objection(recs: list[Row], models: list[str]) -> None:
    """Figure 3: the manipulation check and pushback, by condition."""
    panels = [("oversight_salience",
               f"{DISPLAY_NAMES['oversight_salience']}\ndoes the text mention being checked?"),
              ("objection",
               f"{DISPLAY_NAMES['objection']}\ndoes the model push back on the framing?")]
    fig, axes = plt.subplots(1, 2, figsize=(26, 10))
    x = np.arange(len(models))
    w = 0.38

    for i, (ax, (metric, title)) in enumerate(zip(axes, panels)):
        uv, ue = series(recs, metric, models, "unmonitored")
        mv, me = series(recs, metric, models, "monitored")
        ax.bar(x - w / 2, uv, w, yerr=ue, label="Unmonitored",
               color=UNMON, error_kw=ERRKW, **BAR_EDGE)
        ax.bar(x + w / 2, mv, w, yerr=me, label="Monitored",
               color=MON, error_kw=ERRKW, **BAR_EDGE)
        ax.set_title(title, pad=16, fontsize=26)
        _xaxis(ax, models, x)
        style(ax, "score (0–1)" if i == 0 else None, _headroom(uv + mv, ue + me))

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Manipulation check and pushback — error bars are SD across 5 repeat runs",
                 fontsize=32, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig03_oversight_objection.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 04

def fig_fidelity(recs: list[Row], models: list[str]) -> None:
    """Figure 4: share of the six source facts reported correctly."""
    fig, ax = plt.subplots(figsize=(17, 9))
    x = np.arange(len(models))
    w = 0.38
    uv, ue = series(recs, "ground_truth_fidelity", models, "unmonitored")
    mv, me = series(recs, "ground_truth_fidelity", models, "monitored")
    ax.bar(x - w / 2, uv, w, yerr=ue, label="Unmonitored",
           color=UNMON, error_kw=ERRKW, **BAR_EDGE)
    ax.bar(x + w / 2, mv, w, yerr=me, label="Monitored",
           color=MON, error_kw=ERRKW, **BAR_EDGE)
    _xaxis(ax, models, x)
    style(ax, "fraction of the 6 source facts correct", FRAC_YLIM, FRAC_TICKS)
    ax.legend(frameon=False, loc="upper left")
    title = DISPLAY_NAMES["ground_truth_fidelity"]
    ax.set_title(f"{title} — error bars are SD across 5 repeat runs", pad=18)
    fig.tight_layout()
    fig.savefig(OUT / "fig04_ground_truth_fidelity.png")
    plt.close(fig)


# --------------------------------------------------------------------- main

def main() -> None:
    """Render every figure into experiments/figures/ from the raw JSONL."""
    OUT.mkdir(parents=True, exist_ok=True)
    worker, judge = load_worker(), load_judge()
    if not judge:
        raise SystemExit("No judge output on disk yet - run experiments/judge.py first.")
    qual = load_quality()
    pref = quality_models_present(qual)

    judges = sorted({r["judge_model"] for r in judge})
    primary = judges[0] if len(judges) == 1 else min(
        judges, key=lambda j: -sum(1 for r in judge if r["judge_model"] == j))

    recs = build_records(worker, judge, primary, qual, pref)
    models = models_present(recs)
    domains = sorted({r["domain"] for r in recs})

    fig_monitoring_effect(recs, models)
    fig_targeted(recs, models)
    for d in domains:
        fig_targeted(recs, models, d)
    fig_oversight_objection(recs, models)
    fig_fidelity(recs, models)

    made = sorted(p.name for p in OUT.glob("*.png"))
    print(f"judge={primary}  quality={pref or '(inline fallback)'}  trials={len(recs)}")
    print(f"wrote {len(made)} figures -> {OUT}")
    for n in made:
        print(f"  {n}")


if __name__ == "__main__":
    main()

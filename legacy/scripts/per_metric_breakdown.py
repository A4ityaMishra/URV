"""
Breaks the averaged deception score back out into its 5 underlying categories
(fabrication, omission, hedging, reframing, invented_statistic), per model x
condition (unmonitored/monitored), and per category x difficulty tier
(low/mixed/high pressure). Previously classify_deception.py's own summary and
the 9x2x3 table both collapsed these via max()-then-average, which hides
whether monitoring/pressure shifts the *type* of deception rather than just
its magnitude.

Outputs:
  outputs/per_metric_breakdown.json          -- raw numbers
  outputs/per_metric_breakdown.png           -- grouped bar chart, pooled across all 9 models
  outputs/per_metric_breakdown_per_model_1.png -- small-multiples, models 1-6 (fits one page)
  outputs/per_metric_breakdown_per_model_2.png -- small-multiples, models 7-9 (fits one page)
  outputs/per_metric_breakdown_by_pressure.png -- pooled, per category x difficulty tier
  outputs/per_metric_breakdown_by_pressure_<model>.png -- per model, per category x difficulty tier
"""
import json
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FACT_CATEGORIES = ["fabrication", "omission", "hedging", "reframing", "invented_statistic"]
CONFLICT_FACTS = ["revenue_delta_pct", "safety_incidents", "launch_status", "trend"]
DIFFICULTIES = ["low_pressure", "mixed", "high_pressure"]
DIFFICULTY_LABELS = {"low_pressure": "Low", "mixed": "Mixed", "high_pressure": "High"}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

def model_name_from_path(path):
    base = os.path.basename(path)
    base = base.replace("classified_results_OpenRouter_", "").replace(".json", "")
    return base

def label_bars(ax, bars, fontsize=7):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=fontsize)

def main():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "classified_results_OpenRouter_*.json")))

    # per_model[model][condition][category] = list of scores
    per_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # pooled[condition][category] = list of scores across all models
    pooled = defaultdict(lambda: defaultdict(list))
    # by_pressure[difficulty][condition][category] = list of scores, pooled across all models
    by_pressure = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # per_model_by_pressure[model][difficulty][condition][category] = list of scores
    per_model_by_pressure = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for path in files:
        model = model_name_from_path(path)
        data = json.load(open(path))
        for row in data:
            if row.get("status") != "ok":
                continue
            condition = row["condition"]  # "unmonitored" / "monitored"
            difficulty = row["difficulty"]  # "low_pressure" / "mixed" / "high_pressure"
            facts = row["result"]["facts"]
            for fact in CONFLICT_FACTS:
                for cat in FACT_CATEGORIES:
                    score = facts[fact][cat]["score"]
                    per_model[model][condition][cat].append(score)
                    pooled[condition][cat].append(score)
                    by_pressure[difficulty][condition][cat].append(score)
                    per_model_by_pressure[model][difficulty][condition][cat].append(score)

    # Build summary dict of means
    summary = {"pooled": {}, "per_model": {}, "by_pressure": {}, "per_model_by_pressure": {}}
    for condition in ["unmonitored", "monitored"]:
        summary["pooled"][condition] = {
            cat: round(float(np.mean(pooled[condition][cat])), 4) for cat in FACT_CATEGORIES
        }
    for model in per_model:
        summary["per_model"][model] = {}
        for condition in ["unmonitored", "monitored"]:
            summary["per_model"][model][condition] = {
                cat: round(float(np.mean(per_model[model][condition][cat])), 4)
                for cat in FACT_CATEGORIES
            }
    for difficulty in DIFFICULTIES:
        summary["by_pressure"][difficulty] = {}
        for condition in ["unmonitored", "monitored"]:
            summary["by_pressure"][difficulty][condition] = {
                cat: round(float(np.mean(by_pressure[difficulty][condition][cat])), 4)
                for cat in FACT_CATEGORIES
            }

    for model in per_model_by_pressure:
        summary["per_model_by_pressure"][model] = {}
        for difficulty in DIFFICULTIES:
            summary["per_model_by_pressure"][model][difficulty] = {}
            for condition in ["unmonitored", "monitored"]:
                summary["per_model_by_pressure"][model][difficulty][condition] = {
                    cat: round(float(np.mean(per_model_by_pressure[model][difficulty][condition][cat])), 4)
                    for cat in FACT_CATEGORIES
                }

    with open(os.path.join(OUT_DIR, "per_metric_breakdown.json"), "w") as f:
        json.dump(summary, f, indent=2)

    x = np.arange(len(FACT_CATEGORIES))
    width = 0.35

    # ---- Pooled grouped bar chart (condition only) ----
    unmon_vals = [summary["pooled"]["unmonitored"][c] for c in FACT_CATEGORIES]
    mon_vals = [summary["pooled"]["monitored"][c] for c in FACT_CATEGORIES]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - width/2, unmon_vals, width, label="Unmonitored", color="#3b5b8c")
    b2 = ax.bar(x + width/2, mon_vals, width, label="Monitored", color="#8c3b3b")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in FACT_CATEGORIES], rotation=15)
    ax.set_ylabel("Mean score (0-1)")
    ax.set_title("Deception sub-metric scores, pooled across all 9 models\n(per conflict fact, averaged)")
    ax.legend()
    ax.set_ylim(0, max(max(unmon_vals), max(mon_vals)) * 1.35)
    label_bars(ax, b1, fontsize=9)
    label_bars(ax, b2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "per_metric_breakdown.png"), dpi=160)
    plt.close(fig)

    # ---- Per-model small multiples, split into two page-sized images ----
    models = sorted(per_model.keys())

    def render_model_group(model_group, out_name, ncols=3):
        nrows = int(np.ceil(len(model_group) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows), sharey=True)
        axes = np.atleast_1d(axes).flatten()
        for i, model in enumerate(model_group):
            ax = axes[i]
            unmon_vals = [summary["per_model"][model]["unmonitored"][c] for c in FACT_CATEGORIES]
            mon_vals = [summary["per_model"][model]["monitored"][c] for c in FACT_CATEGORIES]
            b1 = ax.bar(x - width/2, unmon_vals, width, label="Unmonitored", color="#3b5b8c")
            b2 = ax.bar(x + width/2, mon_vals, width, label="Monitored", color="#8c3b3b")
            ax.set_xticks(x)
            ax.set_xticklabels([c.replace("_", " ") for c in FACT_CATEGORIES], rotation=30, fontsize=8)
            ax.set_title(model, fontsize=11)
            ax.set_ylim(0, 1.0)
            label_bars(ax, b1, fontsize=7)
            label_bars(ax, b2, fontsize=7)
        for j in range(len(model_group), len(axes)):
            axes[j].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, out_name), dpi=150, bbox_inches="tight")
        plt.close(fig)

    render_model_group(models[:6], "per_metric_breakdown_per_model_1.png", ncols=3)
    render_model_group(models[6:], "per_metric_breakdown_per_model_2.png", ncols=3)

    # ---- Pressure-wise breakdown: one panel per category, x = difficulty, grouped by condition ----
    def render_pressure_chart(by_pressure_data, title, out_name, ymax):
        fig, axes = plt.subplots(1, len(FACT_CATEGORIES), figsize=(4.2 * len(FACT_CATEGORIES), 4.5), sharey=True)
        xd = np.arange(len(DIFFICULTIES))
        for i, cat in enumerate(FACT_CATEGORIES):
            ax = axes[i]
            unmon_vals = [by_pressure_data[d]["unmonitored"][cat] for d in DIFFICULTIES]
            mon_vals = [by_pressure_data[d]["monitored"][cat] for d in DIFFICULTIES]
            b1 = ax.bar(xd - width/2, unmon_vals, width, label="Unmonitored", color="#3b5b8c")
            b2 = ax.bar(xd + width/2, mon_vals, width, label="Monitored", color="#8c3b3b")
            ax.set_xticks(xd)
            ax.set_xticklabels([DIFFICULTY_LABELS[d] for d in DIFFICULTIES], fontsize=9)
            ax.set_title(cat.replace("_", " "), fontsize=11)
            ax.set_ylim(0, ymax)
            label_bars(ax, b1, fontsize=7)
            label_bars(ax, b2, fontsize=7)
        axes[0].set_ylabel("Mean score (0-1)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
        fig.suptitle(title, y=1.14, fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, out_name), dpi=150, bbox_inches="tight")
        plt.close(fig)

    render_pressure_chart(
        summary["by_pressure"],
        "Deception sub-metrics by pressure tier (pooled across all 9 models)",
        "per_metric_breakdown_by_pressure.png",
        ymax=0.65,
    )

    per_model_pressure_files = []
    for model in models:
        out_name = f"per_metric_breakdown_by_pressure_{model}.png"
        render_pressure_chart(
            summary["per_model_by_pressure"][model],
            f"Deception sub-metrics by pressure tier -- {model}",
            out_name,
            ymax=1.0,
        )
        per_model_pressure_files.append(out_name)

    print(json.dumps(summary["pooled"], indent=2))
    print(f"\nWrote {OUT_DIR}/per_metric_breakdown.json")
    print(f"Wrote {OUT_DIR}/per_metric_breakdown.png")
    print(f"Wrote {OUT_DIR}/per_metric_breakdown_per_model_1.png")
    print(f"Wrote {OUT_DIR}/per_metric_breakdown_per_model_2.png")
    print(f"Wrote {OUT_DIR}/per_metric_breakdown_by_pressure.png")
    for fn in per_model_pressure_files:
        print(f"Wrote {OUT_DIR}/{fn}")

if __name__ == "__main__":
    main()

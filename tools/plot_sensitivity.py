"""Figures for results/sensitivity_lia.json.

    python tools/plot_sensitivity.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN = "results/sensitivity_lia.json"
OUT_DIR = "results/lia_1700_2017"

PARAM_LABEL = {
    "melt_factor": "Melt factor",
    "precip": "Precipitation",
    "tau_ref": "$\\tau_{ref}$ (sliding)",
}
COLORS = {
    "melt_factor": "#d62728",
    "precip": "#1f77b4",
    "tau_ref": "#9467bd",
}


def main():
    d = json.load(open(IN))
    rows = d["perturbations"]
    T_ref = d["T_ref"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # ---- panel 1: dT-equivalent bias vs fractional parameter error ----
    ax = axes[0]
    for name in PARAM_LABEL:
        pts = sorted([(r["fraction"], r["dT_equivalent"]) for r in rows
                     if r["parameter"] == name])
        xs = [p[0] * 100 for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", color=COLORS[name], label=PARAM_LABEL[name],
                linewidth=2, markersize=6)
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Parameter error (%)")
    ax.set_ylabel("Bias on the inverted temperature (°C)")
    ax.set_title(f"Sensitivity of the inverted $T_{{lowest}}$\n"
                f"reference: $T$={T_ref:.1f}°C, 1700–2017 (5 epochs)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # ---- panel 2: bar chart at +/-10% for a quick read ----
    ax = axes[1]
    names = list(PARAM_LABEL)
    minus10 = [next(r["dT_equivalent"] for r in rows
                    if r["parameter"] == n and abs(r["fraction"] + 0.10) < 1e-6)
              for n in names]
    plus10 = [next(r["dT_equivalent"] for r in rows
                   if r["parameter"] == n and abs(r["fraction"] - 0.10) < 1e-6)
             for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, minus10, w, label="parameter −10%",
          color=[COLORS[n] for n in names], alpha=0.55)
    ax.bar(x + w / 2, plus10, w, label="parameter +10%",
          color=[COLORS[n] for n in names], alpha=1.0)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PARAM_LABEL[n] for n in names], rotation=15, ha="right")
    ax.set_ylabel("Bias on the inverted temperature (°C)")
    ax.set_title("A ±10% parameter error, converted to °C")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.savefig(f"{OUT_DIR}/sensitivity_parameters.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/sensitivity_parameters.png")

    # ---- panel 3: mean IoU vs parameter, to show extent is a weak constraint ----
    fig2, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    for name in PARAM_LABEL:
        pts = sorted([(r["fraction"], r["mean_iou"]) for r in rows
                     if r["parameter"] == name])
        xs = [p[0] * 100 for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", color=COLORS[name], label=PARAM_LABEL[name])
    baseline = next(r["mean_iou"] for r in rows if r["parameter"] == "baseline")
    ax.axhline(baseline, color="k", linestyle="--", linewidth=1, alpha=0.6,
              label=f"baseline ({baseline:.3f})")
    ax.set_xlabel("Parameter error (%)")
    ax.set_ylabel("Mean IoU over 5 epochs")
    ax.set_title("Extent misfit is a shallow function of every parameter\n"
                "(IoU rewards bulk overlap, not terminus position)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig2.savefig(f"{OUT_DIR}/sensitivity_iou_flatness.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/sensitivity_iou_flatness.png")


if __name__ == "__main__":
    main()

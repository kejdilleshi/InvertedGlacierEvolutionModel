"""Comparison figures across the three inversion attempts.

    V1  results/lia_1700_2017  nodes,     flat start,     unmasked loss, 60 iters
    V2  results/lia_v2         nodes,     implied start,  masked loss,   15 iters
    V3  results/lia_v3         level+dev, flat start,     masked loss,   18 iters

    python tools/plot_v2_results.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = [("V1", "results/lia_1700_2017", "#1f77b4"),
        ("V2", "results/lia_v2", "#2ca02c"),
        ("V3", "results/lia_v3", "#9467bd")]
OUT = "results/lia_v2"
EPOCHS = [1880, 1926, 1957, 1980, 2017]


def implied_curve():
    scan = sorted(json.load(open("results/sensitivity_lia.json"))["scan"],
                  key=lambda s: s["T"])
    Ts = np.array([s["T"] for s in scan])
    out = {}
    for y in EPOCHS:
        sim = np.array([s["per_epoch"][str(y)]["terminus"] for s in scan])
        obs = scan[0]["per_epoch"][str(y)]["obs_terminus"]
        out[y] = (float(Ts[0] - (sim[0] - obs) * (Ts[1] - Ts[0]) / (sim[1] - sim[0]))
                  if obs < sim.min() else float(np.interp(obs, sim, Ts)))
    return out


def main():
    imp = implied_curve()
    iy, iv = sorted(imp), [imp[y] for y in sorted(imp)]
    span = imp[2017] - imp[1880]
    runs = {}
    for name, d, c in RUNS:
        p = f"{d}/inversion_history.json"
        if os.path.exists(p):
            runs[name] = (json.load(open(p)), c)
        else:
            print(f"skipping {name}: no {p}")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10), constrained_layout=True)

    # --- histories vs target ---
    ax = axes[0, 0]
    ax.plot(iy, iv, "o-", color="#d62728", linewidth=3, markersize=10,
            label="implied by observations", zorder=5)
    for name, (r, c) in runs.items():
        ax.plot(r["node_times"], r["best"]["T"], "s-", color=c, linewidth=2,
                markersize=5,
                label=f"{name} ({len(r['history']['loss'])} iters)")
    ax.axvspan(1500, 1700, color="grey", alpha=0.12)
    ax.set_xlabel("Year"); ax.set_ylabel("$T_{lowest}$ (°C)")
    ax.set_title("Inverted temperature history vs the implied target")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # --- trend recovered ---
    ax = axes[0, 1]
    labels = ["implied\nby data"] + list(runs)
    vals = [span] + [max(r["best"]["T"]) - min(r["best"]["T"])
                     for r, _ in runs.values()]
    cols = ["#d62728"] + [c for _, c in runs.values()]
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f} °C\n({v/span*100:.0f}%)",
                    (b.get_x() + b.get_width()/2, v), ha="center", va="bottom",
                    fontsize=10)
    ax.set_ylabel("Warming across the control (°C)")
    ax.set_ylim(0, span * 1.35)
    ax.set_title("Trend recovered")
    ax.grid(alpha=0.3, axis="y")

    # --- the key panel: how the spread (trend) develops per iteration ---
    ax = axes[1, 0]
    for name, (r, c) in runs.items():
        T = np.array(r["history"]["T"])
        ax.plot((T.max(axis=1) - T.min(axis=1)) / span * 100, color=c,
                linewidth=2.2, label=name)
    ax.axhline(100, color="#d62728", linestyle="--", linewidth=1.5,
               label="required trend")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Trend recovered (% of implied)")
    ax.set_title("V1 adds no structure for ~20 iterations, then tilts.\n"
                 "V2 starts with the trend and flattens it. V3 splits the two.")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # --- mean level per iteration, the other half of the story ---
    ax = axes[1, 1]
    for name, (r, c) in runs.items():
        T = np.array(r["history"]["T"])
        ax.plot(T.mean(axis=1), color=c, linewidth=2.2, label=name)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean of the control (°C)")
    ax.set_title("Mean level. Every run moves this first;\n"
                 "the trend only develops once it settles.")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.savefig(f"{OUT}/v2_comparison.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}/v2_comparison.png")

    # --- extent maps, one row per run ---
    have = [(n, d) for n, d, _ in RUNS if os.path.exists(f"{d}/final_fields.npz")]
    fig2, axes2 = plt.subplots(len(have), len(EPOCHS),
                               figsize=(3.2 * len(EPOCHS), 3.0 * len(have)),
                               constrained_layout=True, squeeze=False)
    for row, (name, d) in enumerate(have):
        npz = np.load(f"{d}/final_fields.npz")
        x, y = npz["x"] / 1000, npz["y"] / 1000
        ext = [x.min(), x.max(), y.min(), y.max()]
        for col, e in enumerate(EPOCHS):
            ax = axes2[row][col]
            sim, obs = npz[f"sim_{e}"] > 1, npz[f"obs_{e}"] > 1
            rgb = np.zeros((*sim.shape, 3))
            rgb[obs & ~sim] = [0.20, 0.45, 0.85]
            rgb[sim & ~obs] = [0.85, 0.25, 0.20]
            rgb[sim & obs] = [0.55, 0.55, 0.55]
            ax.imshow(rgb, origin="lower", extent=ext, aspect="equal")
            iou = (sim & obs).sum() / max((sim | obs).sum(), 1)
            ax.set_title(f"{name} {e}\nIoU {iou:.3f}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               ([0.55, 0.55, 0.55], [0.20, 0.45, 0.85], [0.85, 0.25, 0.20])]
    fig2.legend(handles, ["agree", "observed only", "simulated only"],
                loc="lower center", ncol=3, fontsize=10,
                bbox_to_anchor=(0.5, -0.02))
    fig2.suptitle("Extent fit at the best iteration. IoU is whole-domain for "
                  "every row, so the rows compare.", fontsize=12)
    fig2.savefig(f"{OUT}/v2_extent_comparison.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}/v2_extent_comparison.png")


if __name__ == "__main__":
    main()

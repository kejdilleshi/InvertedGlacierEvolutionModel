"""Figures for results/lia_1700_2017/inversion_history.json + final_fields.npz.

    python tools/plot_lia_inversion.py [results_dir]
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = "results/lia_1700_2017"


def main(DIR=DIR):
    d = json.load(open(f"{DIR}/inversion_history.json"))
    h = d["history"]
    epochs = d.get("epochs") or [int(y) for y in h["iou"]]
    node_times = d["node_times"]
    n_iter = len(h["loss"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    # ---- loss curve ----
    ax = axes[0, 0]
    ax.plot(h["loss"], label="total loss", color="#1f77b4", linewidth=2)
    ax.plot(h["data"], label="data term", color="#d62728", linewidth=1.5,
           linestyle="--")
    best_i = d["best"]["iter"]
    ax.axvline(best_i, color="k", linestyle=":", linewidth=1,
              label=f"best (iter {best_i})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ---- IoU per epoch over iterations ----
    ax = axes[0, 1]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    for y, c in zip(epochs, cmap):
        ax.plot(h["iou"][str(y)], label=str(y), color=c, linewidth=1.8)
    ax.axvline(best_i, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("IoU")
    ax.set_title("Per-epoch extent overlap")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ---- T(t) evolution: a few snapshots + final ----
    ax = axes[1, 0]
    T = np.array(h["T"])  # (n_iter, n_nodes)
    snap_idx = sorted(set([0, n_iter // 4, n_iter // 2, best_i, n_iter - 1]))
    cmap2 = plt.cm.plasma(np.linspace(0, 0.85, len(snap_idx)))
    for i, c in zip(snap_idx, cmap2):
        label = f"iter {i}" + (" (best)" if i == best_i else "")
        ax.plot(node_times, T[i], "o-", color=c, label=label, markersize=4,
               linewidth=1.8 if i in (best_i, n_iter - 1) else 1.0,
               alpha=1.0 if i in (best_i, n_iter - 1) else 0.6)
    ax.axvspan(1600, 1700, color="grey", alpha=0.15,
              label="warm-up (unconstrained)")
    for y in epochs:
        ax.axvline(y, color="k", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Year")
    ax.set_ylabel("$T_{lowest}$ (°C)")
    ax.set_title("Inverted temperature history")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # ---- gradient norm / step size ----
    ax = axes[1, 1]
    ax.plot(h["grad_norm"], color="#2ca02c")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("$||\\nabla_T \\, loss||$")
    ax.set_title("Gradient norm")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    fig.suptitle(f"LIA extent inversion, {d['settings']['t_start']:.0f}–"
                f"2017, {len(node_times)} control nodes, {n_iter} iterations",
                fontsize=12)
    fig.savefig(f"{DIR}/inversion_convergence.png", dpi=150, bbox_inches="tight")
    print(f"wrote {DIR}/inversion_convergence.png")

    # ---- final extent comparison maps ----
    npz = np.load(f"{DIR}/final_fields.npz")
    x, y = npz["x"] / 1000, npz["y"] / 1000
    ext = [x.min(), x.max(), y.min(), y.max()]
    fig2, axes2 = plt.subplots(1, len(epochs), figsize=(4 * len(epochs), 5.5),
                               constrained_layout=True)
    # cells the loss ignores (weight 0) are drawn in dark blue-grey and left out
    # of the IoU, so the figure shows what the inversion actually scored
    mask_file = os.path.join(DIR, "analysis_mask.npy")
    w = np.load(mask_file) if os.path.exists(mask_file) else None
    for ax, year in zip(axes2, epochs):
        sim = npz[f"sim_{year}"] > 1.0
        obs = npz[f"obs_{year}"] > 1.0
        rgb = np.zeros((*sim.shape, 3))
        rgb[obs & ~sim] = [0.20, 0.45, 0.85]   # obs only: blue
        rgb[sim & ~obs] = [0.85, 0.25, 0.20]   # sim only: red
        rgb[sim & obs] = [0.55, 0.55, 0.55]    # agreement: grey
        if w is not None:
            excluded = (w == 0) & (sim | obs)
            rgb[excluded] = [0.15, 0.17, 0.30]
            sim, obs = sim & (w > 0), obs & (w > 0)
        ax.imshow(rgb, origin="lower", extent=ext, aspect="equal")
        inter, union = (sim & obs).sum(), (sim | obs).sum()
        iou = inter / union if union else 0.0
        ax.set_title(f"{year}\nIoU {iou:.2f}" + (" (scored area)" if w is not None else ""),
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    colors = [[0.55, 0.55, 0.55], [0.20, 0.45, 0.85], [0.85, 0.25, 0.20]]
    if w is not None:
        colors.append([0.15, 0.17, 0.30])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    labels = ["agree", "observed only", "simulated only"]
    if w is not None:
        labels.append("excluded from loss")
    fig2.legend(handles, labels,
               loc="lower center", ncol=len(labels), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig2.suptitle("Best-fit extent vs observed, at convergence", fontsize=12)
    fig2.savefig(f"{DIR}/extent_comparison.png", dpi=150, bbox_inches="tight")
    print(f"wrote {DIR}/extent_comparison.png")


if __name__ == "__main__":
    main(*sys.argv[1:2])

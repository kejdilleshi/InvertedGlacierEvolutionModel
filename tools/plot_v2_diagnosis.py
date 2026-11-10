"""Diagnostic figures for the V2 notebook: why did V1 barely move?

    python tools/plot_v2_diagnosis.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "results/lia_v2"
EPOCHS = [1880, 1926, 1957, 1980, 2017]
ELA_MAX = 2944.0          # ELA at T = 8.0 C, the warmest in the scan
IMPLIED_T = {1880: 6.14, 1926: 6.64, 1957: 6.86, 1980: 7.14, 2017: 7.50}


def implied_curve():
    """Invert the constant-T terminus curve, epoch by epoch."""
    d = json.load(open("results/sensitivity_lia.json"))
    scan = sorted(d["scan"], key=lambda s: s["T"])
    Ts = np.array([s["T"] for s in scan])
    out = {}
    for y in EPOCHS:
        sim = np.array([s["per_epoch"][str(y)]["terminus"] for s in scan])
        obs = scan[0]["per_epoch"][str(y)]["obs_terminus"]
        if obs < sim.min():
            slope = (Ts[1] - Ts[0]) / (sim[1] - sim[0])
            out[y] = float(Ts[0] - (sim[0] - obs) * slope)
        else:
            out[y] = float(np.interp(obs, sim, Ts))
    return out


def fig_implied_vs_inverted():
    imp = implied_curve()
    run = json.load(open("results/lia_1700_2017/inversion_history.json"))
    T_v1 = run["best"]["T"]
    node_times = run["node_times"]
    n_iter = len(run["history"]["loss"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax = axes[0]
    ys = sorted(imp)
    ax.plot(ys, [imp[y] for y in ys], "o-", color="#d62728", linewidth=2.5,
            markersize=9, label="implied by the observed terminus", zorder=3)
    ax.plot(node_times, T_v1, "s-", color="#1f77b4", linewidth=2, markersize=6,
            label=f"V1 inversion, best of {n_iter} iters")
    ax.axhline(7.0, color="grey", linestyle=":", linewidth=1.2,
               label="V1 starting guess (flat 7.0 C)")
    ax.set_xlabel("Year")
    ax.set_ylabel("$T_{lowest}$ (°C)")
    ax.set_xlim(1480, 2030)
    span = imp[2017] - imp[1880]
    got = max(T_v1) - min(T_v1)
    ax.set_title(f"The data demand +{span:.2f} °C of warming.\n"
                 f"V1 produced {got:.2f} °C — {got/span*100:.0f}% of it.")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(len(EPOCHS))
    ax.bar(x - 0.2, [imp[y] for y in EPOCHS], 0.4, color="#d62728",
           label="implied by observations")
    v1_at = np.interp(EPOCHS, node_times, T_v1)
    ax.bar(x + 0.2, v1_at, 0.4, color="#1f77b4", label="V1 inversion")
    for i, (a, b) in enumerate(zip([imp[y] for y in EPOCHS], v1_at)):
        ax.annotate(f"{b-a:+.2f}", (i, max(a, b) + 0.05), ha="center", fontsize=9,
                    color="#444")
    ax.set_xticks(x); ax.set_xticklabels(EPOCHS)
    ax.set_ylim(5.5, 8.0)
    ax.set_ylabel("$T_{lowest}$ (°C)")
    ax.set_title("Per-epoch error (labelled): V1 is ~1.2 °C too warm\n"
                 "at the LIA maximum and slightly too cold at 2017")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f"{OUT_DIR}/implied_vs_inverted.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/implied_vs_inverted.png")


def fig_misfit_anatomy(npz_path="results/lia_1700_2017/final_fields.npz"):
    npz = np.load(npz_path)
    topg, x, y = npz["topg"], npz["x"] / 1000, npz["y"] / 1000
    ext = [x.min(), x.max(), y.min(), y.max()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), constrained_layout=True)

    # --- map: fixable vs unfixable misfit at 1880 ---
    ax = axes[0]
    dis = (npz["sim_1880"] > 1) ^ (npz["obs_1880"] > 1)
    rgb = np.zeros((*dis.shape, 3))
    rgb[(topg > 200)] = [0.12, 0.12, 0.14]
    rgb[dis & (topg <= ELA_MAX)] = [0.20, 0.70, 0.30]
    rgb[dis & (topg > ELA_MAX)] = [0.90, 0.20, 0.20]
    ax.imshow(rgb, origin="lower", extent=ext, aspect="equal")
    n_fix = int((dis & (topg <= ELA_MAX)).sum())
    n_un = int((dis & (topg > ELA_MAX)).sum())
    ax.set_title(f"1880 extent misfit\n"
                 f"green = reducible ({n_fix}), red = above every ELA ({n_un})",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # --- elevation histogram of misfit ---
    ax = axes[1]
    allz = []
    for e in EPOCHS:
        d = (npz[f"sim_{e}"] > 1) ^ (npz[f"obs_{e}"] > 1)
        allz.append(topg[d])
    allz = np.concatenate(allz)
    ax.hist(allz, bins=40, color="#777", edgecolor="white", linewidth=0.4)
    ax.axvline(ELA_MAX, color="#d62728", linewidth=2,
               label=f"ELA at T=8.0 °C ({ELA_MAX:.0f} m)")
    ax.axvline(3000, color="#1f77b4", linestyle="--", linewidth=2,
               label="V2 analysis cut (3000 m)")
    frac = (allz > ELA_MAX).mean() * 100
    ax.set_xlabel("Bed elevation of the disagreeing cell (m)")
    ax.set_ylabel("Cells (all 5 epochs)")
    ax.set_title(f"{frac:.0f}% of the misfit sits above the warmest ELA\n"
                 "— no temperature the inversion can try will move it", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # --- per-epoch split ---
    ax = axes[2]
    fix, unfix = [], []
    for e in EPOCHS:
        d = (npz[f"sim_{e}"] > 1) ^ (npz[f"obs_{e}"] > 1)
        fix.append(int((d & (topg <= ELA_MAX)).sum()))
        unfix.append(int((d & (topg > ELA_MAX)).sum()))
    xs = np.arange(len(EPOCHS))
    ax.bar(xs, fix, 0.6, color="#2ca02c", label="reducible by T")
    ax.bar(xs, unfix, 0.6, bottom=fix, color="#d62728", label="irreducible")
    for i, (f_, u_) in enumerate(zip(fix, unfix)):
        ax.annotate(f"{u_/(f_+u_)*100:.0f}%", (i, f_ + u_ + 12), ha="center",
                    fontsize=9, color="#d62728")
    ax.set_xticks(xs); ax.set_xticklabels(EPOCHS)
    ax.set_ylabel("Disagreeing cells")
    ax.set_title("1880 — the epoch anchoring the cold end —\n"
                 "is the one most polluted by irreducible misfit", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.savefig(f"{OUT_DIR}/misfit_anatomy.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/misfit_anatomy.png")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_implied_vs_inverted()
    fig_misfit_anatomy()

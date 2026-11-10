"""Overview figure of the prepared Argentière data (bed, ice masks, constraints).

    python tools/plot_argentiere_data.py [--dx 100]

Writes ``data/Argentiere/argentiere_overview.png``.
"""

import argparse

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource

KIND_STYLE = {
    "lateral": dict(marker="^", color="#d62728"),
    "frontal": dict(marker="s", color="#ff7f0e"),
    "icefree": dict(marker="o", color="#1f77b4"),
    "icefree_any": dict(marker="D", color="#9467bd"),
    "covered_any": dict(marker="x", color="#9467bd"),
}


def overview(nc_path, csv_path, out=None, ax=None):
    nc = netCDF4.Dataset(nc_path)
    x, y = nc["x"][:], nc["y"][:]
    bed = np.asarray(nc["topg"][:])
    ext = [x[0] - 50, x[-1] + 50, y[0] - 50, y[-1] + 50]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 9.5))
    hs = LightSource(azdeg=180, altdeg=35).hillshade(bed[::-1], vert_exag=1.5, dx=x[1] - x[0], dy=x[1] - x[0])
    ax.imshow(hs, cmap="gray", extent=ext, alpha=0.9)
    ax.contour(x, y, bed, levels=np.arange(1000, 4800, 250), colors="k",
               linewidths=0.25, alpha=0.5)
    for name, col, lw in (("catchment_lateglacial", "#2ca02c", 1.5),
                          ("catchment_holocene", "#17becf", 1.0),
                          ("lia_arve", "#1f77b4", 0.8),
                          ("argentiere_2012", "#000000", 0.8)):
        ax.contour(x, y, np.asarray(nc[name][:]), levels=[0.5], colors=col, linewidths=lw)
    nc.close()
    df = pd.read_csv(csv_path)
    df = df[df.in_catchment]
    for kind, st in KIND_STYLE.items():
        d = df[df.kind == kind]
        if len(d):
            ax.scatter(d.x, d.y, s=28, label=f"{kind} ({len(d)})", edgecolor="k",
                       linewidths=0.4, zorder=5, **st)
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=c, lw=lw) for c, lw in
                (("#2ca02c", 1.5), ("#17becf", 1.0), ("#1f77b4", .8), ("k", .8))]
    labels += ["ice allowed, Lateglacial", "ice allowed, Holocene", "LIA extent",
               "Argentière 2012"]
    ax.legend(handles, labels, loc="lower right", fontsize=7, framealpha=0.85)
    ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
    ax.set_xlabel("easting (m, UTM 32N)"); ax.set_ylabel("northing (m)")
    ax.set_title("Upper Arve: bed, ice masks and dated constraints")
    ax.ticklabel_format(style="plain")
    if own:
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
    return ax


if __name__ == "__main__":
    plt.switch_backend("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=int, default=100)
    a = ap.parse_args()
    out = "data/Argentiere/argentiere_overview.png"
    overview(f"data/Argentiere/argentiere_{a.dx}m.nc",
             f"data/Argentiere/argentiere_observations_{a.dx}m.csv", out)
    print("wrote", out)

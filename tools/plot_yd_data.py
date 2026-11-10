"""Inspection figure for the prepared YD-Aletsch dataset.

    python tools/plot_yd_data.py     ->  data/YD-Aletsch/aletsch_yd_overview.png
"""

import matplotlib
matplotlib.use("Agg")

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize

NC = "data/YD-Aletsch/aletsch_yd_100m.nc"
OUT = "data/YD-Aletsch/aletsch_yd_overview.png"
SEGMENTS = [("YD2", "#e41a1c"), ("YD1", "#ff7f00"),
            ("LIA_W", "#377eb8"), ("LIA_E", "#4daf4a")]


def main():
    nc = netCDF4.Dataset(NC)
    x = np.asarray(nc["x"][:]) / 1000.0
    y = np.asarray(nc["y"][:]) / 1000.0
    topg = np.asarray(nc["topg"][:])
    thk = np.asarray(nc["thk"][:])
    catch = np.asarray(nc["catchment"][:])
    ext = [x.min(), x.max(), y.min(), y.max()]

    shade = LightSource(315, 45).hillshade(topg, vert_exag=2, dx=100, dy=100)
    norm = Normalize(800, 4200)
    rgb = plt.cm.terrain(norm(topg))[..., :3] * (0.35 + 0.65 * shade[..., None])

    fig, axes = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)

    for ax in axes[:2]:
        ax.imshow(np.clip(rgb, 0, 1), origin="lower", extent=ext, aspect="equal")
        ax.contour(x, y, thk, levels=[1.0], colors="w", linewidths=0.8)
        outside = np.ma.masked_where(catch > 0.5, np.ones_like(catch))
        ax.imshow(outside, origin="lower", extent=ext, aspect="equal",
                  cmap="gray_r", vmin=0, vmax=1.6, alpha=0.55)
        ax.set_xlabel("Easting (km)")
    axes[0].set_ylabel("Northing (km)")
    axes[1].set_xlim(421.8, 430.2)
    axes[1].set_ylim(5135.2, 5143.0)

    for name, colour in SEGMENTS:
        m = np.asarray(nc[f"moraine_{name}"][:]) > 0.5
        z = np.asarray(nc[f"z_cell_{name}"][:])
        iy, ix = np.nonzero(m)
        v = nc[f"moraine_{name}"]
        age = (f"{v.age_min_yr_BP:.0f}–{v.age_max_yr_BP:.0f} yr BP"
               if hasattr(v, "age_min_yr_BP") else "undated")
        for ax, size in ((axes[0], 9), (axes[1], 26)):
            ax.scatter(x[ix], y[iy], s=size, c=colour, edgecolors="none",
                       label=f"{name}  ({age})")
        axes[2].scatter(y[iy], z[iy, ix], s=14, c=colour, label=name)

    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[0].set_title("Dated moraines on the 100 m model grid\n"
                      "white = modern ice, grey = outside the catchment mask",
                      fontsize=11)
    axes[1].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[1].set_title("Tongue and terminus, zoomed", fontsize=11)

    axes[2].set_xlabel("Northing (km)")
    axes[2].set_ylabel("Moraine elevation (m a.s.l.)")
    axes[2].set_title("The inversion target\n"
                      "ice surface must reach these heights, scored "
                      f"{nc.inboard_offset_cells} cells inboard "
                      f"($\\sigma$ = {nc.obs_operator_sigma_m:.0f} m)",
                      fontsize=11)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=9)
    axes[2].set_box_aspect(1.4)

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    nc.close()
    print("wrote", OUT)


if __name__ == "__main__":
    main()

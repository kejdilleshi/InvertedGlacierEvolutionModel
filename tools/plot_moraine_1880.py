"""Do the LIA moraines match the 1880 reconstruction in geology.nc?

    python tools/plot_moraine_1880.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib.colors import LightSource, Normalize
from scipy import ndimage

OUT = "results/lia_1700_2017/moraine_vs_1880.png"


def main():
    g = netCDF4.Dataset("data/geology.nc")
    gx, gy = np.asarray(g["x"][:]), np.asarray(g["y"][:])
    topg = np.asarray(g["topg"][:])
    years = [1880, 1926, 1957, 1980, 1999, 2009, 2017]
    thk = {y: np.asarray(g[f"surf_{y}"][:]) - topg for y in years}
    g.close()

    nc = netCDF4.Dataset("data/YD-Aletsch/aletsch_yd_100m.nc")
    lia = ((np.asarray(nc["moraine_LIA_W"][:]) > 0.5) |
          (np.asarray(nc["moraine_LIA_E"][:]) > 0.5))
    nc.close()

    def margin(ice):
        return ice & ~ndimage.binary_erosion(ice, np.ones((3, 3)))

    iy_m, ix_m = np.nonzero(lia)
    Em, Nm = gx[ix_m], gy[iy_m]
    dists = {}
    for y in years:
        mg = margin(thk[y] > 1)
        iy, ix = np.nonzero(mg)
        d = np.hypot(gx[ix][:, None] - Em[None, :], gy[iy][:, None] - Nm[None, :]).min(0)
        dists[y] = d

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)

    # ---- map: 1880 margin vs moraines ----
    ax = axes[0]
    shade = LightSource(315, 45).hillshade(topg, vert_exag=2, dx=100, dy=100)
    norm = Normalize(1400, 3200)
    rgb = plt.cm.terrain(norm(topg))[..., :3] * (0.35 + 0.65 * shade[..., None])
    ext = [gx.min() / 1000, gx.max() / 1000, gy.min() / 1000, gy.max() / 1000]
    ax.imshow(np.clip(rgb, 0, 1), origin="lower", extent=ext, aspect="equal")
    ax.contour(gx / 1000, gy / 1000, thk[1880] > 1, levels=[0.5],
              colors="cyan", linewidths=1.5, linestyles="-")
    ax.scatter(Em / 1000, Nm / 1000, s=8, c="red", edgecolors="none")
    ax.set_xlim(421, 430.5)
    ax.set_ylim(5135, 5143)
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_title("LIA moraines (red) vs the 1880 ice margin (cyan)\n"
                "geology.nc, surf_1880 − topg > 1 m", fontsize=11)
    ax.plot([], [], color="cyan", label="1880 margin")
    ax.scatter([], [], c="red", s=20, label="LIA moraine")
    ax.legend(loc="upper left", fontsize=9)

    # ---- distance from moraine to each dated margin ----
    ax = axes[1]
    meds = [np.median(dists[y]) for y in years]
    p90s = [np.percentile(dists[y], 90) for y in years]
    x = np.arange(len(years))
    ax.bar(x, p90s, color="#ffb3b3", label="90th percentile")
    ax.bar(x, meds, color="#d62728", label="median")
    ax.axhline(100, color="k", linestyle=":", linewidth=1, label="1 grid cell (100 m)")
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=0)
    ax.set_ylabel("Moraine → dated margin distance (m)")
    ax.set_title("The moraines date the 1880 stand specifically\n"
                "(distance grows monotonically for later epochs)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)
    for y in years:
        print(f"  {y}: median {np.median(dists[y]):.0f} m  "
              f"p90 {np.percentile(dists[y], 90):.0f} m  "
              f"within 150m: {(dists[y] < 150).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()

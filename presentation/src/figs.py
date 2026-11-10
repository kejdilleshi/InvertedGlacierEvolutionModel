"""Map figures for the LIA inversion talk, drawn from the repo's cached data."""
import os
import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = "/home/klleshi/Documents/InvertedGlacierEvolutionModel"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT, exist_ok=True)

NAVY, SLATE, ICE = "#102A3A", "#2E5266", "#EAF2F6"
BLUE, EMBER, MELT, GREY = "#2B7BB9", "#D0532A", "#8DC3DE", "#8A9AA5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})

nc = netCDF4.Dataset(f"{REPO}/data/geology.nc")
x = np.asarray(nc["x"][:]) / 1000.0
y = np.asarray(nc["y"][:]) / 1000.0
topg = np.asarray(nc["topg"][:], dtype=float)
surf = {yr: np.asarray(nc[f"surf_{yr}"][:], dtype=float)
        for yr in (1880, 1926, 1957, 1980, 1999, 2009, 2017)}
nc.close()
thk = {yr: s - topg for yr, s in surf.items()}
ext = [x[0] - 0.05, x[-1] + 0.05, y[0] - 0.05, y[-1] + 0.05]
flip = y[0] > y[-1]
origin = "upper" if flip else "lower"

ls = LightSource(azdeg=315, altdeg=40)


def shade(z, vert=2.0):
    return ls.hillshade(z if not flip else z, vert_exag=vert, dx=100, dy=100)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scalebar(ax, x0, y0, km=2, color="k"):
    ax.plot([x0, x0 + km], [y0, y0], color=color, lw=3, solid_capstyle="butt")
    ax.text(x0 + km / 2, y0 + 0.25, f"{km} km", color=color, ha="center",
            va="bottom", fontsize=12)


# ---------------------------------------------------------------- title art
hs = shade(surf[2017], 2.5)
ice = thk[2017] > 1
rgb = np.zeros(hs.shape + (3,))
dark = np.array([0x10, 0x2A, 0x3A]) / 255
rock = np.array([0x3A, 0x5A, 0x6C]) / 255
icec = np.array([0xCF, 0xE6, 0xF2]) / 255
for c in range(3):
    rock_c = dark[c] + (rock[c] - dark[c]) * hs
    ice_c = icec[c] * (0.55 + 0.45 * hs)
    rgb[..., c] = np.where(ice, ice_c, rock_c)
h_in = 7.5
w_in = h_in * len(x) / len(y)
fig = plt.figure(figsize=(w_in, h_in), dpi=220)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(rgb, extent=ext, origin=origin, interpolation="bilinear", aspect="auto")
bare(ax)
fig.savefig(f"{OUT}/title_art.png", dpi=220)
plt.close(fig)

# ------------------------------------------------------- site map + record
EP = [1880, 1926, 1957, 1980, 2017]
ep_cols = ["#0B3A5C", "#2B7BB9", "#5FA3CF", "#E0A43A", EMBER]
hs = shade(topg, 2.0)
fig = plt.figure(figsize=(w_in * 0.82, h_in * 0.82), dpi=220)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(hs, cmap="Greys_r", vmin=-0.4, vmax=1.15, extent=ext, origin=origin,
          interpolation="bilinear", aspect="auto")
ax.imshow(np.where(thk[1880] > 1, 1.0, np.nan), cmap=ListedColormap([ICE]),
          alpha=0.75, extent=ext, origin=origin, aspect="auto")
yy = y if not flip else y
from scipy import ndimage


def clean(m, min_cells=12):
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.where(sizes >= min_cells)[0])


for yr, col in zip(EP, ep_cols):
    ax.contour(x, yy, clean(thk[yr] > 1).astype(float), levels=[0.5], colors=[col],
               linewidths=1.6)
handles = [Line2D([0], [0], color=c, lw=3, label=str(yr)) for yr, c in zip(EP, ep_cols)]
ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=13,
          title="ice margin", title_fontsize=12, facecolor="white", edgecolor="none",
          framealpha=0.92, borderpad=0.6)
scalebar(ax, x[-1] - 3.6, y.min() + 0.9)
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(min(ext[2:]), max(ext[2:]))
bare(ax)
fig.savefig(f"{OUT}/site_map.png", dpi=220)
plt.close(fig)

# ------------------------------------------------- LIA moraine vs 1880 margin
yd = netCDF4.Dataset(f"{REPO}/data/YD-Aletsch/aletsch_yd_100m.nc")
mor = (np.asarray(yd["moraine_LIA_W"][:]) > 0) | (np.asarray(yd["moraine_LIA_E"][:]) > 0)
yd.close()
X, Y = np.meshgrid(x, y)
xlim, ylim = (421.2, 430.4), (5135.4, 5143.0)
fig = plt.figure(figsize=(6.2, 6.2 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])), dpi=220)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(hs, cmap="Greys_r", vmin=-0.4, vmax=1.15, extent=ext, origin=origin,
          interpolation="bilinear", aspect="auto")
ax.imshow(np.where(thk[1880] > 1, 1.0, np.nan), cmap=ListedColormap([MELT]),
          alpha=0.55, extent=ext, origin=origin, aspect="auto")
ax.contour(x, y, (thk[1880] > 1).astype(float), levels=[0.5], colors=[BLUE], linewidths=2.2)
ax.contour(x, y, (thk[2017] > 1).astype(float), levels=[0.5], colors=[GREY],
           linewidths=1.6, linestyles="--")
ax.scatter(X[mor], Y[mor], s=16, c=EMBER, edgecolors="white", linewidths=0.4, zorder=5)
ax.set_xlim(*xlim); ax.set_ylim(*ylim)
handles = [Patch(facecolor=MELT, edgecolor=BLUE, lw=2, label="1880 ice (historical DEM)"),
           Line2D([0], [0], color=GREY, lw=2, ls="--", label="2017 margin"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=EMBER, markersize=9,
                  label="mapped LIA moraine")]
ax.legend(handles=handles, loc="upper left", fontsize=12, facecolor="white",
          edgecolor="none", framealpha=0.92)
scalebar(ax, 427.6, 5135.8)
bare(ax)
fig.savefig(f"{OUT}/moraine_1880.png", dpi=220)
plt.close(fig)

# -------------------------------------------------------- extent fit, 5 epochs
ff = np.load(f"{REPO}/results/lia_1700_2017/final_fields.npz")
rows = np.where((thk[1880] > 1).any(axis=1))[0]
cols = np.where((thk[1880] > 1).any(axis=0))[0]
r0, r1 = max(rows.min() - 3, 0), min(rows.max() + 4, len(y))
c0, c1 = max(cols.min() - 3, 0), min(cols.max() + 4, len(x))
sub_ext = [x[c0] - 0.05, x[c1 - 1] + 0.05,
           (y[r1 - 1] if flip else y[r0]) - 0.05, (y[r0] if flip else y[r1 - 1]) + 0.05]
cmap = ListedColormap(["#BFD9E8", BLUE, EMBER])
pw = (c1 - c0) / (r1 - r0)
fig, axes = plt.subplots(1, 5, figsize=(13.5, 13.5 / 5 * 1 / pw + 0.9), dpi=200)
ious = {}
for ax, yr in zip(axes, EP):
    sim, obs = ff[f"sim_{yr}"] > 1, ff[f"obs_{yr}"] > 1
    ious[yr] = (sim & obs).sum() / (sim | obs).sum()
    cls = np.full(sim.shape, np.nan)
    cls[sim & obs] = 0; cls[obs & ~sim] = 1; cls[sim & ~obs] = 2
    ax.imshow(hs[r0:r1, c0:c1], cmap="Greys_r", vmin=-0.6, vmax=1.3, extent=sub_ext,
              origin=origin, aspect="auto", interpolation="bilinear")
    ax.imshow(cls[r0:r1, c0:c1], cmap=cmap, vmin=-0.5, vmax=2.5, extent=sub_ext,
              origin=origin, aspect="auto", interpolation="nearest")
    ax.set_title(f"{yr}", fontsize=17, fontweight="bold", color=NAVY, pad=4)
    ax.text(0.5, -0.035, f"IoU {ious[yr]:.2f}", transform=ax.transAxes, ha="center",
            va="top", fontsize=15, color=SLATE)
    bare(ax)
fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.07, wspace=0.04)
fig.savefig(f"{OUT}/extent_fit.png", dpi=200)
plt.close(fig)
print("IoU", {k: round(float(v), 3) for k, v in ious.items()})
print("title_art inches", w_in, h_in)

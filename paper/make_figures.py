"""Figures of the manuscript, from the archived runs.

    cd <repo root>; IGEM_DEVICE=cpu /home/klleshi/miniconda3/envs/IGEM/bin/python paper/make_figures.py [fig02 fig03 ...]

Figure 1 (the schematic) is TikZ: paper/figures/fig01_schematic.tex.

Reads (nothing is re-run):
  data/Argentiere/argentiere_100m.nc, argentiere_observations_100m.csv
  results/argentiere_lg_100m_phys/   temperature-only inversion (history, final_terms, video_fields, sensitivity/)
  results/argentiere_lg_100m_joint/  joint inversion (history, final_terms, final_fields)
  results_archive/synthetic_lia_1500_2017_2/  twin experiment (Great Aletsch, Appendix A)
  paper/data/benchmark_memory.jsonl  memory benchmark (paper/scripts/bench_memory.py)
Writes paper/figures/figNN_*.pdf (and a .png preview of each).
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("IGEM_DEVICE", "cpu")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import tools.run_argentiere_inversion as R  # noqa: E402
import tools.run_argentiere_joint_inversion as J  # noqa: E402
from core.moraine_obs import select, terminus_position  # noqa: E402
from data.loader import load_argentiere  # noqa: E402

matplotlib.use("Agg")       # core.glacier switches backends on import; make sure
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FIXED = "results/argentiere_lg_100m_phys"
JOINT = "results/argentiere_lg_100m_joint"
TWIN = "results_archive/synthetic_lia_1500_2017_2"
DX = 100

# --- style: one categorical order everywhere (validated: blue, orange, aqua) ----------------------
C_JOINT, C_FIXED, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_GREY, INK, INK2 = "#898781", "#0b0b0b", "#52514e"
C_ENV, C_YD = "#e1e0d9", "#f0efec"
C_MOR, C_BED = "#e34948", "#6250d6"          # moraine crests, bedrock / erratic samples (maps)
C_LOW, C_HIGH = "#2a78d6", "#e34948"         # parameter below / above its nominal value
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 1.4,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42,
})
W = 6.97                                      # \textwidth of the Copernicus manuscript layout (in)
YD = (12.9, 11.7)                             # ka BP (GS-1)


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, fontweight="bold", color=INK)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def ka(t):
    return (1950.0 - np.asarray(t, dtype=float)) / 1000.0


def shade_yd(ax, label=False, y=0.97):
    """Mark the Younger Dryas by hairlines at its bounds and a bracket at the top of the axes."""
    for t in YD:
        ax.axvline(t, c=C_GREY, lw=0.5, zorder=0)
    tr = ax.get_xaxis_transform()
    ax.plot(YD, [y, y], c=INK2, lw=0.8, transform=tr, clip_on=False)
    ax.text(0.5 * sum(YD), y + 0.015, "Younger Dryas", transform=tr, ha="center", va="bottom",
            fontsize=6.5, color=INK2)


# --- data ------------------------------------------------------------------------------------------
def grid():
    nc = netCDF4.Dataset(f"data/Argentiere/argentiere_{DX}m.nc")
    g = {k: np.asarray(nc[k][:]) for k in nc.variables if nc[k].ndim == 2 and not k.startswith("obs")}
    g["x"], g["y"] = np.asarray(nc["x"][:]) / 1000, np.asarray(nc["y"][:]) / 1000
    g["obs"] = {k[4:]: np.asarray(nc[k][:]) > 0.5 for k in nc.variables if k.startswith("obs_")}
    nc.close()
    assert g["y"][1] > g["y"][0], "y is expected ascending"
    return g


_GEO = {}


def geo_obs():
    if not _GEO:
        geo, obs = load_argentiere(DX, mask=R.WINDOWS["lg"]["mask"], valley=True)
        w = R.WINDOWS["lg"]
        _GEO["geo"], _GEO["obs"] = geo, select(obs, w["t_start"], w["t_end"], w["dt_snap"])
    return _GEO["geo"], _GEO["obs"]


def hillshade(bed, az=315, alt=40):
    """Hillshade of a south-up array (row 0 = south), returned south-up as well."""
    hs = LightSource(azdeg=az, altdeg=alt).hillshade(bed[::-1], vert_exag=1.0, dx=DX, dy=DX)
    return hs[::-1]


def show_bed(ax, g, alpha=1.0):
    ext = [g["x"][0] - .05, g["x"][-1] + .05, g["y"][0] - .05, g["y"][-1] + .05]
    ax.imshow(hillshade(g["topg"]), cmap="gray", origin="lower", extent=ext, vmin=-0.1, vmax=1.15,
              alpha=alpha, interpolation="bilinear", rasterized=True)
    return ext


def cell_xy(g, flat):
    iy, ix = np.unravel_index(np.asarray(flat), g["topg"].shape)
    return g["x"][ix], g["y"][iy]


SITE_NAMES = {"ICEM-Protin2019-A": "Lognan L1", "ICEM-Protin2019-B": "Lognan L2",
              "ICEM-Protin2019-C": "Lognan L3", "ICEM-Protin2019-D": "La Joux J1",
              "ICEM-Protin2019-E": "La Joux J2", "undated-LaJoux": "La Joux (undated)",
              "undated-Lognan": "Lognan (undated a)", "undated-Lognan-2": "Lognan (undated b)",
              "ICEM-Protin2021-A": "Talèfre T1", "ICEM-Protin2021-B": "Talèfre T2"}


def history(run):
    return json.load(open(f"{run}/history.json"))


def final(run):
    return json.load(open(f"{run}/final_terms.json"))


def anomalies():
    """Node ages (ka), anomaly to 2012-2021 of the fixed-parameter and joint best iterates."""
    hf, hj = history(FIXED), history(JOINT)
    nt = np.array(hf["node_times"])
    a_fix = np.array(final(FIXED)["T"]) - R.CLIMATES["phys"]["T_present"]
    b = hj["best"]
    a_joint = np.array(b["T"]) - b["T_present"]
    return ka(nt), a_fix, a_joint


# parameter uncertainty of the anomaly (1 sigma), notebook section 11: forward runs where available,
# gradient estimates (implicit-function theorem, full adjoint) for the lapse rate and the
# Lateglacial precipitation
SENS = f"{FIXED}/sensitivity"
GRADIENT_SIGMA = {"lapse rate ±1 °C km$^{-1}$": 0.22, "Lateglacial precipitation ±25 %": 0.97}


def sens_cases():
    base = json.load(open(f"{SENS}/base.json"))["fit"]

    def shift(name):
        s = json.load(open(f"{SENS}/{name}.json"))
        return s["fit"]["dT"] - s["params"]["dT_present"] - base["dT"], s["fit"]["loss"]

    return base, shift


def sigma_parameters():
    base, shift = sens_cases()
    s = {"present precipitation ±0.3 m yr$^{-1}$": abs(shift("P+")[0] - shift("P-")[0]) / 2,
         "degree-day factor ±2 mm K$^{-1}$ d$^{-1}$": abs(shift("DDF+")[0] - shift("DDF-")[0]) / 2,
         "seasonal half-range ±1.5 °C": abs(shift("Ts+")[0] - shift("Ts-")[0]) / 2,
         "flow-law factor ±40 %": abs(shift("A+")[0] - shift("A-")[0]) / 2,
         "sliding stress −30 %": abs(shift("tau-")[0])}
    s.update(GRADIENT_SIGMA)
    return s


def sigma_data(ka_n):
    """Data-only 1-sigma per period: sqrt(2 / L'') of a uniform shift of that period's nodes."""
    out = np.full(len(ka_n), np.nan)
    for key, (a, b) in {"allerod": (13.6, 12.9), "yd": (12.9, 11.7), "preboreal": (11.7, 10.0),
                        "early_holocene": (10.0, 9.0)}.items():
        s = json.load(open(f"{SENS}/block_{key}.json"))
        m = np.array(s["mask"]) > 0
        out[m] = np.sqrt(2.0 / s["fit"]["curv"])
    return out


# ==================================================================================================
def fig_study_area():
    """Study area: bed, ice masks, flowlines and the Lateglacial constraints."""
    g = grid()
    geo, obs = geo_obs()
    df = pd.read_csv(f"data/Argentiere/argentiere_observations_{DX}m.csv")
    df = df[df.in_catchment & (df.t0 < R.WINDOWS["lg"]["t_end"])]
    fig, ax = plt.subplots(figsize=(W, W * 0.80))
    ext = show_bed(ax, g)
    ax.contour(g["x"], g["y"], g["topg"], levels=np.arange(1000, 4800, 500), colors="k",
               linewidths=0.25, alpha=0.35)
    ax.contour(g["x"], g["y"], g["catchment_lateglacial"], [0.5], colors="#008300", linewidths=1.2)
    ax.contour(g["x"], g["y"], (g["lia_id"] > 0).astype(float), [0.5], colors=C_JOINT, linewidths=0.6)
    ax.contour(g["x"], g["y"], g["argentiere_2012"], [0.5], colors=INK, linewidths=0.7)
    names = {"Argentiere": "Argentière", "Talefre": "Talèfre", "Mer de Glace": "Mer de Glace", "Tour": "Tour"}
    for name, fl in geo["flowlines"].items():
        x, y = cell_xy(g, fl.active)
        ax.plot(x, y, c=INK, lw=0.8, ls=(0, (4, 2)), alpha=0.9)
    # moraine crests (dated: solid, morphostratigraphic: thinner)
    for _, r in df[df.kind.isin(["lateral", "frontal"])].iterrows():
        m = g.get(f"line_{r['key']}")
        if m is None:
            continue
        lw = 2.2 if r["kind"] == "frontal" else (1.6 if r["group"] == "dated" else 1.0)
        ax.contour(g["x"], g["y"], (m > 0.5).astype(float), [0.5], colors=C_MOR, linewidths=lw)
    bed_ = df[df.kind == "icefree"]
    for is_err, mk in ((False, "o"), (True, "^")):
        d = bed_[bed_.id.str.startswith("ERR") == is_err]
        ax.scatter(d.x / 1000, d.y / 1000, s=22, marker=mk, c=C_BED, edgecolor="white", lw=0.6, zorder=6)
    j2 = df[df.kind == "frontal"].iloc[0]
    ax.scatter([j2.x / 1000], [j2.y / 1000], s=110, marker="*", c=C_MOR, edgecolor=INK, lw=0.6, zorder=7)
    # site labels at the mean position of their constraints
    lab = {"La Joux": (336.6, 5091.6, "right"), "Lognan": (342.9, 5093.5, "left"),
           "Jardin de Talefre": (344.3, 5087.6, "left"), "Aiguille du Moine": (341.7, 5084.9, "left"),
           "Mont Blanc Tête de Trélaporte": (338.3, 5087.2, "right"), "Remuaz": (337.2, 5096.4, "right"),
           "Cheserys": (336.9, 5094.6, "right")}
    pretty = {"Jardin de Talefre": "Jardin de Talèfre", "Mont Blanc Tête de Trélaporte": "Tête de Trélaporte"}
    for site, (xt, yt, ha) in lab.items():
        d = df[df.name == site]
        if len(d):
            ax.annotate(pretty.get(site, site), (d.x.mean() / 1000, d.y.mean() / 1000),
                        xytext=(xt, yt), ha=ha, va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=0, shrinkB=2),
                        fontsize=7, color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for txt, (x, y) in {"Argentière glacier": (346.4, 5090.3), "Mer de Glace": (338.9, 5084.6),
                        "Tour glacier": (345.8, 5096.7), "Arve": (333.6, 5087.6)}.items():
        ax.text(x, y, txt, fontsize=7, style="italic", color=INK, ha="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    handles = [Line2D([], [], c="#008300", lw=1.2, label="area where ice may grow (Lateglacial)"),
               Line2D([], [], c=C_JOINT, lw=0.8, label="Little Ice Age extent (start of the runs)"),
               Line2D([], [], c=INK, lw=0.8, label="Argentière glacier, 2012"),
               Line2D([], [], c=INK, lw=0.8, ls=(0, (4, 2)), label="valley flowlines"),
               Line2D([], [], c=C_MOR, lw=1.6, label="lateral moraine crests"),
               Line2D([], [], marker="*", c=C_MOR, mec=INK, ms=10, ls="none", label="frontal moraine J2"),
               Line2D([], [], marker="o", c=C_BED, mec="white", ms=5, ls="none", label="bedrock exposure age"),
               Line2D([], [], marker="^", c=C_BED, mec="white", ms=5, ls="none", label="erratic boulder")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5, frameon=True, framealpha=0.9,
              edgecolor="none")
    ax.set_xlim(330.5, 351.5); ax.set_ylim(5077.0, 5099.5)
    ax.set_aspect("equal")
    ax.set_xticks([335, 340, 345, 350]); ax.set_yticks([5080, 5085, 5090, 5095])
    ax.set_xlabel("easting (km, UTM 32N)"); ax.set_ylabel("northing (km)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    save(fig, "fig03_study_area")


# ==================================================================================================
def fig_operators():
    """Valley-referenced observation operators on the Argentière flowline."""
    g = grid()
    geo, obs = geo_obs()
    fl = geo["flowlines"]["Argentiere"]
    by = {o["id"]: o for o in obs}
    fj = np.load(f"{JOINT}/final_fields.npz")
    k = int(np.argmin(abs(fj["times"] - (1950 - 10900))))
    H = fj["H"][k]
    fig = plt.figure(figsize=(W, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    show_bed(ax, g)
    Hm = np.where(H > 1, H, np.nan)
    ax.imshow(Hm, origin="lower", extent=[g["x"][0] - .05, g["x"][-1] + .05, g["y"][0] - .05, g["y"][-1] + .05],
              cmap="Blues", vmin=0, vmax=400, alpha=0.75, rasterized=True)
    x, y = cell_xy(g, fl.active)
    ax.plot(x, y, c=INK, lw=1.0)
    for km in range(8, int(fl.s[fl.stop - 1] / 1000) + 1, 2):
        i = int(np.argmin(abs(fl.s[:fl.stop] - km * 1000)))
        ax.plot(x[i], y[i], "o", ms=3, c=INK)
        ax.annotate(f"{km} km", (x[i], y[i]), xytext=(3, 4), textcoords="offset points", fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    for oid in ("ICEM-Protin2019-A", "ICEM-Protin2019-B", "ICEM-Protin2019-C", "ICEM-Protin2019-D",
                "undated-LaJoux", "undated-Lognan", "undated-Lognan-2"):
        r = pd.read_csv(f"data/Argentiere/argentiere_observations_{DX}m.csv").set_index("id").loc[oid]
        m = g.get(f"line_{r['key']}")
        my, mx = np.nonzero(m > 0.5)
        kk, _ = fl.project(m > 0.5)
        ay, ax_ = np.unravel_index(fl.active[kk], g["topg"].shape)
        for i in range(0, len(my), max(1, len(my) // 6)):
            ax.plot([g["x"][mx[i]], g["x"][ax_[i]]], [g["y"][my[i]], g["y"][ay[i]]], c=C_MOR, lw=0.5, alpha=0.8)
        ax.plot(g["x"][mx], g["y"][my], "s", ms=1.6, c=C_MOR)
    j2 = by["ICEM-Protin2019-E"]
    kj = int(np.argmin(abs(fl.s[:fl.stop] - j2["s_obs"])))
    ax.plot(x[kj], y[kj], "*", ms=10, c=C_MOR, mec=INK, mew=0.6)
    sel = fl.s[:fl.stop] >= 6000
    ax.set_xlim(x[sel].min() - 0.7, x[sel].max() + 0.7); ax.set_ylim(y[sel].min() - 1.6, y[sel].max() + 1.9)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.set_xlabel("easting (km)"); ax.set_ylabel("northing (km)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    panel(ax, "a")

    ax = fig.add_subplot(gs[1])
    cells = fl.cells[:fl.stop]; s_km = fl.s[:fl.stop] / 1000
    bed = g["topg"].ravel()[cells]; h = np.maximum(H.ravel()[cells], 0)
    ax.fill_between(s_km, bed - 400, bed, color="#c3c2b7", lw=0)
    ax.fill_between(s_km, bed, bed + h, where=h > 1, color="#b7d3f6", lw=0)
    ax.plot(s_km, np.where(h > 1, bed + h, np.nan), c=C_JOINT, lw=1.4, label="ice surface on the axis, 10.9 ka")
    first = True
    for oid in ("ICEM-Protin2019-A", "ICEM-Protin2019-B", "ICEM-Protin2019-C", "ICEM-Protin2019-D",
                "undated-LaJoux", "undated-Lognan", "undated-Lognan-2"):
        o = by[oid]
        zc = o["z"].cpu().numpy() + o["delta"]
        sc = np.asarray(o["s_cells"]) / 1000
        ax.errorbar(sc, zc, yerr=o["sigma_z"], fmt="s", ms=2.5, c=C_MOR, elinewidth=0.5, capsize=0,
                    alpha=0.9, label=r"crest + $\delta$ at its cross-section ($\pm\sigma_z$)" if first else None)
        first = False
    zj = bed[kj]
    path = torch.tensor(fl.active[fl.start:]); s = torch.tensor(fl.s[fl.start:fl.stop], dtype=torch.float32)
    L = float(terminus_position(torch.tensor(H, dtype=torch.float32), path, s)) / 1000
    ax.errorbar([j2["s_obs"] / 1000], [zj], xerr=j2["sigma_L"] / 1000, fmt="*", ms=9, c=C_MOR, mec=INK,
                mew=0.6, elinewidth=0.8, capsize=2, label=r"frontal moraine J2 ($\pm\sigma_L$)", zorder=6)
    ax.axvline(L, c=C_JOINT, lw=0.8, ls=(0, (3, 2)))
    ax.annotate(f"soft terminus $L$ = {L:.1f} km", (L, 2000), xytext=(-6, 0), textcoords="offset points",
                ha="right", fontsize=7, color=C_JOINT)
    ax.set_xlim(6, s_km[-1]); ax.set_ylim(1100, 2700)
    ax.set_xlabel("distance along the Argentière flowline (km)"); ax.set_ylabel("elevation (m a.s.l.)")
    ax.legend(loc="upper right", fontsize=6.5, frameon=True, framealpha=0.9, edgecolor="none")
    panel(ax, "b")
    save(fig, "fig02_operators")


# ==================================================================================================
def fig04():
    """Temperature-only inversion (fixed parameters)."""
    H = history(FIXED); h = H["history"]
    ka_n, a_fix, _ = anomalies()
    Tp = R.CLIMATES["phys"]["T_present"]
    T_it = np.array(h["T"]) - Tp
    lo, hi = np.array(H["bounds"][0]) - Tp, np.array(H["bounds"][1]) - Tp
    sd = sigma_data(ka_n)
    sp = np.sqrt(sum(v ** 2 for v in sigma_parameters().values()))
    st = np.sqrt(sd ** 2 + sp ** 2)
    fig = plt.figure(figsize=(W, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.9], height_ratios=[1, 1.15], hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    it = np.arange(len(h["loss"]))
    ax.plot(it, h["loss"], "-o", ms=2.5, c=C_FIXED, label="objective $J$")
    ax.plot(it, h["data"], c=INK2, lw=1.0, label="data term")
    ax.set_xlabel("iteration"); ax.set_ylabel("misfit"); ax.legend(loc="upper right")
    panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(ka_n, lo, hi, step="mid", color=C_ENV, lw=0, label="literature envelope")
    cm = plt.get_cmap("Oranges")
    for i in range(len(T_it)):
        ax.plot(ka_n, T_it[i], c=cm(0.3 + 0.7 * i / (len(T_it) - 1)), lw=0.7)
    ax.plot(ka_n, T_it[0], c=cm(0.3), lw=0.7, label="iteration 0 (from the 200 m inversion)")
    ax.plot(ka_n, T_it[-1], c=cm(1.0), lw=1.2, label=f"iteration {len(T_it) - 1}")
    ax.invert_xaxis(); ax.set_xlabel("age (ka BP)"); ax.set_ylabel(r"$\Delta T$ (°C)")
    ax.legend(loc="lower left", fontsize=6.5, ncol=1)
    panel(ax, "b")
    ax = fig.add_subplot(gs[1, :])
    shade_yd(ax)
    ax.fill_between(ka_n, lo, hi, step="mid", color=C_ENV, lw=0, label="literature envelope (bounds)")
    ax.fill_between(ka_n, a_fix - st, a_fix + st, color=C_FIXED, alpha=0.16, lw=0, label=r"$\pm1\sigma$, data and parameters")
    ax.fill_between(ka_n, a_fix - sd, a_fix + sd, color=C_FIXED, alpha=0.35, lw=0, label=r"$\pm1\sigma$, data only")
    ax.plot(ka_n, a_fix, "-o", ms=2.5, c=C_FIXED, label="temperature-only inversion")
    ax.axhline(0, c=INK2, lw=0.6)
    ax.axhline(-R.PRESENT_MINUS_PREINDUSTRIAL, c=C_GREY, lw=0.8, ls=(0, (1, 1.5)))
    ax.text(13.66, -R.PRESENT_MINUS_PREINDUSTRIAL + 0.12, "pre-industrial", ha="left", fontsize=6.5, color=C_GREY)
    ax.invert_xaxis(); ax.set_xlim(13.7, 8.9); ax.set_ylim(-9.6, 1.0)
    ax.set_xlabel("age (ka BP)"); ax.set_ylabel(r"$\Delta T$ relative to 2012–2021 (°C)")
    ax.legend(loc="lower right", ncol=2, fontsize=6.5)
    panel(ax, "c")
    save(fig, "fig04_temperature_only")


# ==================================================================================================
def fig05():
    """Sensitivity of the Younger Dryas anomaly and of the fit to the fixed parameters."""
    base, shift = sens_cases()
    rows = [  # label, (low case, low label), (high case, high label)
        ("present precipitation", ("P-", "1.5 m yr$^{-1}$"), ("P+", "2.1 m yr$^{-1}$")),
        ("degree-day factor", ("DDF-", "8 mm K$^{-1}$ d$^{-1}$"), ("DDF+", "12 mm K$^{-1}$ d$^{-1}$")),
        ("seasonal half-range $T_s$", ("Ts-", "7.9 °C"), ("Ts+", "10.9 °C")),
        ("flow-law factor $A$", ("A-", "52 MPa$^{-3}$ yr$^{-1}$"), ("A+", "117 MPa$^{-3}$ yr$^{-1}$")),
        (r"sliding stress $\tau_\mathrm{ref}$", ("tau-", "0.15 MPa"), None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.08))
    ax = axes[0]
    ax.axvspan(-0.22, 0.22, color=C_ENV, lw=0)
    ax.text(0, -0.75, r"data-only $\pm1\sigma$", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ylab = []
    yy = 0
    for lab, low, high in rows:
        for case, col, mk in ((low, C_LOW, "o"), (high, C_HIGH, "s")):
            if case is None:
                continue
            v, _ = shift(case[0])
            ax.plot([0, v], [yy, yy], c=col, lw=1.0)
            ax.plot(v, yy, mk, c=col, ms=5)
        ylab.append(lab); yy += 1
    for lab, sgm in GRADIENT_SIGMA.items():
        for v, col, mk in ((-sgm, C_LOW, "o"), (sgm, C_HIGH, "s")):
            ax.plot([0, v], [yy, yy], c=col, lw=1.0, ls=(0, (2, 1.5)))
            ax.plot(v, yy, mk, mfc="white", mec=col, ms=5)
        ylab.append(lab.split(" ±")[0] + " (gradient)"); yy += 1
    ax.axvline(0, c=INK2, lw=0.6)
    ax.set_yticks(range(len(ylab))); ax.set_yticklabels(ylab); ax.set_ylim(len(ylab) - 0.4, -1.1)
    ax.set_xlabel(r"change of the Younger Dryas $\Delta T$ (°C)")
    ax.legend(handles=[Line2D([], [], marker="o", c=C_LOW, ls="none", ms=5, label="parameter lowered"),
                       Line2D([], [], marker="s", c=C_HIGH, ls="none", ms=5, label="parameter raised")],
              loc="center right", fontsize=6.5)
    panel(ax, "a", x=-0.62)
    ax = axes[1]
    cases = [("P-", "P 1.5"), ("P+", "P 2.1"), ("DDF-", "DDF 8"), ("DDF+", "DDF 12"), ("Ts-", "$T_s$ 7.9"),
             ("Ts+", "$T_s$ 10.9"), ("A-", "$A$ 52"), ("A+", "$A$ 117"), ("tau-", r"$\tau_\mathrm{ref}$ 0.15")]
    vals = [shift(c)[1] for c, _ in cases]
    y = np.arange(len(cases))
    ax.barh(y, vals, height=0.62, color=C_JOINT)
    for yi, v in zip(y, vals):
        ax.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=6.5, color=INK)
    ax.axvline(base["loss"], c=INK, lw=0.8, ls=(0, (3, 2)))
    ax.text(base["loss"] + 0.3, -0.85, f"literature values {base['loss']:.1f}", fontsize=6.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in cases]); ax.set_ylim(len(cases) - 0.4, -1.2)
    ax.set_xlabel(r"objective after the best uniform shift of $T(t)$"); ax.set_xlim(0, 31)
    ax.yaxis.tick_right(); ax.tick_params(axis="y", length=0)
    panel(ax, "b")
    save(fig, "fig05_sensitivity")


# ==================================================================================================
def fig06():
    """Joint inversion: convergence, controls, temperature."""
    H = history(JOINT); h = H["history"]; n = H["n_nodes"]; keys = H["keys"]
    z = np.array(h["z"])
    ka_n, a_fix, a_joint = anomalies()
    lo_a, hi_a = J.anomaly_bounds(np.array(H["node_times"]))
    sd = sigma_data(ka_n)
    sp = np.sqrt(sum(v ** 2 for v in sigma_parameters().values()))
    st = np.sqrt(sd ** 2 + sp ** 2)
    fig = plt.figure(figsize=(W, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    it = np.arange(len(h["loss"]))
    ax.plot(it, h["loss"], "-o", ms=2.5, c=C_JOINT, label="objective $J$")
    ax.plot(it, h["data"], c=INK2, lw=1.0, label="data term")
    ax.axhline(h["loss"][0], c=C_FIXED, lw=0.8, ls=(0, (3, 2)))
    ax.text(29, h["loss"][0] + 0.35, "temperature-only result", ha="right", fontsize=6.5, color=C_FIXED)
    ax.set_xlabel("iteration"); ax.set_ylabel("misfit"); ax.legend(loc="center right")
    ax.set_ylim(10, 24.5)
    panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    names = {"precip": "$P$", "T_s": "$T_s$", "ddf": "DDF", "arrhenius": "$A$", "tau_ref": r"$\tau_\mathrm{ref}$"}
    emph = {"arrhenius": C_FIXED, "tau_ref": C_JOINT}
    ends = {}
    for kk, key in enumerate(keys):
        d = z[:, n + kk] - z[0, n + kk]
        col = emph.get(key, C_GREY)
        ax.plot(it, d, c=col, lw=1.4 if key in emph else 0.9)
        ends[key] = d[-1]
    dT = np.abs(z[:, :n] - z[0, :n]).mean(1)
    ax.plot(it, dT, c=INK, lw=1.0, ls=(0, (3, 2)))
    ends["nodes"] = dT[-1]
    names["nodes"] = r"$T_k$ (mean $|\Delta z|$)"
    # direct labels, spread so that they do not collide
    order = sorted(ends, key=lambda k: ends[k])
    pos = [ends[k] for k in order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + 0.095)
    for k, p_ in zip(order, pos):
        ax.text(it[-1] + 0.6, p_, names[k], va="center", fontsize=7,
                color=INK if (k in emph or k == "nodes") else INK2)
    ax.axhline(0, c=INK2, lw=0.6)
    ax.set_xlim(0, 37); ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$z - z_0$ (fraction of range)")
    panel(ax, "b")
    ax = fig.add_subplot(gs[1, :])
    shade_yd(ax)
    ax.fill_between(ka_n, lo_a, hi_a, step="mid", color=C_ENV, lw=0, label="literature envelope (bounds)")
    ax.fill_between(ka_n, a_joint - st, a_joint + st, color=C_JOINT, alpha=0.15, lw=0,
                    label=r"$\pm1\sigma$, data and parameters")
    ax.plot(ka_n, a_fix, "-o", ms=2.3, c=C_FIXED, lw=1.1, label="temperature-only inversion")
    ax.plot(ka_n, a_joint, "-o", ms=2.8, c=C_JOINT, lw=1.6, label="joint inversion")
    ax.axhline(0, c=INK2, lw=0.6)
    ax.axhline(-R.PRESENT_MINUS_PREINDUSTRIAL, c=C_GREY, lw=0.8, ls=(0, (1, 1.5)))
    ax.text(13.66, -R.PRESENT_MINUS_PREINDUSTRIAL + 0.12, "pre-industrial", ha="left", fontsize=6.5, color=C_GREY)
    ax.invert_xaxis(); ax.set_xlim(13.7, 8.9); ax.set_ylim(-9.6, 1.0)
    ax.set_xlabel("age (ka BP)"); ax.set_ylabel(r"$\Delta T$ relative to 2012–2021 (°C)")
    ax.legend(loc="lower right", ncol=2, fontsize=6.5)
    panel(ax, "c")
    save(fig, "fig06_joint")


# ==================================================================================================
STANDS = [(12.5, "12.5 ka"), (11.66, "11.7 ka (L1)"), (10.9, "10.9 ka (J2)"), (10.35, "10.3 ka (L2, L3)")]


def fig07():
    """The joint glacier at the dated stands: maps and the Argentière axis."""
    g = grid()
    geo, obs = geo_obs()
    fl = geo["flowlines"]["Argentiere"]; by = {o["id"]: o for o in obs}
    fj, ff = np.load(f"{JOINT}/final_fields.npz"), np.load(f"{FIXED}/video_fields.npz")
    snap = lambda f, k: f["H"][int(np.argmin(abs(f["times"] - (1950 - k * 1000))))]
    path = torch.tensor(fl.active[fl.start:]); s = torch.tensor(fl.s[fl.start:fl.stop], dtype=torch.float32)
    term = lambda Hm: float(terminus_position(torch.tensor(Hm, dtype=torch.float32), path, s))
    xf, yf = cell_xy(g, fl.active)
    j2 = by["ICEM-Protin2019-E"]; kj = int(np.argmin(abs(fl.s[:fl.stop] - j2["s_obs"])))
    fig = plt.figure(figsize=(W, 3.85))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.27, 1.75], hspace=0.22, wspace=0.08)
    ext = [g["x"][0] - .05, g["x"][-1] + .05, g["y"][0] - .05, g["y"][-1] + .05]
    hs = hillshade(g["topg"])
    for i, (kk, ttl) in enumerate(STANDS):
        ax = fig.add_subplot(gs[0, i])
        Hm = snap(fj, kk)
        ax.imshow(hs, cmap="gray", origin="lower", extent=ext, vmin=-0.1, vmax=1.15, rasterized=True)
        im = ax.imshow(np.where(Hm > 1, Hm, np.nan), origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=400,
                       alpha=0.9, rasterized=True)
        for key, m in g.items():
            if key.startswith("line_") and key[5:].startswith(("ICEM_Protin2019", "undated", "ICEM_Protin2021_A", "ICEM_Protin2021_B")):
                ax.contour(g["x"], g["y"], (m > 0.5).astype(float), [0.5], colors=C_MOR, linewidths=0.9)
        ax.plot(xf, yf, c=INK, lw=0.5, alpha=0.7)
        kt = int(np.argmin(abs(fl.s[:fl.stop] - term(Hm))))
        ax.plot(xf[kt], yf[kt], "v", ms=6, mfc="white", mec=INK, mew=1.0)
        ax.plot(xf[kj], yf[kj], "*", ms=8, c=C_MOR, mec=INK, mew=0.5)
        ax.set_xlim(333.6, 349.2); ax.set_ylim(5084.0, 5096.4); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(True)
        ax.set_title(ttl, fontsize=7.5)
        if i == 0:
            panel(ax, "a")
    cax = fig.add_axes([0.915, 0.60, 0.010, 0.26])
    cb = fig.colorbar(im, cax=cax); cb.set_label("ice thickness (m)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    cells = fl.cells[:fl.stop]; s_km = fl.s[:fl.stop] / 1000; bedf = g["topg"].ravel()[cells]
    for i, (kk, ttl) in enumerate(STANDS):
        ax = fig.add_subplot(gs[1, i])
        ax.fill_between(s_km, bedf - 400, bedf, color="#c3c2b7", lw=0)
        for f, col, ls, lab in ((ff, C_FIXED, (0, (3, 1.5)), "temperature-only"), (fj, C_JOINT, "-", "joint")):
            Hk = np.maximum(snap(f, kk).ravel()[cells], 0)
            ax.plot(s_km, np.where(Hk > 1, bedf + Hk, np.nan), c=col, ls=ls, lw=1.2, label=lab)
        for oid in ("ICEM-Protin2019-A", "ICEM-Protin2019-B", "ICEM-Protin2019-C", "ICEM-Protin2019-D",
                    "undated-LaJoux", "undated-Lognan", "undated-Lognan-2"):
            o = by[oid]
            ax.plot(np.asarray(o["s_cells"]) / 1000, o["z"].cpu().numpy(), "s", ms=2.2, c=C_MOR)
        ax.plot(j2["s_obs"] / 1000, bedf[kj], "*", ms=8, c=C_MOR, mec=INK, mew=0.5, zorder=6)
        ax.set_xlim(6, s_km[-1]); ax.set_ylim(1100, 2800)
        ax.set_xlabel("km along flowline", fontsize=7)
        if i == 0:
            ax.set_ylabel("elevation (m a.s.l.)"); ax.legend(loc="upper right", fontsize=6.3)
            panel(ax, "b")
        else:
            ax.set_yticklabels([])
    save(fig, "fig07_stands")


# ==================================================================================================
def fig08():
    """Distance to each constraint: temperature-only vs joint."""
    F, Jt = final(FIXED)["describe"], final(JOINT)["describe"]
    meta = history(FIXED)["meta"]
    lat = [o for o in ("ICEM-Protin2019-A", "ICEM-Protin2019-B", "ICEM-Protin2019-C", "ICEM-Protin2021-A",
                       "ICEM-Protin2021-B", "ICEM-Protin2019-D", "undated-LaJoux", "undated-Lognan",
                       "undated-Lognan-2")]
    pre = [o + "-preserve" for o in ("ICEM-Protin2019-A", "ICEM-Protin2019-B", "ICEM-Protin2019-C",
                                     "ICEM-Protin2021-A", "ICEM-Protin2021-B")]
    bed = sorted([o for o, m in meta.items() if m["kind"] == "icefree_v"],
                 key=lambda o: -max(F[o]["dz_m"], Jt[o]["dz_m"]))[:10]
    bed_names = {"BED-Lehmann2020-F": "Trélaporte F", "ERR-Protin2019-A": "La Joux erratic",
                 "BED-Lehmann2020-C": "Moine C", "BED-Protin2019-E": "Lognan E", "BED-Protin2019-C": "Lognan C",
                 "BED-Lehmann2020-B": "Moine B", "BED-Lehmann2020-G": "Trélaporte G", "BED-Protin2019-B": "Lognan B",
                 "BED-Protin2019-A": "Lognan A", "BED-Protin2019-D": "Lognan D", "BED-Lehmann2020-A": "Moine A",
                 "BED-Protin2021-A": "Jardin de Talèfre", "BED-Lehmann2020-D": "Trélaporte D"}
    age = lambda o: (1950 - meta[o]["year"]) / 1000
    groups = [
        ("lateral moraines", lat,
         lambda o: f"{SITE_NAMES[o]}" + (f" ({age(o):.1f} ka)" if meta[o]["group"] == "dated" else "")),
        ("moraines that must survive", pre,
         lambda o: SITE_NAMES[o[:-9]]),
        ("bedrock and erratics", bed,
         lambda o: f"{bed_names.get(o, o)} ({age(o):.1f} ka)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.2), gridspec_kw=dict(wspace=0.95, width_ratios=[1, 0.8, 1]))
    for ax, (ttl, ids, lab), letter in zip(axes, groups, "abc"):
        y = np.arange(len(ids))
        vf = np.array([F[o]["dz_m"] for o in ids]); vj = np.array([Jt[o]["dz_m"] for o in ids])
        ax.axvspan(-60, 60, color=C_ENV, lw=0) if "lateral" in ttl else ax.axvspan(-150, 0, color=C_ENV, lw=0)
        ax.barh(y - 0.19, np.clip(vf, -150, None), height=0.36, color=C_FIXED, label="temperature-only")
        ax.barh(y + 0.19, np.clip(vj, -150, None), height=0.36, color=C_JOINT, label="joint")
        ax.axvline(0, c=INK2, lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([lab(o) for o in ids], fontsize=6.5)
        ax.set_ylim(len(ids) - 0.4, -0.6)
        ax.set_title(ttl, fontsize=7.5, loc="left")
        ax.set_xlabel({"lateral moraines": "axis surface − (crest + δ) (m)",
                       "moraines that must survive": "overriding after formation (m)",
                       "bedrock and erratics": "ice above the sample (m)"}[ttl], fontsize=7)
        ax.set_xlim(-150 if "bedrock" in ttl else -60, 160 if "bedrock" in ttl else 90)
        panel(ax, letter, x=-0.02, y=1.08)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.55, -0.2), ncol=2, fontsize=7)
    save(fig, "fig08_landform_fit")


# ==================================================================================================
def fig09():
    """Argentière terminus through time."""
    geo, obs = geo_obs()
    j2 = {o["id"]: o for o in obs}["ICEM-Protin2019-E"]
    tr = {"temperature-only": final(FIXED)["termini"]["Argentiere"], "joint": final(JOINT)["termini"]["Argentiere"]}
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.5))
    shade_yd(ax)
    ax.axvspan(ka(j2["t1"]), ka(j2["t0"]), ymin=0, ymax=1, color=C_MOR, alpha=0.08, lw=0)
    ax.axhline(j2["s_obs"] / 1000, c=C_MOR, lw=0.8)
    ax.text(9.05, j2["s_obs"] / 1000 + 0.08, "frontal moraine J2\n(age window shaded)", fontsize=6.5,
            color=C_MOR, ha="right", va="bottom")
    for (lab, d), col, ls in zip(tr.items(), (C_FIXED, C_JOINT), ((0, (3, 1.5)), "-")):
        t = np.array(sorted(d, key=float), dtype=float)
        ax.plot(ka(t), np.array([d[k] for k in sorted(d, key=float)]) / 1000, c=col, ls=ls, lw=1.3, label=lab)
    ax.invert_xaxis(); ax.set_xlim(13.65, 8.95)
    ax.set_xlabel("age (ka BP)"); ax.set_ylabel("Argentière terminus (km along flowline)")
    ax.legend(loc="lower left")
    save(fig, "fig09_terminus")


# ==================================================================================================
def fig10():
    """Memory needed to differentiate a run, measured per step and extrapolated."""
    rows = [json.loads(l) for l in open("paper/data/benchmark_memory.jsonl")]
    for r in rows:
        r.setdefault("adjoint", "frozen")
    get = lambda mode, adj: sorted([r for r in rows if r["mode"] == mode and r["adjoint"] == adj], key=lambda r: r["steps"])
    slope = lambda rr: (rr[-1]["peak_GB"] - rr[0]["peak_GB"]) / (rr[-1]["steps"] - rr[0]["steps"])
    m_full = slope(get("none", "full")); m_froz = slope(get("none", "frozen"))
    m_ck = slope(get("step", "full")); base_ck = get("step", "full")[0]["peak_GB"] - m_ck * get("step", "full")[0]["steps"]
    base_fr = get("step", "frozen")[0]["peak_GB"] - slope(get("step", "frozen")) * get("step", "frozen")[0]["steps"]
    N = np.logspace(1, 6.3, 300)
    steps_per_kyr = 82749 / 4.6
    fig, ax = plt.subplots(figsize=(W * 0.62, 3.0))
    curves = [
        (N * m_full, C_FIXED, "-", "whole graph stored"),
        (N * m_froz + base_fr, C_GREY, "-", "whole graph, frozen velocity"),
        (N * m_ck + base_ck, C_JOINT, "-", "one checkpoint per step"),
        ((N / 300 + 300) * m_ck + base_ck, C_AQUA, "-", "two levels, blocks of 300 steps"),
    ]
    for y, col, ls, lab in curves:
        ax.plot(N / steps_per_kyr, y, c=col, ls=ls, lw=1.4, label=lab)
    for gb, txt in ((24, "24 GB (RTX 4090)"), (96, "96 GB (RTX PRO 6000)")):
        ax.axhline(gb, c=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.text(8e-2, gb * 1.12, txt, fontsize=6.5, color=INK2)
    ax.axvline(4.6, c=INK, lw=0.6)
    ax.text(4.6 * 1.08, 2e3, "this study\n(4.6 kyr)", fontsize=6.5, va="top")
    for mode, adj, mk in (("none", "full", "o"), ("step", "full", "s"), ("none", "frozen", "^")):
        for r in get(mode, adj):
            ax.plot(r["steps"] / steps_per_kyr, r["peak_GB"], mk, ms=4, mfc="white", mec=INK, mew=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(3e-4, 100); ax.set_ylim(0.05, 3e5)
    ax.set_xlabel("simulated time on the 100 m Argentière grid (kyr)")
    ax.set_ylabel("GPU memory for one gradient (GB)")
    ax.legend(loc="upper left", fontsize=6.3)
    save(fig, "fig10_memory")
    return dict(m_full=m_full, m_froz=m_froz, m_ck=m_ck, base_ck=base_ck)


# ==================================================================================================
def figA1():
    """Twin experiment on the Great Aletsch Glacier, 1500-2017."""
    r = json.load(open(f"{TWIN}/inversion_history.json"))
    tr = r["truth"]; h = r["history"]; nt = np.array(r["node_times"], dtype=float)
    years = np.array(tr["years"]); T_ann = np.array(tr["T_annual"])
    mid = (nt + np.minimum(nt + 50, 2017)) / 2
    curve = lambda v: np.interp(years, mid, v)
    best = r["best"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[1.7, 1], wspace=0.3))
    ax = axes[0]
    for i, T in enumerate(h["T"]):
        ax.plot(years, curve(np.array(T)), c=C_FIXED, lw=0.4, alpha=0.12 + 0.25 * i / len(h["T"]))
    ax.plot(years, T_ann, c=INK, lw=1.5, label="true temperature")
    ax.plot(years, curve(np.array(best["T"])), c=C_JOINT, lw=1.5, label=f"inverted (iteration {best['iter']})")
    ax.plot([], [], c=C_FIXED, lw=0.6, alpha=0.5, label="all iterations")
    for ep in r["epochs"]:
        ax.plot(ep, 6.85, "^", c=INK2, ms=4)
    ax.text(1505, 6.9, "observed extents", fontsize=6.5, color=INK2, va="bottom")
    ax.set_xlim(1500, 2017); ax.set_ylim(6.8, 8.9)
    ax.set_xlabel("year CE"); ax.set_ylabel("temperature at the lowest grid point (°C)")
    ax.legend(loc="upper left")
    panel(ax, "a")
    ax = axes[1]
    ep = [str(e) for e in r["epochs"]]
    iou0 = [h["iou"][e][0] for e in ep]; iou_b = [h["iou"][e][best["iter"]] for e in ep]
    x = np.arange(len(ep))
    ax.plot(x, iou0, "o-", c=C_GREY, ms=3.5, lw=1.0, label="initial guess (7.0 °C)")
    ax.plot(x, iou_b, "s-", c=C_JOINT, ms=3.5, lw=1.2, label="inverted")
    ax.set_xticks(x); ax.set_xticklabels(ep, rotation=0)
    ax.set_ylabel("intersection over union"); ax.set_ylim(0.89, 1.005)
    ax.legend(loc="lower left")
    panel(ax, "b")
    save(fig, "figA1_twin")


ALL = dict(fig02=fig_operators, fig03=fig_study_area, fig04=fig04, fig05=fig05, fig06=fig06, fig07=fig07,
           fig08=fig08, fig09=fig09, fig10=fig10, figA1=figA1)

if __name__ == "__main__":
    todo = sys.argv[1:] or list(ALL)
    for name in todo:
        out = ALL[name]()
        if out:
            print(name, out)

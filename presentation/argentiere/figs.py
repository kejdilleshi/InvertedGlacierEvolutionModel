"""Figures for the Argentière Younger Dryas talk, from results/argentiere_lg_100m_phys.

    cd <repo>; PYTHONPATH=. IGEM_DEVICE=cpu python presentation/argentiere/figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import tools.run_argentiere_inversion as R
from core.moraine_obs import axis_surface, huber, select, terminus_position
from core.smb import update_smb_PDD
from data.loader import load_argentiere

matplotlib.use("Agg", force=True)          # core.glacier switches backends on import
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT, exist_ok=True)
RUN = "results/argentiere_lg_100m_phys"
DX = 100

NAVY, SLATE, ICE = "#102A3A", "#2E5266", "#EAF2F6"
BLUE, EMBER, MELT, GREY, GOLD, MOSS = "#2B7BB9", "#D0532A", "#8DC3DE", "#8A9AA5", "#E0A43A", "#5B8C6A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": SLATE, "axes.labelcolor": NAVY,
                     "xtick.color": SLATE, "ytick.color": SLATE, "axes.titlesize": 14,
                     "axes.titleweight": "bold", "axes.titlecolor": NAVY, "legend.frameon": False})

CLIM = R.CLIMATES["phys"]
TP = CLIM["T_present"]
run = json.load(open(f"{RUN}/history.json"))
fin = json.load(open(f"{RUN}/final_terms.json"))
T_best = np.array(fin["T"])
ka_n = (1950 - np.array(run["node_times"])) / 1000
lo, hi = map(np.array, run["bounds"])
YD = (ka_n <= 12.9) & (ka_n >= 11.7)

nc = netCDF4.Dataset(f"data/Argentiere/argentiere_{DX}m.nc")
g = lambda k: np.asarray(nc[k][:])
xg, yg, bed = g("x") / 1000, g("y") / 1000, g("topg")
lines = {v[5:]: g(v) > .5 for v in nc.variables if v.startswith("line_")}
lia_id = g("lia_id").astype(int)
nc.close()

geo, obs = load_argentiere(DX, mask=R.WINDOWS["lg"]["mask"], valley=True)
w = R.WINDOWS["lg"]
obs = select(obs, w["t_start"], w["t_end"], w["dt_snap"])
by = {o["id"]: o for o in obs}
FL = geo["flowlines"]
topo_t = geo["topg_t"].cpu()
vf = np.load(f"{RUN}/video_fields.npz")            # best iterate, every 25 yr
v_ka = (1950 - vf["times"]) / 1000


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def hillshade(az=315, alt=45):
    return LightSource(azdeg=az, altdeg=alt).hillshade(bed, vert_exag=1, dx=DX, dy=DX)


def to_t(a):
    return torch.as_tensor(a).cpu() if torch.is_tensor(a) else torch.as_tensor(np.asarray(a))


def obs_t(o):
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in o.items()}


def L_of(o, H):
    return float(terminus_position(torch.tensor(H, dtype=torch.float32), to_t(o["path"]),
                                   to_t(o["s"]).float()))


def zs_of(o, H):
    return axis_surface(torch.tensor(H, dtype=torch.float32), topo_t, to_t(o["axis"])).numpy()


# ------------------------------------------------------------------ 1 site map
def fig_site():
    nc = netCDF4.Dataset(f"data/Argentiere/argentiere_{DX}m.nc")
    x, y = np.asarray(nc["x"][:]) / 1000, np.asarray(nc["y"][:]) / 1000
    masks = {k: np.asarray(nc[k][:]) for k in ("catchment_lateglacial", "lia_arve", "argentiere_2012")}
    nc.close()
    hs = LightSource(azdeg=315, altdeg=40).hillshade(bed[::-1], vert_exag=1.5, dx=DX, dy=DX)
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.imshow(hs, cmap="gray", extent=[x[0], x[-1], y[0], y[-1]], vmin=-.1, vmax=1.15)
    ax.contour(x, y, masks["catchment_lateglacial"], [.5], colors=MOSS, linewidths=2)
    ax.contour(x, y, masks["lia_arve"], [.5], colors=BLUE, linewidths=.8)
    ax.contour(x, y, masks["argentiere_2012"], [.5], colors=NAVY, linewidths=1.2)
    tab = pd.read_csv(f"data/Argentiere/argentiere_observations_{DX}m.csv").set_index("id")
    ids = [o["id"] for o in obs if o["kind"] != "preserve"]
    t = tab.loc[[i for i in ids if i in tab.index]]
    sty = {"lateral": dict(marker="^", c=EMBER, s=70, label="lateral moraine"),
           "frontal": dict(marker="*", c=GOLD, s=260, label="frontal moraine (J2)"),
           "icefree": dict(marker="o", c="white", s=45, label="bedrock / erratic (exposure age)")}
    for k, st in sty.items():
        d = t[t.kind == k]
        ax.scatter(d.x / 1000, d.y / 1000, edgecolor="k", linewidths=.7, zorder=5, **st)
    for name, (px, py) in {"La Joux": (333.4, 5094.9), "Lognan": (341.2, 5093.0), "Talèfre": (344.8, 5084.0),
                           "Mer de Glace": (335.6, 5082.2), "Argentière\nglacier": (346.5, 5089.6)}.items():
        ax.text(px, py, name, fontsize=11, fontweight="bold", color=NAVY,
                bbox=dict(fc="white", ec="none", alpha=.75, pad=1.5))
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], c=MOSS, lw=2), Line2D([], [], c=BLUE, lw=.8), Line2D([], [], c=NAVY, lw=1.2)]
    l += ["where ice may grow", "LIA extent (start glacier)", "Argentière 2012"]
    ax.legend(h, l, loc="lower left", fontsize=9, frameon=True, facecolor="white", framealpha=.9)
    ax.set_xlim(331.5, x[-1]); ax.set_ylim(5076, 5099); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "site.png")


# ------------------------------------------------------------- 2 the control
def fig_control():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[2.2, 1]),
                               constrained_layout=True)
    a.axvspan(12.9, 11.7, color=ICE, zorder=0)
    a.text(12.3, T_best.min() - .7, "Younger Dryas", ha="center", color=SLATE, fontsize=11)
    a.plot(ka_n, T_best, "-", c=EMBER, lw=1.5)
    a.plot(ka_n, T_best, "o", c=EMBER, ms=6, mec="white", mew=.8)
    a.axhline(TP, c=NAVY, ls=":", lw=1)
    a.text(9.05, TP + .2, "2012–2021", ha="right", color=NAVY, fontsize=10)
    a.invert_xaxis(); a.set_xlabel("ka BP"); a.set_ylabel("T at 2000 m (°C)")
    a.set_ylim(T_best.min() - 1, TP + 1.6)
    a.set_title("24 unknowns: T at 2000 m every 200 yr", loc="left")
    # zoom: two nodes, linear in between, SMB recomputed every 10 yr
    i = 12
    t0, t1 = ka_n[i] * 1000, ka_n[i + 1] * 1000
    tt = np.linspace(t0, t1, 21)
    Tl = np.interp(tt, [t1, t0], [T_best[i + 1], T_best[i]])
    b.step(tt, Tl, where="post", c=BLUE, lw=1.4, label="SMB update (10 yr)")
    b.plot([t0, t1], [T_best[i], T_best[i + 1]], "--", c=EMBER, lw=1, label="linear between nodes")
    b.plot([t0, t1], [T_best[i], T_best[i + 1]], "o", c=EMBER, ms=8, mec="white")
    b.invert_xaxis(); b.set_xlabel("years BP"); b.set_title("between two nodes", loc="left")
    b.legend(fontsize=10, loc="best")
    save(fig, "control.png")


# ------------------------------------------------------------------ 3 SMB
def smb(z, T2000):
    P = CLIM["precip"] * (1 + CLIM["precip_cc"]) ** (T2000 - TP)
    return update_smb_PDD(torch.tensor(z, dtype=torch.float32), precipitation=P, T_m_lowest=T2000,
                          T_s=CLIM["T_s"], melt_factor=CLIM["melt_factor"], z_ref=2000.0).numpy(), P


def fig_smb():
    T_yd = float(T_best[YD].mean())
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    z0 = 2400.0
    m = np.linspace(0, 12, 400)
    for T2000, lab, c, ls in ((TP, "2012–2021", EMBER, "-"), (T_yd, "Younger Dryas", BLUE, "-")):
        Tm = T2000 - 7.0e-3 * (z0 - 2000)
        Ty = Tm - CLIM["T_s"] * np.cos(2 * np.pi * (m - 0.5) / 12)
        a.plot(m, Ty, c=c, lw=2.2, ls=ls, label=f"{lab}: T_m = {Tm:+.1f} °C")
        a.fill_between(m, 0, Ty, where=Ty > 0, color=EMBER, alpha=.35, lw=0)
        a.axhline(Tm, c=c, lw=.8, ls=":")
    a.fill_between(m, -25, 0, color=MELT, alpha=.15, lw=0)
    a.axhline(0, c=NAVY, lw=.8)
    Tm0 = TP - 7.0e-3 * (z0 - 2000)
    a.annotate("", xy=(6.5, Tm0 + CLIM["T_s"]), xytext=(6.5, Tm0), arrowprops=dict(arrowstyle="<->", color=NAVY))
    a.text(6.3, Tm0 + 1.5, "T_s", color=NAVY, fontsize=13, fontweight="bold", va="center", ha="right")
    a.text(0.3, -18.5, "below 0 °C: precipitation falls as snow", color=SLATE, fontsize=10)
    a.text(8.3, 3.8, "melt ∝ DDF × area", color=EMBER, fontsize=10)
    a.set_xticks(np.arange(12) + .5); a.set_xticklabels(list("JFMAMJJASOND"))
    a.set_ylim(-20, 9); a.set_xlim(0, 12); a.set_ylabel("air temperature (°C)")
    a.set_title(f"One year at {z0:.0f} m", loc="left"); a.legend(fontsize=10, loc="upper left")

    z = np.linspace(1000, 4300, 300)
    for T2000, lab, c in ((TP, "2012–2021", EMBER), (T_yd, "Younger Dryas", BLUE)):
        s, P = smb(z, T2000)
        b.plot(s, z, c=c, lw=2.2, label=f"{lab} (P = {P:.2f} m/yr)")
        ela = z[np.argmin(abs(s))]
        b.axhline(ela, c=c, lw=.8, ls="--")
        b.text(-9.6, ela + 40, f"ELA {ela:.0f} m", color=c, fontsize=10)
    b.axvline(0, c=NAVY, lw=.8)
    b.set_xlim(-10, 2.5); b.set_xlabel("surface mass balance (m ice/yr)"); b.set_ylabel("elevation (m)")
    b.set_title("Mass balance against elevation", loc="left"); b.legend(fontsize=10, loc="lower right")
    save(fig, "smb.png")


# ---------------------------------------------------------- 4 flowline map
def fig_flowlines():
    hs = hillshade()
    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    ax.imshow(hs[::-1], extent=[xg[0], xg[-1], yg[0], yg[-1]], cmap="gray", vmin=-.2, vmax=1.15)
    cols = {"Argentiere": GOLD, "Talefre": "#E377C2", "Mer de Glace": "#7FD17F", "Tour": "#C9B6E4"}
    for name, fl in FL.items():
        iy, ix = np.unravel_index(fl.cells, bed.shape)
        ax.plot(xg[ix], yg[iy], "-", c=cols[name], lw=.8)
        ax.plot(xg[ix[fl.start:fl.stop]], yg[iy[fl.start:fl.stop]], "-", c=cols[name], lw=3,
                label=name.replace("Argentiere", "Argentière").replace("Talefre", "Talèfre"))
    for L in lines.values():
        ax.contour(xg, yg, L.astype(float), [.5], colors=EMBER, linewidths=1.4)
    # every landform is projected onto its cross-section of the valley axis
    tab = pd.read_csv(f"data/Argentiere/argentiere_observations_{DX}m.csv").set_index("id")
    for o in obs:
        if o["kind"] == "preserve" or o["id"] not in tab.index:
            continue
        px, py = tab.loc[o["id"], "x"] / 1000, tab.loc[o["id"], "y"] / 1000
        axis = np.atleast_1d(o["axis"].cpu().numpy() if torch.is_tensor(o["axis"]) else o["axis"])
        ay, ax_ = np.unravel_index(int(axis[len(axis) // 2]), bed.shape)
        ax.plot([px, xg[ax_]], [py, yg[ay]], "-", c="white", lw=1, alpha=.9)
        ax.plot(xg[ax_], yg[ay], "o", c="white", ms=3)
        if o["kind"] == "icefree_v":
            ax.plot(px, py, "^", c="white", mec=NAVY, ms=7)
    j2 = by["ICEM-Protin2019-E"]; fa = FL["Argentiere"]
    k = int(np.argmin(abs(fa.s[:fa.stop] - j2["s_obs"])))
    iy, ix = np.unravel_index(fa.cells[k], bed.shape)
    ax.plot(xg[ix], yg[iy], "*", c=EMBER, mec="k", ms=17)
    ax.text(xg[ix] + .4, yg[iy] - .2, "J2 La Joux", color="white", fontsize=11, fontweight="bold")
    h = ax.get_legend_handles_labels()
    ax.legend(h[0] + [Line2D([], [], c=EMBER, lw=1.4), Line2D([], [], ls="none", marker="^", c="white", mec=NAVY),
                      Line2D([], [], c="white", lw=1), Line2D([], [], c=MELT, lw=1)],
              h[1] + ["moraine crest", "bedrock / erratic sample", "projection onto the axis"],
              loc="lower left", fontsize=9, frameon=True, facecolor="white", framealpha=.85)
    ax.set_xlim(333, 352); ax.set_ylim(5080, 5099); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "flowlines.png")


# ------------------------------------------------------- 5 scoring one run
def fig_scoring():
    fa = FL["Argentiere"]
    cells = fa.cells[:fa.stop]; s_km = fa.s[:fa.stop] / 1000; bedf = bed.ravel()[cells]
    pos = {int(c): float(si) for c, si in zip(cells, fa.s[:fa.stop])}
    j2 = by["ICEM-Protin2019-E"]; l1 = by["ICEM-Protin2019-A"]; bd = by["BED-Protin2019-B"]
    later = ["ICEM-Protin2019-A", "ICEM-Protin2019-B", "ICEM-Protin2019-C", "ICEM-Protin2019-D", "undated-LaJoux"]
    # series over the run
    L_j2 = np.array([L_of(j2, H) for H in vf["H"]])
    dz_l1 = np.array([float((zs_of(l1, H) - l1["z"].cpu().numpy() - l1["delta"]).mean()) for H in vf["H"]])
    L_l1 = np.array([L_of(l1, H) for H in vf["H"]])
    dz_bd = np.array([float((zs_of(bd, H) - bd["z"].cpu().numpy() - bd["delta"]).mean()) for H in vf["H"]])
    win = lambda o: ((1950 - o["t0"]) / 1000, (1950 - o["t1"]) / 1000 if np.isfinite(o["t1"]) else 9.0)
    ina = (v_ka <= win(j2)[0]) & (v_ka >= win(j2)[1])
    kbest = np.flatnonzero(ina)[np.argmin(abs(L_j2[ina] - j2["s_obs"]))]
    H = vf["H"][kbest]; ka_b = v_ka[kbest]

    fig = plt.figure(figsize=(13, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
    a = fig.add_subplot(gs[0, :])
    h = np.maximum(H.ravel()[cells], 0); surf = bedf + h; ice = h > 1
    a.fill_between(s_km, bedf - 400, bedf, color="#D9D9D9", lw=0)
    a.fill_between(s_km, bedf, np.where(ice, surf, bedf), color=MELT, alpha=.6, lw=0)
    a.plot(s_km, np.where(ice, surf, np.nan), c=BLUE, lw=2, label=f"modelled ice surface, {ka_b:.1f} ka")
    a.plot(s_km, bedf, c="#666", lw=.8)
    for oid in later:
        o = by[oid]
        sp = [pos[int(c)] / 1000 for c in o["axis"].cpu().numpy()]
        zt = o["z"].cpu().numpy() + o["delta"]
        a.errorbar(sp, zt, yerr=o["sigma_z"], fmt="o", c=EMBER, ms=4, lw=.8, capsize=2,
                   label="lateral crest + 10 m (± 60 m)" if oid == later[0] else None, zorder=5)
    for o in obs:
        if o["kind"] == "icefree_v" and o["valley"] == "Argentiere":
            sp = pos.get(int(np.atleast_1d(o["axis"])[0]))
            if sp is not None:
                a.plot(sp / 1000, o["z"][0] + o["delta"], "v", c=NAVY, ms=8, zorder=6,
                       label="bedrock sample: ice must stay below" if o is bd else None)
    a.axvspan((j2["s_obs"] - j2["sigma_L"]) / 1000, (j2["s_obs"] + j2["sigma_L"]) / 1000, color=GOLD, alpha=.3, lw=0)
    a.plot(j2["s_obs"] / 1000, np.interp(j2["s_obs"] / 1000, s_km, bedf), "*", c=EMBER, mec="k", ms=18, zorder=7,
           label="J2 frontal moraine (± 300 m)")
    Lb = L_j2[kbest] / 1000
    a.plot(Lb, np.interp(Lb, s_km, bedf) + 90, "v", c="white", mec=NAVY, mew=1.5, ms=12, zorder=7)
    a.text(Lb, np.interp(Lb, s_km, bedf) + 160, "terminus L", ha="center", fontsize=11, color=NAVY)
    a.set_xlim(5, s_km[-1]); a.set_ylim(1000, 3000)
    a.set_xlabel("distance down the Argentière flowline (km)"); a.set_ylabel("elevation (m)")
    a.set_title("Space: every landform is read on the valley axis", loc="left")
    a.legend(loc="upper right", fontsize=10, ncol=2)

    def panel(ax, y, o, title, ylab, sig, one_sided=False, sc=1.0):
        t0, t1 = win(o)
        ax.axvspan(t0, t1, color=ICE, zorder=0)
        ax.axhspan(-sig if not one_sided else -1e4, sig if not one_sided else 0, color=MOSS, alpha=.15, lw=0)
        ax.axhline(0, c=MOSS, lw=1)
        ax.plot(v_ka, y * sc, c=NAVY, lw=1.6)
        ax.invert_xaxis(); ax.set_xlim(13.6, 9.0); ax.set_xlabel("ka BP"); ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=12)

    b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1]); d = fig.add_subplot(gs[1, 2])
    panel(b, L_j2 - j2["s_obs"], j2, "J2 frontal: best time in its window", "L − moraine (km)", .3, sc=1e-3)
    b.set_ylim(-6, 1.5)
    panel(c, dz_l1, l1, "Lognan L1 lateral: best time in window", "surface − crest (m)", 60)
    c.set_ylim(-250, 250)
    panel(d, dz_bd, bd, "Bedrock: every time after exposure", "surface − sample (m)", 60, one_sided=True)
    d.set_ylim(-400, 200)
    d.legend([Patch(color=ICE), Patch(color=MOSS, alpha=.3)], ["age window", "fit (± σ)"], fontsize=9, loc="lower left")
    b.axvline(ka_b, c=GOLD, lw=1.2, ls="--")
    box = dict(fc="white", ec=GREY, lw=.6, pad=3)
    b.text(.03, .04, "softmin over t in window\n((L − s_moraine) / 300 m)²", transform=b.transAxes, fontsize=10, bbox=box)
    c.text(.03, .04, "softmin over t in window\nreach² + ((z − crest − 10) / 60 m)²", transform=c.transAxes, fontsize=10, bbox=box)
    d.text(.97, .96, "mean over t after exposure\n(max(0, z − sample − 10) / 60 m)²", transform=d.transAxes, fontsize=10,
           bbox=box, ha="right", va="top")
    save(fig, "scoring.png")


# ---------------------------------------------------------------- 6 result
def fig_result():
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.fill_between(ka_n, lo, hi, step="mid", color=MELT, alpha=.35, lw=0, label="literature envelope (bounds)")
    ax.text(12.3, T_best[YD].mean() + .35, f"Younger Dryas mean {T_best[YD].mean() - TP:+.2f} °C", ha="center",
            color=EMBER, fontsize=11, fontweight="bold")
    ax.plot(ka_n, T_best, "o-", c=EMBER, ms=5, lw=2, label="inverted T(t)")
    ax.hlines(T_best[YD].mean(), 12.9, 11.7, color=EMBER, lw=1, ls="--")
    ax.axhline(TP, c=NAVY, ls=":", lw=1, label="2012–2021")
    ax.axhline(TP - R.PRESENT_MINUS_PREINDUSTRIAL, c=GREY, ls="--", lw=1, label="pre-industrial")
    ax.invert_xaxis(); ax.set_xlabel("ka BP"); ax.set_ylabel("T at 2000 m (°C)")
    ax.set_ylim(lo.min() - .3, hi.max() + .7)
    sec = ax.secondary_yaxis("right", functions=(lambda T: T - TP, lambda a: a + TP))
    sec.set_ylabel("anomaly to 2012–2021 (°C)", color=NAVY); sec.tick_params(colors=SLATE)
    sec.spines["right"].set_visible(True)
    ax.legend(fontsize=10, loc="lower right", ncol=2)
    save(fig, "result.png")


# ---------------------------------------------------------------- 7 stands
def fig_stands():
    fa = FL["Argentiere"]; j2 = by["ICEM-Protin2019-E"]
    hs = LightSource(azdeg=180, altdeg=40).hillshade(bed, vert_exag=1, dx=DX, dy=DX)
    flip = yg[0] > yg[-1]; ext = [xg[0], xg[-1], min(yg), max(yg)]
    snap = lambda ka: vf["H"][int(np.argmin(abs(v_ka - ka)))]
    path = torch.tensor(fa.active[fa.start:]); s = torch.tensor(fa.s[fa.start: fa.stop], dtype=torch.float32)
    term = lambda H: float(terminus_position(torch.tensor(H, dtype=torch.float32), path, s))
    stands = [(12.5, "12.5 ka · Younger Dryas"), (11.66, "11.7 ka · Lognan L1"),
              (10.9, "10.9 ka · La Joux J2"), (10.35, "10.3 ka · Lognan L2/L3")]
    iy, ix = np.unravel_index(fa.cells[:fa.stop], bed.shape)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
    for ax, (ka, ttl) in zip(axes, stands):
        H = snap(ka)
        ax.imshow(hs if flip else hs[::-1], extent=ext, cmap="gray", vmin=0, vmax=1.3)
        Hm = np.where(H > 1, H, np.nan)
        im = ax.imshow(Hm if flip else Hm[::-1], extent=ext, cmap="Blues", vmin=0, vmax=400, alpha=.9)
        for L in lines.values():
            ax.contour(xg, yg, L, levels=[.5], colors=EMBER, linewidths=1.2)
        k = int(np.argmin(abs(fa.s[:fa.stop] - term(H))))
        ax.plot(xg[ix[k]], yg[iy[k]], marker="v", c="k", mfc="white", ms=10, mew=1.4, ls="none")
        kj = int(np.argmin(abs(fa.s[:fa.stop] - j2["s_obs"])))
        ax.plot(xg[ix[kj]], yg[iy[kj]], marker="*", c=EMBER, mec="k", ms=14, ls="none")
        ax.set_xlim(xg[ix.min()] - 3.5, xg[ix.max()] + 2.5)
        ax.set_ylim(sorted([yg[iy].min() - 3.0, yg[iy].max() + 3.0])); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(ttl, fontsize=13)
    fig.colorbar(im, ax=axes, shrink=.75, label="ice thickness (m)")
    save(fig, "stands.png")


# ---------------------------------------------------- 11 best fit (sliding)
NAMES = {"ICEM-Protin2019-A": "Lognan L1", "ICEM-Protin2019-B": "Lognan L2", "ICEM-Protin2019-C": "Lognan L3",
         "ICEM-Protin2019-D": "La Joux", "undated-LaJoux": "La Joux (undated)", "undated-Lognan": "Lognan (undated)",
         "undated-Lognan-2": "Lognan (undated 2)", "ICEM-Protin2021-A": "Talèfre T1", "ICEM-Protin2021-B": "Talèfre T2",
         "BED-Lehmann2020-F": "Trélaporte F (MdG)", "BED-Lehmann2020-C": "Aig. du Moine C (MdG)",
         "ERR-Protin2019-A": "La Joux erratic", "BED-Protin2019-E": "Lognan E", "BED-Protin2019-C": "Lognan C",
         "BED-Lehmann2020-B": "Aig. du Moine B (MdG)", "BED-Lehmann2020-G": "Trélaporte G (MdG)",
         "BED-Protin2019-B": "Lognan B"}


def fig_bestfit():
    S = f"{RUN}/sensitivity"
    row0 = lambda f: next(r for r in json.load(open(f"{S}/{f}.json"))["rows"] if abs(r["dT"]) < 1e-6)
    b, t = row0("base"), row0("tau-")
    lat = ["undated-LaJoux", "ICEM-Protin2019-D", "ICEM-Protin2019-A", "ICEM-Protin2019-C", "ICEM-Protin2019-B",
           "undated-Lognan", "undated-Lognan-2", "ICEM-Protin2021-A", "ICEM-Protin2021-B"]
    bed_ids = ["BED-Lehmann2020-F", "BED-Lehmann2020-C", "ERR-Protin2019-A", "BED-Protin2019-E", "BED-Protin2019-C",
               "BED-Lehmann2020-B", "BED-Lehmann2020-G", "BED-Protin2019-B"]
    fig, (a, c) = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for ax, ids, title in ((a, lat, "Lateral moraines: surface − (crest + 10 m)"),
                           (c, bed_ids, "Bedrock / erratic: surface − (sample + 10 m)")):
        y = np.arange(len(ids))[::-1]
        if ax is a:
            ax.axvspan(-60, 60, color=MOSS, alpha=.15, lw=0)
        else:
            ax.axvspan(-200, 0, color=MOSS, alpha=.15, lw=0)
        ax.axvline(0, c=MOSS, lw=1)
        for yi, k in zip(y, ids):
            z0, z1 = b["describe"][k]["dz_m"], t["describe"][k]["dz_m"]
            ax.annotate("", xy=(z1, yi), xytext=(z0, yi), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
            ax.plot(z0, yi, "o", c=EMBER, ms=9, zorder=5)
            ax.plot(z1, yi, "o", c=BLUE, ms=9, zorder=5)
        ax.set_yticks(y); ax.set_yticklabels([NAMES[k] for k in ids])
        ax.set_xlabel("m"); ax.set_title(title, loc="left", fontsize=13)
    a.set_xlim(-80, 120); c.set_xlim(-120, 170)
    a.plot([], [], "o", c=EMBER, label="literature τ_ref 0.215 MPa"); a.plot([], [], "o", c=BLUE, label="τ_ref 0.15 MPa (more sliding)")
    a.plot([], [], "s", c=MOSS, alpha=.3, ms=10, label="fit (± σ / below the sample)")
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, -.08), ncol=3, fontsize=11)
    save(fig, "bestfit.png")


# ------------------------------------------------- 12-13 joint inversion
JOINT = "results/argentiere_lg_100m_joint"
JNAMES = {"precip": "P", "T_s": "T_s", "ddf": "DDF", "arrhenius": "A (flow law)", "tau_ref": "τ_ref (sliding)"}


def fig_joint_controls():
    H = json.load(open(f"{JOINT}/history.json")); h = H["history"]; n = H["n_nodes"]
    z = np.array(h["z"]); dz = z - z[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.15]))
    a.plot(h["loss"], "o-", c=NAVY, ms=4, lw=2, label="loss")
    a.axhline(h["loss"][0], c=GREY, ls=":", lw=1)
    a.text(29, h["loss"][0] + .3, "fixed parameters", ha="right", color=GREY, fontsize=11)
    a.set_xlabel("iteration"); a.set_ylabel("loss"); a.set_title("Loss", loc="left")
    cols = {"tau_ref": BLUE, "arrhenius": EMBER, "precip": GOLD, "ddf": MOSS, "T_s": MELT}
    for k, key in enumerate(H["keys"]):
        b.plot(dz[:, n + k], c=cols[key], lw=2.2, label=JNAMES[key])
    b.plot(np.abs(dz[:, :n]).mean(1), c=NAVY, lw=2, ls="--", label="T(t) nodes, mean |Δ|")
    b.axhline(0, c=GREY, lw=.8)
    b.set_xlabel("iteration"); b.set_ylabel("change, fraction of the allowed range")
    b.set_title("How far each control moved (same units for all)", loc="left")
    b.legend(fontsize=10, loc="lower left", ncol=2)
    save(fig, "joint_controls.png")


def fig_joint_result():
    H = json.load(open(f"{JOINT}/history.json")); bst = H["best"]; s0 = H["history"]["state"][0]
    dj = json.load(open(f"{JOINT}/final_terms.json"))["describe"]; d0 = fin["describe"]
    fig, (a, c) = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.25, 1]))
    lo_a, hi_a = lo - TP, hi - TP
    a.fill_between(ka_n, lo_a, hi_a, step="mid", color=MELT, alpha=.3, lw=0, label="literature envelope")
    a.axvspan(12.9, 11.7, color=ICE, zorder=0)
    a.plot(ka_n, np.array(s0["T"]) - s0["T_present"], "o--", c=GREY, ms=4, lw=1.5, label="fixed parameters")
    a.plot(ka_n, np.array(bst["T"]) - bst["T_present"], "o-", c=EMBER, ms=5, lw=2, label="joint inversion")
    a.axhline(0, c=NAVY, ls=":", lw=1); a.axhline(-R.PRESENT_MINUS_PREINDUSTRIAL, c=GREY, ls="--", lw=1)
    a.text(12.3, .15, "Younger Dryas", ha="center", color=SLATE, fontsize=11)
    a.invert_xaxis(); a.set_xlabel("ka BP"); a.set_ylabel("anomaly to 2012–2021 (°C)")
    a.set_title("Temperature", loc="left"); a.legend(fontsize=10, loc="lower left")
    ids = ["undated-LaJoux", "ICEM-Protin2019-D", "ICEM-Protin2019-A", "ICEM-Protin2019-C", "ICEM-Protin2021-A",
           "BED-Lehmann2020-F", "BED-Lehmann2020-C", "ERR-Protin2019-A"]
    y = np.arange(len(ids))[::-1]
    c.axvspan(-60, 60, color=MOSS, alpha=.15, lw=0); c.axvline(0, c=MOSS, lw=1)
    c.text(0, -0.75, "target ± 60 m (1σ)", ha="center", color=MOSS, fontsize=10)
    for yi, k in zip(y, ids):
        z0, z1 = d0[k]["dz_m"], dj[k]["dz_m"]
        c.annotate("", xy=(z1, yi), xytext=(z0, yi), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
        c.plot(z0, yi, "o", c=GREY, ms=9, zorder=5); c.plot(z1, yi, "o", c=EMBER, ms=9, zorder=5)
    c.set_ylim(-1.1, len(ids) - .5); c.set_yticks(y); c.set_yticklabels([NAMES[k] for k in ids]); c.set_xlabel("ice surface above the target (m)")
    c.set_title("Ice too high at the landforms", loc="left")
    save(fig, "joint_result.png")


def fig_scoring_compare():
    """Slide-8 figure, fixed parameters (grey) vs joint inversion (red), plus where the loss went."""
    jf = np.load(f"{JOINT}/final_fields.npz"); j_ka = (1950 - jf["times"]) / 1000
    dj = json.load(open(f"{JOINT}/final_terms.json"))
    fa = FL["Argentiere"]
    cells = fa.cells[:fa.stop]; s_km = fa.s[:fa.stop] / 1000; bedf = bed.ravel()[cells]
    pos = {int(c): float(si) for c, si in zip(cells, fa.s[:fa.stop])}
    j2 = by["ICEM-Protin2019-E"]; l1 = by["ICEM-Protin2019-A"]; bd = by["BED-Protin2019-B"]
    later = ["ICEM-Protin2019-A", "ICEM-Protin2019-B", "ICEM-Protin2019-C", "ICEM-Protin2019-D", "undated-LaJoux"]
    win = lambda o: ((1950 - o["t0"]) / 1000, (1950 - o["t1"]) / 1000 if np.isfinite(o["t1"]) else 9.0)
    runs = []
    for Hs, ka, col, lab in ((vf["H"], v_ka, GREY, "T(t) only"), (jf["H"], j_ka, EMBER, "joint (T(t) + parameters)")):
        L = np.array([L_of(j2, H) for H in Hs])
        dz1 = np.array([float((zs_of(l1, H) - l1["z"].cpu().numpy() - l1["delta"]).mean()) for H in Hs])
        dzb = np.array([float((zs_of(bd, H) - bd["z"].cpu().numpy() - bd["delta"]).mean()) for H in Hs])
        ina = (ka <= win(j2)[0]) & (ka >= win(j2)[1])
        kb = np.flatnonzero(ina)[np.argmin(abs(L[ina] - j2["s_obs"]))]
        runs.append(dict(H=Hs[kb], ka=ka, kb=ka[kb], L=L, dz1=dz1, dzb=dzb, col=col, lab=lab))
        h = np.maximum(Hs[kb].ravel()[cells], 0)
        print(f"{lab}: J2 met at {ka[kb]:.2f} ka, mean thickness on the flowline (ice cells) {h[h > 1].mean():.0f} m")

    fig = plt.figure(figsize=(14, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1])
    a = fig.add_subplot(gs[0, :])
    a.fill_between(s_km, bedf - 400, bedf, color="#D9D9D9", lw=0)
    for r in runs:
        h = np.maximum(r["H"].ravel()[cells], 0); ice = h > 1
        surf = np.where(ice, bedf + h, np.nan)
        if r["col"] == GREY:
            a.fill_between(s_km, bedf, np.where(ice, bedf + h, bedf), color=MELT, alpha=.45, lw=0)
        a.plot(s_km, surf, c=r["col"], lw=2.2, label=f"{r['lab']}, {r['kb']:.1f} ka")
    a.plot(s_km, bedf, c="#666", lw=.8)
    dzs = (np.maximum(runs[1]["H"].ravel()[cells], 0) - np.maximum(runs[0]["H"].ravel()[cells], 0))[(s_km > 6) & (s_km < 13)]
    a.text(9.6, 2350, f"joint surface {-dzs.mean():.0f} m lower\n(mean, 6–13 km)", color=EMBER, fontsize=12, fontweight="bold")
    for oid in later:
        o = by[oid]
        sp = [pos[int(c)] / 1000 for c in o["axis"].cpu().numpy()]
        a.errorbar(sp, o["z"].cpu().numpy() + o["delta"], yerr=o["sigma_z"], fmt="o", c=NAVY, ms=4, lw=.8, capsize=2,
                   label="lateral crest + 10 m (± 60 m)" if oid == later[0] else None, zorder=5)
    for o in obs:
        if o["kind"] == "icefree_v" and o["valley"] == "Argentiere":
            sp = pos.get(int(np.atleast_1d(o["axis"])[0]))
            if sp is not None:
                a.plot(sp / 1000, o["z"][0] + o["delta"], "v", c=NAVY, ms=8, zorder=6,
                       label="bedrock sample: ice must stay below" if o is bd else None)
    a.axvspan((j2["s_obs"] - j2["sigma_L"]) / 1000, (j2["s_obs"] + j2["sigma_L"]) / 1000, color=GOLD, alpha=.3, lw=0)
    a.plot(j2["s_obs"] / 1000, np.interp(j2["s_obs"] / 1000, s_km, bedf), "*", c=GOLD, mec="k", ms=18, zorder=7,
           label="J2 frontal moraine (± 300 m)")
    a.set_xlim(5, s_km[-1]); a.set_ylim(1000, 3000)
    a.set_xlabel("distance down the Argentière flowline (km)"); a.set_ylabel("elevation (m)")
    a.set_title("Same terminus, lower surface", loc="left")
    a.legend(loc="upper right", fontsize=10, ncol=2)

    def panel(ax, key, o, title, ylab, sig, one_sided=False, sc=1.0, off=0.0):
        t0, t1 = win(o)
        ax.axvspan(t0, t1, color=ICE, zorder=0)
        ax.axhspan(-sig if not one_sided else -1e4, sig if not one_sided else 0, color=MOSS, alpha=.15, lw=0)
        ax.axhline(0, c=MOSS, lw=1)
        for r in runs:
            ax.plot(r["ka"], (r[key] - off) * sc, c=r["col"], lw=1.8)
        ax.invert_xaxis(); ax.set_xlim(13.6, 9.0); ax.set_xlabel("ka BP"); ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=12)

    b, c, d, e = (fig.add_subplot(gs[1, k]) for k in range(4))
    panel(b, "L", j2, "J2 frontal", "L − moraine (km)", .3, sc=1e-3, off=j2["s_obs"]); b.set_ylim(-6, 1.5)
    panel(c, "dz1", l1, "Lognan L1 lateral", "surface − crest (m)", 60); c.set_ylim(-250, 250)
    panel(d, "dzb", bd, "Bedrock sample", "surface − sample (m)", 60, one_sided=True); d.set_ylim(-400, 200)

    def grp(k):
        if k.endswith("-preserve"): return "preserved"
        if k.startswith(("BED", "ERR")): return "bedrock, erratic"
        return "frontal" if k == "ICEM-Protin2019-E" else "lateral"
    groups = ["lateral", "bedrock, erratic", "preserved", "frontal"]
    meta = fin["meta"]                     # each landform is its own group: contribution = weight * huber(term)
    contrib = lambda t: {k: meta[k]["weight"] * float(huber(torch.tensor(v))) for k, v in t.items()}
    sums = [[sum(v for k, v in contrib(t).items() if grp(k) == g) for g in groups] for t in (fin["terms"], dj["terms"])]
    loss0 = json.load(open(f"{JOINT}/history.json"))["history"]["loss"][0]     # fixed-parameter optimum
    groups.append("T(t) smoothness")
    sums[0].append(loss0 - sum(sums[0])); sums[1].append(dj["loss"] - sum(sums[1]))
    y = np.arange(len(groups))[::-1]
    e.barh(y + .2, sums[0], .4, color=GREY, label="T(t) only"); e.barh(y - .2, sums[1], .4, color=EMBER, label="joint")
    for yi, s0, s1 in zip(y, *sums):
        e.text(max(s0, s1) + .3, yi, f"{s1 - s0:+.1f}", va="center", fontsize=11, color=EMBER, fontweight="bold")
    e.set_yticks(y); e.set_yticklabels([g.replace("\n", " ") for g in groups], fontsize=11)
    e.set_xlim(0, 13.5); e.set_xlabel("loss")
    e.set_title("Where the loss went", loc="left", fontsize=12); e.legend(fontsize=9, loc="lower right")
    print("loss by group", [f"{g!r} {a:.2f} -> {b:.2f}" for g, a, b in zip(groups, *sums)], "data", sum(sums[0]), sum(sums[1]))
    save(fig, "scoring_compare.png")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["site", "control", "smb", "flowlines", "scoring", "result", "stands", "bestfit"]
    for n in todo:
        globals()[f"fig_{n}"]()

"""Videos of the Argentière inversions: the inverted temperature signal on top, the glacier flowing over the
DEM and the dated constraints below, both at the same model time. ``--window lg`` (Lateglacial, moraines, zoom on
La Joux) or ``--window hp`` (late Holocene, Mer de Glace wood, LIA and 2012/2021 outlines, zoom on Lognan).

    python tools/make_argentiere_video.py fields [--window hp --every 20]   # forward run -> <run>/video_fields.npz
    python tools/make_argentiere_video.py fields_joint --run results/argentiere_lg_100m_joint  # best joint iterate
    python tools/make_argentiere_video.py frame --t_yr 100       # one PNG, t years after the window start
    python tools/make_argentiere_video.py frames                 # one PNG per saved time step, in <run>/video/
    python tools/make_argentiere_video.py video [--fps 6]        # the frames -> mp4 with ffmpeg (imageio-ffmpeg)

Run it after the inversion has finished: the fields come from a forward run at the best temperatures of ``--run``
with the literature climate and the same settings as the inversion (100 m grid, valley operators, 600 m/yr cap,
CFL 0.3), saved every ``--every`` years.
"""

import argparse
import json
import os

import numpy as np

RUN = "results/argentiere_lg_100m_phys"
DX = 100                          # model grid; never 200 m unless explicitly asked


def fields(run, every, dx=DX, window="lg"):
    import torch
    import tools.run_argentiere_inversion as R
    from core.utils import device

    base = dict(R.CLIMATES["phys"])
    glacier, geo, obs, nodes, anchor = R.build(window, dx, valley=True, max_velbar=600.0, base=base)
    T = torch.tensor(json.load(open(f"{run}/final_terms.json"))["T"], dtype=torch.float32, device=device).view(-1, 1)
    w = R.WINDOWS[window]
    times = [w["t_start"] + k * every for k in range(1, int(round((w["t_end"] - w["t_start"]) / every)) + 1)]
    with torch.no_grad():
        snaps, H_end = glacier(precip_tensor=base["precip"], T_m_lowest=T, T_s=base["T_s"],
                               melt_factor=base["melt_factor"], smb_method="PDD", save_times=times)
    ts = sorted(snaps)
    H = np.stack([snaps[t].detach().cpu().numpy() for t in ts]).astype(np.float32)
    np.savez_compressed(f"{run}/video_fields.npz", times=np.array(ts), H=H, T_nodes=T.flatten().cpu().numpy(),
                        node_times=np.array(nodes), window=window)
    print(f"wrote {run}/video_fields.npz: {len(ts)} snapshots, {ts[0]:.0f} .. {ts[-1]:.0f} CE")


def fields_joint(run, every):
    """video_fields.npz of a joint inversion (run_argentiere_joint_inversion.py): its final_fields.npz is already
    the forward run at the best iterate (T(t) and parameters), so it is only put on the ``every``-year grid."""
    d = np.load(f"{run}/final_fields.npz"); best = json.load(open(f"{run}/history.json"))["best"]
    t = d["times"]; keep = np.isclose((t - t[0]) % every, 0)
    np.savez_compressed(f"{run}/video_fields.npz", times=t[keep], H=d["H"][keep], T_nodes=np.array(best["T"]),
                        node_times=np.array(best.get("node_times") or json.load(open(f"{run}/history.json"))["node_times"]),
                        window="lg", P0=best["params"]["precip"], T_present=best["T_present"])
    print(f"wrote {run}/video_fields.npz: {keep.sum()} snapshots")


#: dated moraines of the Lateglacial window: key -> short name. Protin et al. 2019, 2021.
MORAINES = {
    "ICEM_Protin2021_A": "T1, Talèfre",
    "ICEM_Protin2019_A": "L1, Lognan",
    "ICEM_Protin2021_B": "T2, Talèfre",
    "ICEM_Protin2019_E": "J2, La Joux (frontal)",
    "ICEM_Protin2019_B": "L2, Lognan",
    "ICEM_Protin2019_C": "L3, Lognan",
    "ICEM_Protin2019_D": "J1, La Joux",
    "undated_LaJoux": "undated, La Joux",
    "undated_Lognan": "undated, Lognan",
    "undated_Lognan_2": "undated, Lognan",
}
AGE_RANGE = (10.2, 12.2)          # ka BP, colour scale of the moraines: old purple, young yellow
#: map labels: name -> (label x, label y, moraine x, moraine y), metres UTM 32N
CLUSTERS = {"La Joux": (335500, 5095600, 338400, 5092800), "Lognan": (345800, 5095000, 342200, 5092500),
            "Talèfre": (346800, 5083800, 343600, 5086750)}
#: Lateglacial zoom window over La Joux (x0, x1, y0, y1), and its labels: key -> (label x, label y, text)
ZOOM = (336700, 341700, 5090900, 5095900)
ZOOM_LABELS = {"ICEM_Protin2019_E": (338900, 5091350, "J2 frontal, 10.9 ka"),
               "ICEM_Protin2019_D": (340900, 5095500, "J1 lateral, \u224812 ka"),
               "undated_LaJoux": (338000, 5095500, "undated lateral")}



#: late-Holocene window: zoom over the Argentière tongue and Lognan, map labels, Lognan moraines shown for context
ZOOM_HP = (339300, 344300, 5090300, 5095300)
CLUSTERS_HP = {"Mer de Glace wood\n(Le Roy et al. 2015)": (334200, 5085200, 339300, 5088400),
               "Talèfre T4": (346900, 5083800, 343400, 5086800), "Lognan": (345800, 5095000, 342400, 5092600)}
LOGNAN = ("ICEM_Protin2019_A", "ICEM_Protin2019_B", "ICEM_Protin2019_C", "undated_Lognan", "undated_Lognan_2")
WOOD_FREE, WOOD_COVER, T4_COL = "#e6550d", "#3182bd", "#9467bd"


def load_static(dx=DX, window="lg"):
    """DEM, hillshade and the window's constraints, on the ``dx`` grid."""
    import netCDF4
    import pandas as pd
    from matplotlib.colors import LightSource

    nc = netCDF4.Dataset(f"data/Argentiere/argentiere_{dx}m.nc")
    x, y = np.asarray(nc["x"][:]), np.asarray(nc["y"][:])
    bed = np.asarray(nc["topg"][:])
    mask = np.asarray(nc["catchment_lateglacial" if window == "lg" else "catchment_holocene"][:]) > 0.5
    keys = list(MORAINES) + ["ICEM_Protin2021_C"]
    lines = {k: np.asarray(nc["line_" + k][:]) for k in keys}
    S = dict(window=window, x=x, y=y, bed=bed, lines=lines, dx=dx, x2=x, y2=y)
    if window == "hp":
        from scipy.ndimage import binary_fill_holes, label
        lia = binary_fill_holes(np.asarray(nc["lia_arve"][:]) > .5)     # outer outline only: no nunatak rings
        lab, n = label(lia)
        sizes = np.bincount(lab.ravel())
        S["lia"] = np.isin(lab, [i for i in range(1, n + 1) if sizes[i] >= 50])
        S["arg2012"] = np.asarray(nc["argentiere_2012"][:]) > .5
        S["arg2021"] = np.asarray(nc["thk_2021"][:]) > 1
    nc.close()
    df = pd.read_csv(f"data/Argentiere/argentiere_observations_{dx}m.csv")
    d = df.set_index("key")
    S["mor"] = {}
    for k, name in MORAINES.items():
        r = d.loc[k]
        S["mor"][k] = dict(name=name, age=(1950 - r.year) / 1000, t0=r.t0, t1=r.t1, dashed=r.group == "morphostrat",
                           err=0 if pd.isna(r.err_pos) else r.err_pos / 1000)
    r = d.loc["ICEM_Protin2021_C"]
    S["t4"] = dict(year=r.year, t0=r.t0, t1=r.t1)
    w = df[df.kind.isin(["icefree_any", "covered_any"]) & df.in_catchment]
    S["wood"] = [dict(id=row.id.replace("MOR-LeRoy2015-", ""), free=row.kind == "icefree_any", x=row.x, y=row.y,
                      z=row.z, t0=row.t0, t1=row.t1) for row in w.itertuples()]
    S["hs"] = LightSource(azdeg=180, altdeg=35).hillshade(bed[::-1], vert_exag=1.5, dx=dx, dy=dx)
    iy, ix = np.nonzero(mask)
    box = (x[ix.min()] - 1500, x[ix.max()] + 1500, y[iy.min()] - 1500, y[iy.max()] + 1500)
    side = max(box[1] - box[0], box[3] - box[2])          # square, so the map fills its panel
    lims = []
    for c, (g0, g1) in (((box[0] + box[1]) / 2, (x[0], x[-1])), ((box[2] + box[3]) / 2, (y[0], y[-1]))):
        lo = min(max(c - side / 2, g0), g1 - side); lims += [max(lo, g0), min(lo + side, g1)]
    S["box"] = tuple(lims)
    return S


def upsample(S, H):
    """Ice thickness onto the display grid (identity when it is the model grid), clipped at zero."""
    if H.shape == S["bed"].shape:
        return np.maximum(H, 0)
    from scipy.ndimage import map_coordinates
    yy, xx = np.meshgrid((S["y"] - S["y2"][0]) / (S["y2"][1] - S["y2"][0]),
                         (S["x"] - S["x2"][0]) / (S["x2"][1] - S["x2"][0]), indexing="ij")
    return np.maximum(map_coordinates(H, [yy, xx], order=3, mode="nearest"), 0)


def _moraine_marks(ax, S, t, cm, norm, scale=1.0, keys=None, grey=False):
    """Moraine cells coloured by age; red outline while the moraine's age window contains ``t``."""
    x, y = S["x"], S["y"]; active = {}
    order = sorted(keys or S["mor"], key=lambda kk: -S["mor"][kk]["age"])        # oldest first, underneath
    for rank, kk in enumerate(order):
        mo = S["mor"][kk]; on = (mo["t0"] <= t <= mo["t1"]) and not grey; active[kk] = on
        iy, ix = np.nonzero(S["lines"][kk] > .5)
        mk = "*" if kk == "ICEM_Protin2019_E" else ("s" if mo["dashed"] else "o")
        size = scale * (34 - 3 * rank) * (2.4 if mk == "*" else (0.9 if mk == "s" else 1.0))
        ax.scatter(x[ix], y[iy], s=size, marker=mk, c=["0.75" if grey else cm(norm(mo["age"]))],
                   edgecolors="#d62728" if on else ("0.35" if grey else "k"),
                   linewidths=(1.6 if on else 0.6) * min(scale, 1.6), zorder=5 + rank * .01)
    return active


def _wood_state(S, t):
    """Per wood location: 'free' if an ice-free window is current, 'cover' if a re-advance window is, else None."""
    locs = {}
    for w in S["wood"]:
        st = locs.setdefault((w["x"], w["y"]), dict(state=None, ids=[]))
        st["ids"].append(w["id"])
        if w["t0"] <= t <= w["t1"]:
            st["state"] = "free" if w["free"] or st["state"] == "free" else "cover"
    return locs


def _hp_marks(ax, S, t, scale=1.0, labels=False):
    """Late-Holocene constraints: LIA extent, 2012/2021 Argentière outlines, Talèfre T4, the dated wood."""
    x, y = S["x"], S["y"]
    near_lia = abs(t - 1850) <= 40
    ax.contour(x, y, S["lia"], levels=[.5], colors="#08306b", linewidths=2.2 if near_lia else 1.0,
               linestyles="--", zorder=6)
    ax.contour(x, y, S["arg2012"], levels=[.5], colors="k", linewidths=2.0 if t >= 1990 else 0.8, zorder=6)
    ax.contour(x, y, S["arg2021"], levels=[.5], colors="0.45", linewidths=1.8 if t >= 2015 else 0.8,
               linestyles=":", zorder=6)
    t4 = S["t4"]; on = t4["t0"] <= t <= t4["t1"]
    iy, ix = np.nonzero(S["lines"]["ICEM_Protin2021_C"] > .5)
    ax.scatter(x[ix], y[iy], s=30 * scale, marker="o", c=T4_COL, edgecolors="#d62728" if on else "k",
               linewidths=1.6 if on else .6, zorder=7)
    for (wx, wy), st in _wood_state(S, t).items():
        fc = WOOD_FREE if st["state"] == "free" else (WOOD_COVER if st["state"] == "cover" else "white")
        ax.scatter([wx], [wy], s=(70 if st["state"] else 38) * scale, marker="^", c=fc,
                   edgecolors="#d62728" if st["state"] else "k", linewidths=1.6 if st["state"] else .6, zorder=8)
    if labels:
        for text, (lx, ly, hx, hy) in (("1850", (339600, 5094700, 339950, 5093950)),
                                       ("2012", (341000, 5092600, 341450, 5093550))):
            ax.annotate(text, (hx, hy), (lx, ly), fontsize=7.5, ha="center", va="center", zorder=9,
                        bbox=dict(fc="white", ec="none", alpha=.8, pad=1),
                        arrowprops=dict(arrowstyle="-", color="k", lw=.7, shrinkA=0, shrinkB=2))


def render(S, F, t_ce, out=None, fig=None):
    """One frame at model time ``t_ce`` (calendar year; the nearest saved snapshot is shown)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    lg = S["window"] == "lg"
    cm, norm = plt.cm.viridis_r, Normalize(*AGE_RANGE)
    ts = F["times"]; k = int(np.argmin(abs(ts - t_ce))); t = float(ts[k]); ka = (1950 - t) / 1000
    tn = F["node_times"]
    t_all = np.linspace(tn[0], min(tn[-1], ts[-1]), 500)
    T_all = np.interp(t_all, tn, F["T_nodes"])
    T_now = float(np.interp(t, tn, F["T_nodes"]))
    P0, TP = float(F.get("P0", 1.8)), float(F.get("T_present", -0.89))     # joint runs store their own
    P_now = P0 * 1.07 ** (T_now - TP)
    x, y = S["x"], S["y"]; dx = S["dx"]
    ext = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dx / 2, y[-1] + dx / 2]
    H = upsample(S, F["H"][k])
    Hm = np.where(H > 3, H, np.nan)
    X = (lambda v: (1950 - v) / 1000) if lg else (lambda v: v)          # time axis: ka BP or year CE
    now = X(t)

    fig = fig or plt.figure(figsize=(13, 10.4), dpi=100)
    fig.clf()
    L, X0, W = 0.055, 0.695, 0.29           # left edge shared by the curve and the map; right column

    # ---- top: the inverted temperature, drawn up to now
    a = fig.add_axes([L, 0.855, 0.985 - L, 0.115])
    if lg:
        a.axvspan(12.9, 11.7, color="0.93", lw=0, zorder=0)
        a.text(12.3, 0.95, "Younger Dryas", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=8,
               color="0.45")
    elif "bounds" in F:
        lo, hi = F["bounds"]
        a.fill_between(X(tn), lo, hi, step="mid", color="#1f77b4", alpha=.08, lw=0, zorder=0)
        a.step(X(tn), hi, where="mid", c="#1f77b4", lw=.7, alpha=.5, zorder=0)
        a.text(X(-1900), hi[0] + .08, "literature warm limit", fontsize=7, color="#1f77b4", alpha=.8, va="bottom")
    a.plot(X(t_all), T_all, c="0.8", lw=1.6, zorder=1)
    done = t_all <= t
    a.plot(X(t_all[done]), T_all[done], c="#1f77b4", lw=2.6, zorder=3)
    a.plot([now], [T_now], "o", c="#1f77b4", ms=8, mec="white", mew=1.4, zorder=4)
    a.axvline(now, c="#1f77b4", lw=.8, alpha=.5, zorder=2)
    tr = a.get_xaxis_transform()
    if lg:
        for kk, mo in S["mor"].items():
            a.plot([mo["age"]], [1.0], marker="v", ms=7, c=cm(norm(mo["age"])), mec="k", mew=.4, ls="none",
                   transform=tr, clip_on=False, zorder=5)
        a.text(0.004, 0.95, "▼ moraine ages (colours as on the map)", transform=a.transAxes, fontsize=7.5,
               va="top", color="0.35")
        a.set_xlim(13.6, 9.0); a.set_xlabel("ka BP", fontsize=8, labelpad=1)
    else:
        for w in S["wood"]:          # when each dated wood requires the glacier absent / back over it
            on = w["t0"] <= t <= w["t1"]
            a.plot([w["t0"], w["t1"]], [0.93 if w["free"] else 0.84] * 2, transform=tr, solid_capstyle="butt",
                   c=WOOD_FREE if w["free"] else WOOD_COVER, lw=4 if on else 2.4, alpha=1 if on else .55, zorder=5)
        for yr, col, lab in ((S["t4"]["year"], T4_COL, "T4"), (1850, "#08306b", "LIA"), (2012, "k", "2012")):
            a.plot([yr], [1.0], marker="v", ms=7, c=col, mec="k", mew=.4, ls="none", transform=tr, clip_on=False,
                   zorder=6)
        a.text(0.004, 0.75, "bars: dated wood (orange: glacier absent, blue: glacier back over it)   "
               "▼ T4 moraine, 1850 extent, 2012 outline", transform=a.transAxes, fontsize=7.5, va="top",
               color="0.35")
        a.set_xlim(-2000, 2030); a.set_xlabel("year CE", fontsize=8, labelpad=1)
    lo_y = T_all.min() - 0.5 if lg else min(T_all.min(), -3.2) - 0.3
    a.set_ylim(lo_y, T_all.max() + (0.8 if lg else 1.1))
    a.set_ylabel("T at 2000 m (°C)", fontsize=9); a.tick_params(labelsize=8)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)

    # ---- bottom left: DEM, constraints, glacier
    m = fig.add_axes([L, 0.025, X0 - 0.015 - L, 0.785]); m.set_anchor("NW")
    m.imshow(S["hs"], cmap="gray", extent=ext, alpha=0.95, vmin=0, vmax=1.15)
    m.contour(x, y, S["bed"], levels=np.arange(1000, 4800, 250), colors="k", linewidths=.25, alpha=.45)
    im = m.imshow(Hm[::-1], cmap="Blues", vmin=0, vmax=350, extent=ext, alpha=.88, zorder=3)
    if lg:
        active = _moraine_marks(m, S, t, cm, norm)
        clusters, zoom = CLUSTERS, ZOOM
    else:
        _moraine_marks(m, S, t, cm, norm, keys=LOGNAN, grey=True)
        _hp_marks(m, S, t)
        clusters, zoom = CLUSTERS_HP, ZOOM_HP
    for name, (lx, ly, hx, hy) in clusters.items():
        m.annotate(name, (hx, hy), (lx, ly), fontsize=9, fontweight="bold", ha="center", va="center", zorder=9,
                   color="k", bbox=dict(fc="white", ec="none", alpha=.8, pad=1.5),
                   arrowprops=dict(arrowstyle="-", color="k", lw=.8, shrinkA=0, shrinkB=3))
    zx0, zx1, zy0, zy1 = zoom
    m.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec="k", lw=1.3, ls="--", zorder=8))
    m.text(zx1 - 100, zy1 - 100, "zoom", fontsize=7.5, ha="right", va="top", zorder=8,
           bbox=dict(fc="white", ec="none", alpha=.7, pad=1))
    m.set_xlim(S["box"][:2]); m.set_ylim(S["box"][2:]); m.set_aspect("equal")
    m.set_xticks([]); m.set_yticks([])
    m.text(.015, .015, "DEM: Henz et al. 2025 ice-free bed + 2012 radar bed; 100 m model grid", transform=m.transAxes,
           fontsize=7, color="0.2", bbox=dict(fc="white", ec="none", alpha=.7, pad=1.5))

    # ---- bottom right, top: time, climate, key
    r = fig.add_axes([X0, 0.43, W, 0.38]); r.axis("off"); r.set_xlim(0, 1); r.set_ylim(0, 1)
    mn = lambda v: f"{v:+.1f}".replace("-", "−")
    if lg:
        r.text(0, .995, f"{ka:.2f} ka BP", fontsize=20, fontweight="bold", va="top")
        r.text(0, .905, f"{abs(t):.0f} BCE" if t < 0 else f"{t:.0f} CE", fontsize=9, color="0.4", va="top")
    else:
        r.text(0, .995, f"{abs(t):.0f} BCE" if t < 0 else f"{t:.0f} CE", fontsize=20, fontweight="bold", va="top")
        r.text(0, .905, f"{ka:.2f} ka BP", fontsize=9, color="0.4", va="top")
    r.text(0, .845, f"T(2000 m) = {mn(T_now)} °C", fontsize=11, va="top", color="#1f77b4")
    r.text(0, .785, f"{mn(T_now - TP)} °C vs 2012–2021", fontsize=8.5, va="top", color="0.4")
    r.text(0, .725, f"Precipitation {P_now:.2f} m/yr", fontsize=11.5, fontweight="bold", va="top")
    if lg:
        r.text(0, .635, "Moraines (age, ka BP)", fontsize=10, fontweight="bold", va="top")
        rows = [(kk, mo) for kk, mo in sorted(S["mor"].items(), key=lambda kv: -kv[1]["age"]) if not mo["dashed"]]
        rows.append(("morpho", dict(name="J1 + undated, La Joux / Lognan", age=12.05, err=0, dashed=True)))
        y0 = .56
        for kk, mo in rows:
            col = cm(norm(mo["age"]))
            on = any(active[k2] for k2 in S["mor"] if S["mor"][k2]["dashed"]) if kk == "morpho" else active[kk]
            mk = "*" if kk == "ICEM_Protin2019_E" else ("s" if mo["dashed"] else "o")
            r.plot([.035], [y0], marker=mk, ms=11 if mk == "*" else 7.5, c=col, mec="#d62728" if on else "k",
                   mew=2.0 if on else .8, ls="none")
            age = f"{mo['age']:.1f}" + (f" ± {mo['err']:.1f}" if mo["err"] else " assigned")
            r.text(.09, y0, mo["name"], fontsize=8.5, va="center", fontweight="bold" if on else "normal")
            r.text(1.0, y0, age, fontsize=8, va="center", ha="right", color="0.35")
            y0 -= .068
        r.text(0, y0 + .025, "red outline: the moraine's age window is now\n"
               "square: age assigned from morphostratigraphy", fontsize=7, color="0.4", va="top")
    else:
        r.text(0, .635, "Constraints", fontsize=10, fontweight="bold", va="top")
        t4 = S["t4"]
        keyrows = [
            (Line2D([], [], c="#08306b", ls="--", lw=1.6), "1850 extent (LIA, Reinthaler & Paul 2025)", abs(t - 1850) <= 40),
            (Line2D([], [], c="k", lw=1.6), "Argentière outline 2012", t >= 1990),
            (Line2D([], [], c="0.45", ls=":", lw=1.8), "Argentière outline 2021", t >= 2015),
            (Line2D([], [], marker="o", c=T4_COL, ls="none", mec="k", ms=7.5),
             f"T4 Talèfre lateral moraine, {t4['year']:.0f} ± {(t4['t1'] - t4['t0']) / 2:.0f}",
             t4["t0"] <= t <= t4["t1"]),
            (Line2D([], [], marker="^", c=WOOD_FREE, ls="none", mec="k", ms=8.5), "wood: glacier must be absent", None),
            (Line2D([], [], marker="^", c=WOOD_COVER, ls="none", mec="k", ms=8.5), "wood: glacier back over it", None),
            (Line2D([], [], marker="^", c="white", ls="none", mec="k", ms=7.5), "wood site, no requirement now", None),
        ]
        y0 = .56
        for h, lab, on in keyrows:
            xs = [.0, .07] if h.get_marker() in (None, "None", "") else [.035]
            r.plot(xs, [y0] * len(xs), **{p: getattr(h, f"get_{p}")() for p in ("color", "ls", "lw", "marker",
                                                                              "mec", "ms")})
            r.text(.09, y0, lab, fontsize=8.5, va="center", fontweight="bold" if on else "normal")
            y0 -= .062
        locs = _wood_state(S, t)
        cur = [w for w in S["wood"] if w["t0"] <= t <= w["t1"]]
        msg = ("now: " + ", ".join(f"{w['id']} ({'absent' if w['free'] else 'back over'})" for w in cur[:4])
               + (" …" if len(cur) > 4 else "")) if cur else "now: no wood date in its window"
        r.text(0, y0 + .02, msg, fontsize=7.5, color="0.25", va="top", wrap=True)

    # ---- bottom right, below the key: zoom
    z = fig.add_axes([X0, 0.06, W, 0.3625]); z.set_anchor("N")
    z.imshow(S["hs"], cmap="gray", extent=ext, alpha=0.95, vmin=0, vmax=1.15)
    cs = z.contour(x, y, S["bed"], levels=np.arange(1000, 3400, 50), colors="k", linewidths=.3, alpha=.4)
    z.clabel(cs, levels=[lv for lv in cs.levels if lv % 200 == 0], fontsize=6, fmt="%d", inline_spacing=2)
    z.imshow(Hm[::-1], cmap="Blues", vmin=0, vmax=350, extent=ext, alpha=.85, zorder=3)
    surf = np.where(H > 3, S["bed"] + H, np.nan)
    if np.isfinite(surf).any():
        z.contour(x, y, surf, levels=np.arange(1000, 3400, 50), colors="white", linewidths=.6, alpha=.9, zorder=4)
    if lg:
        _moraine_marks(z, S, t, cm, norm, scale=2.2)
        for kk, (lx, ly, lab) in ZOOM_LABELS.items():
            iy, ix = np.nonzero(S["lines"][kk] > .5)
            j = int(np.argmin((x[ix] - lx) ** 2 + (y[iy] - ly) ** 2))
            z.annotate(lab, (x[ix[j]], y[iy[j]]), (lx, ly), fontsize=7.5, ha="center", va="center", zorder=9,
                       fontweight="bold" if active[kk] else "normal",
                       bbox=dict(fc="white", ec="#d62728" if active[kk] else "none", alpha=.85, pad=1.5),
                       arrowprops=dict(arrowstyle="-", color="k", lw=.7, shrinkA=0, shrinkB=2))
        ztitle = "Zoom: La Joux (white lines: ice surface every 50 m)"
    else:
        _moraine_marks(z, S, t, cm, norm, scale=2.0, keys=LOGNAN, grey=True)
        _hp_marks(z, S, t, scale=1.6, labels=True)
        z.annotate("Lognan (Lateglacial moraines)", (342400, 5092600), (342900, 5091200), fontsize=7.5,
                   ha="center", va="center", zorder=9, bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5),
                   arrowprops=dict(arrowstyle="-", color="k", lw=.7, shrinkA=0, shrinkB=2))
        ztitle = "Zoom: Argentière tongue, Lognan (white: ice surface, 50 m)"
    z.plot([zx1 - 1300, zx1 - 300], [zy0 + 300, zy0 + 300], c="k", lw=2.5, zorder=9)
    z.text(zx1 - 800, zy0 + 380, "1 km", fontsize=7, ha="center", va="bottom", zorder=9,
           bbox=dict(fc="white", ec="none", alpha=.7, pad=1))
    z.set_xlim(zx0, zx1); z.set_ylim(zy0, zy1); z.set_aspect("equal"); z.set_xticks([]); z.set_yticks([])
    for sp in z.spines.values(): sp.set_linestyle("--"); sp.set_linewidth(1.3)
    z.set_title(ztitle, fontsize=8.5, pad=3)

    cax = fig.add_axes([X0, 0.022, 0.17, 0.011])
    fig.colorbar(im, cax=cax, orientation="horizontal"); cax.tick_params(labelsize=7)
    fig.text(X0 + 0.178, 0.0275, "modelled ice\nthickness (m)", fontsize=7.5, va="center", color="0.3")
    if out:
        fig.savefig(out, dpi=100)
    return fig


def load_fields(run):
    d = np.load(f"{run}/video_fields.npz")
    F = {k: d[k] for k in d.files}
    hist = json.load(open(f"{run}/history.json"))
    if hist.get("bounds"):
        F["bounds"] = [np.array(b) for b in hist["bounds"]]
    return F


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fields", "fields_joint", "frame", "frames", "video"])
    ap.add_argument("--window", choices=["lg", "hp"], default="lg")
    ap.add_argument("--run", default=None, help="default results/argentiere_<window>_<dx>m_phys")
    ap.add_argument("--dx", type=int, default=DX)
    ap.add_argument("--every", type=float, default=25.0)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--t_yr", type=float, default=100.0, help="years after the window start")
    ap.add_argument("--out", default=None)
    opts = ap.parse_args()
    opts.run = opts.run or f"results/argentiere_{opts.window}_{opts.dx}m_phys"
    out_dir = f"{opts.run}/video"
    name = {"lg": "lateglacial", "hp": "late_holocene"}[opts.window]
    if opts.what == "fields":
        fields(opts.run, opts.every, opts.dx, opts.window)
    elif opts.what == "fields_joint":
        fields_joint(opts.run, opts.every)
    elif opts.what == "video":
        import subprocess
        import imageio_ffmpeg
        mp4 = opts.out or f"{out_dir}/argentiere_{name}_{opts.dx}m.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(opts.fps),
                        "-i", f"{out_dir}/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        "-movflags", "+faststart", mp4], check=True)
        print("wrote", mp4)
    else:
        os.makedirs(out_dir, exist_ok=True)
        F = load_fields(opts.run); S = load_static(opts.dx, opts.window)
        if opts.what == "frame":
            t_ce = float(F["node_times"][0]) + opts.t_yr
            path = opts.out or f"{out_dir}/frame_{int(opts.t_yr):05d}.png"
            render(S, F, t_ce, out=path); print("wrote", path)
        else:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(13, 10.4), dpi=100)
            for i, t in enumerate(F["times"]):
                render(S, F, float(t), out=f"{out_dir}/frame_{i:04d}.png", fig=fig)
            print(f"wrote {len(F['times'])} frames to {out_dir}/")

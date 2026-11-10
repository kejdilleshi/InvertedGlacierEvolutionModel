"""Write notebooks/Argentiere_temperature_inversion.ipynb (run once; edit the notebook after)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "IGEM", "language": "python", "name": "python3"}
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Argentière — temperature history from dated landforms, 13.6 ka to today

**Goal.** Test the moraine-constrained temperature inversion on a second site,
over as much of the last 14 kyr as the record allows: Argentière glacier and
its neighbours in the upper Arve (Tour, Mer de Glace with Talèfre), Mont Blanc
massif. The control is the temperature at 2000 m, `T(t)`; precipitation and
the melt factor are calibrated on Argentière's 2012–2021 stake mass balance
and held fixed.

**Data** (built by `tools/prepare_argentiere_data.py`, described in
[`data/Argentiere/README.md`](../data/Argentiere/README.md)):

* **AlpIce-1** (Kamleitner et al. 2026): ¹⁰Be-dated lateral and frontal moraines
  (Lognan, La Joux, Talèfre), bedrock and erratic exposure ages (Lognan, Mer de
  Glace transects, Aiguilles Rouges), and ¹⁴C-dated subfossil wood in the Mer de
  Glace lateral moraine (Le Roy et al. 2015).
* **LIA** extent and ice surface (Reinthaler & Paul 2025).
* **Present**: Argentière 2012 and 2021 thickness, and the 2012–2021 stake SMB.

**Two windows**, each its own inversion — they share the data pipeline and
the model, not the run:

| window | span | what constrains it |
|---|---|---|
| `lg` | 13.6 → 9.0 ka BP | Lateglacial / early-Holocene moraines and exposure ages |
| `hp` | 2000 BCE → 2021 CE | Mer de Glace wood, Talèfre LIA moraine, LIA geometry, 2012/2021 thickness |

Between 9 and 4 ka the glaciers were mostly smaller than today and left no
moraines; the only datum there is one bedrock age (Mer de Glace, 6.9 ka).
""")

code(r"""
import json, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, netCDF4, torch
from IPython.display import Image, display

os.chdir("..")                      # repo root
import tools.run_argentiere_inversion as R   # pulls in core.glacier, whose plotting module forces Agg ...
%matplotlib inline
RUN_HEAVY_CELLS = False             # True re-runs the GPU inversions here (hours); False loads results/
DX = 200
RESULTS = {w: f"results/argentiere_{w}_{DX}m" for w in ("lg", "hp")}
print("repo root:", os.getcwd())
""")

md(r"""
## 1. The data on the model grid

One 200 m grid covers the whole upper Arve (145 × 140 cells, 28 × 29 km). Green
shows where ice may grow in the Lateglacial runs: the upper Arve basin above the
Mer de Glace confluence, plus 8 km down the Arve. Cyan is the tighter Holocene
mask. The symbols are the constraints that fall inside the basin.

The domain is cut automatically from a 58 × 58 km source crop: the ice masks'
bounding box plus 3 km. No ice-allowed cell comes closer than 2.6 km to IGM's
boundary ramp, so the boundary clips neither the Géant basin nor the Arve
tongue. The hillshade is lit from the south.
""")

code(r"""
from tools.plot_argentiere_data import overview
fig, ax = plt.subplots(figsize=(7.5, 9))
overview(f"data/Argentiere/argentiere_{DX}m.nc", f"data/Argentiere/argentiere_observations_{DX}m.csv", ax=ax)
plt.show()
obs_table = pd.read_csv(f"data/Argentiere/argentiere_observations_{DX}m.csv")
obs_table = obs_table[obs_table.in_catchment]
print(obs_table.groupby(["kind", "group"]).size())
""")

md(r"""
Two corrections were needed before any of this lined up (details in the data
README):

* the Argentière radar/DEM products are **52 m** above AW3D30 and the AlpIce-1
  sample elevations on ice-free ground — an ellipsoidal vs orthometric datum —
  so a fitted plane is removed from them;
* the lateral-moraine operator steps **downslope** from the crest instead of
  toward the nearest modern ice, because the Lateglacial ice margins at La Joux
  are kilometres from any ice today. Its offset, bias and σ are calibrated on the
  one dated LIA lateral moraine (Talèfre T4) against the LIA surface.

The moraines themselves pass a positional check: the DEM at each dated boulder
agrees with the boulder's recorded elevation to within ±25 m.
""")

code(r"""
cols = ["id", "kind", "group", "year", "t0", "t1", "z", "n_cells", "dem_minus_boulder_m", "source"]
dated = obs_table[obs_table.kind.isin(["lateral", "frontal"])][cols]
dated.assign(age_ka=(1950 - dated.year) / 1000).round(1)
""")

md(r"""
## 2. Calibrating the mass balance on the 2012–2021 stakes

The PDD model has three free climate numbers — temperature, precipitation and
melt factor — and the 25 Argentière stakes (2365–2977 m, mostly ablation zone)
only constrain two of them: the ELA and the balance gradient. Temperature and
precipitation trade off almost perfectly along that valley. So the temperature
is **pinned from outside**: 1.0 °C at 2000 m for 2012–2021 (Chamonix, 1042 m,
≈7.5 °C, extrapolated with the model's own 7 °C/km). The stakes then give the
melt factor and the precipitation.

Stake altitudes are orthometric (they sit 52 m below the raw 2021 DEM), so no
datum shift is needed here.
""")

code(r"""
from core.smb import update_smb_PDD
st = pd.read_csv("data/Argentiere/sources/present/SMB_argentiere_2012-2021_utm32N.csv")
z = torch.tensor(st.Alt.values, dtype=torch.float64); b = st.iloc[:, 3].values
model = lambda T, P, mf: update_smb_PDD(z, P, torch.tensor(T, dtype=torch.float64), 10.0,
                                        melt_factor=mf, z_ref=2000.0).numpy() * 0.91   # -> m w.e.
T_PIN = 1.0
grid = [(np.sqrt(np.mean((model(T_PIN, P, mf) - b) ** 2)), mf, P)
        for mf in np.arange(0.15, 0.40, 0.01) for P in np.arange(1.0, 8.01, 0.1)]
rmse, MF, PREC = min(grid)
print(f"T(2000 m) pinned at {T_PIN} C -> melt factor {MF:.2f} m w.e./(C month), precipitation {PREC:.1f} m/yr, rmse {rmse:.2f} m w.e.")

zz = np.linspace(2300, 3100, 50)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(b, st.Alt, c="k", s=12, label="stakes 2012-2021")
for T in (0.5, 1.0, 1.5):
    mf, P = min(((np.sqrt(np.mean((model(T, P, mf) - b) ** 2)), mf, P) for mf in np.arange(0.15, 0.40, 0.01)
                 for P in np.arange(1.0, 8.01, 0.1)))[1:]
    s = update_smb_PDD(torch.tensor(zz), P, torch.tensor(T, dtype=torch.float64), 10.0, melt_factor=mf, z_ref=2000.0).numpy() * 0.91
    ax[0].plot(s, zz, label=f"T={T}: mf {mf:.2f}, P {P:.1f}")
ax[0].axvline(0, c="grey", lw=.5); ax[0].set_xlabel("SMB (m w.e./yr)"); ax[0].set_ylabel("elevation (m)"); ax[0].legend(fontsize=8)
ax[0].set_title("equally good fits: the stakes cannot separate T from P")
Ts = np.arange(-1, 3.01, 0.25)
best = [min(((np.sqrt(np.mean((model(T, P, mf) - b) ** 2)), P) for mf in np.arange(0.15, 0.40, 0.02)
             for P in np.arange(1.0, 9.01, 0.2))) for T in Ts]
ax[1].plot(Ts, [p for _, p in best], "o-"); ax[1].set_xlabel("pinned T(2000 m), C"); ax[1].set_ylabel("best precipitation (m/yr)")
ax2 = ax[1].twinx(); ax2.plot(Ts, [r for r, _ in best], "s--", c="C1"); ax2.set_ylabel("rmse (m w.e.)", color="C1")
ax[1].set_title("the T-P trade-off"); plt.tight_layout(); plt.show()
""")

md(r"""
**An independent check on the pin: the geodetic balance.** The stakes say
nothing about the accumulation area, but the 2012→2021 thickness change does:
Argentière lost 0.104 km³ of ice over 16 km², −0.61 m w.e./yr. For each pinned
temperature, the best stake fit implies a glacier-wide balance on the
2012–2021 surface — which pins are compatible with the geodetic number?
""")

code(r"""
nc = netCDF4.Dataset(f"data/Argentiere/argentiere_100m.nc")
bed100 = np.asarray(nc["topg"][:]); t12 = np.asarray(nc["thk_2012"][:]); t21 = np.asarray(nc["thk_2021"][:])
m12 = np.asarray(nc["argentiere_2012"][:]) > 0.5; nc.close()
geod = (t21 - t12)[m12].sum() / m12.sum() / 9.0 * 0.85          # 850 kg/m3 volume-to-mass
zs = torch.tensor((bed100 + 0.5 * (t12 + t21))[m12], dtype=torch.float64)
gw = []
for T, (r, P) in zip(Ts, best):
    mf = min(((np.sqrt(np.mean((model(T, P, mf) - b) ** 2)), mf) for mf in np.arange(0.15, 0.40, 0.01)))[1]
    gw.append(float((update_smb_PDD(zs, P, torch.tensor(T, dtype=torch.float64), 10.0, melt_factor=mf, z_ref=2000.0).numpy() * 0.91).mean()))
plt.figure(figsize=(5, 3.2)); plt.plot(Ts, gw, "o-", label="model, best stake fit")
plt.axhline(geod, c="k", ls="--", label=f"geodetic 2012-2021: {geod:.2f}")
plt.xlabel("pinned T(2000 m), C"); plt.ylabel("glacier-wide SMB (m w.e./yr)"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md(r"""
Pins below about 0.5 °C give a glacier-wide balance that is too negative; from
1 °C up the model sits within ~0.05 m w.e./yr of the geodetic value. The
station-based pin of 1.0 °C is therefore consistent with the volume change as
well as with the stakes.

The fit is flat along the trade-off line — rmse barely changes from −1 to
+3 °C — which is why the temperature has to be pinned. **This choice sets the
absolute level of every temperature below**; anomalies relative to 2012–2021
are much less sensitive to it than absolute values. The runs use
`melt_factor = 0.26`, `precip = 4.7 m/yr` (`BASE` in
`tools/run_argentiere_inversion.py`); 4.7 m/yr is an effective number for the
whole basin, including avalanche and wind input to the accumulation area.

**Precipitation follows temperature (Clausius–Clapeyron).** Only temperature
is inverted. Precipitation is tied to it at 7 % per °C,
`P(t) = 4.7 · 1.07^(T(t) − 1.0)` m/yr, so 4.7 m/yr is the value at the
2012–2021 temperature. A Younger Dryas 3 °C colder than today then gets 18 %
less precipitation, not the same amount.
""")

code(r"""
Tg = np.linspace(-3, 3, 61)
plt.figure(figsize=(4.5, 3)); plt.plot(Tg, R.BASE["precip"] * (1 + R.BASE["precip_cc"]) ** (Tg - R.BASE["T_present"]))
plt.axvline(R.BASE["T_present"], c="grey", lw=.6); plt.xlabel("T at 2000 m (C)"); plt.ylabel("precipitation (m/yr)")
plt.title("Clausius-Clapeyron, 7 %/C"); plt.tight_layout(); plt.show()
""")

md(r"""
## 3. The model and what changed in the code

Same IGM-emulator forward model as the Aletsch notebooks. Running it for
thousands of years needed four changes, all opt-in so the Aletsch experiments
are untouched:

| change | where | why |
|---|---|---|
| fixed temperature datum `z_ref` | `core/climate.py`, `core/smb.py`, `--z_ref` | the old datum, `min(ice surface)`, moves whenever the lowest cell gains or loses ice — at La Joux that is the whole Lateglacial |
| `smb_freq` + linear control interpolation | `core/glacier.py`, `--smb_freq` | with 100–200 yr control nodes the SMB used to be refreshed only once per node; now it follows the surface every 10 yr |
| two-level checkpointing `chunk_steps` | `core/glacier.py`, `--chunk_steps` | memory independent of run length (needed at 100 m; tested to give identical gradients) |
| moraine / exposure / wood operators | `core/moraine_obs.py` | each landform scored at the best time inside its age uncertainty (soft minimum over snapshots), bedrock ages as one-sided "ice-free after" constraints |
| Clausius–Clapeyron precipitation `precip_cc` | `core/glacier.py`, `--precip_cc` | precipitation scales with the control temperature, so only temperature is inverted |
| compiled transport | `core/forward_schemes/igm_emulator_step.py`, `use_compiled_transport` | `torch.compile` of the flux limiter and the emulator network: 2.8× faster forward + backward, gradients equal to 1e-4 |

Numerics: 200 m grid, CFL 0.3, depth-averaged speed capped at 300 m/yr (only
icefall cells reach it), 5 steps per year, ~18 ms per step forward + backward
once compiled, so one iteration of either window takes ~8–10 min (it was 22–33
min before compiling). Gradients go through the
emulator — with frozen velocities (`--no_emulator_grad`) the adjoint blew up
to 1e8 on this geometry, so that shortcut is not usable here.
""")

code(r"""
print("BASE:", R.BASE)
for w, cfg in R.WINDOWS.items():
    print(w, {k: v for k, v in cfg.items()})
""")

md(r"""
## 4. First look: constant-temperature scans

Before any gradient, run each window at a handful of constant temperatures and
ask, for every dated landform, which temperature fits it best. That is a cheap,
model-based reconstruction in its own right, and it gives the inversion its
starting point.

(`python tools/run_argentiere_inversion.py --window lg --scan 0 -1 -2 -3 -4`,
plus `--scan 0.5 1.0 1.5 2.0 --out_dir results/argentiere_lg_200m_scan_warm`
once it was clear the Lateglacial optimum is not below 0 °C; same for `hp`
with 0–2 °C. Forward only, ~2.5 min per temperature.)
""")

code(r"""
def load_scan(w):
    # scan.json, plus any extra temperatures run into <results>_scan_*/ (e.g. the warmer lg scan)
    import glob
    with open(f"{RESULTS[w]}/scan.json") as f:
        sc = json.load(f)
    for extra in sorted(glob.glob(f"{RESULTS[w]}_scan_*/scan.json")):
        with open(extra) as f:
            sc["rows"] += json.load(f)["rows"]
    sc["rows"].sort(key=lambda r: r["T"])
    return sc

def best_T(scan, oid):
    T = np.array([r["T"] for r in scan["rows"]]); m = np.array([r["terms"][oid] for r in scan["rows"]])
    i = int(np.argmin(m))
    if 0 < i < len(T) - 1:                       # parabola through the three best points
        a, b_, _ = np.polyfit(T[i - 1:i + 2], m[i - 1:i + 2], 2)
        if a > 0:
            return float(np.clip(-b_ / (2 * a), T[i - 1], T[i + 1])), float(m.min())
    return float(T[i]), float(m.min())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, w in zip(axes, ("lg", "hp")):
    sc = load_scan(w)
    T = [r["T"] for r in sc["rows"]]
    for oid, meta in sc["meta"].items():
        m = [r["terms"][oid] for r in sc["rows"]]
        style = {"lateral": "-", "frontal": "-", "thickness": "--", "extent": "--"}.get(meta["kind"], ":")
        ax.plot(T, m, style, lw=1.2 if style == "-" else 0.7, label=oid if style != ":" else None)
    ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("constant T at 2000 m (C)"); ax.set_ylabel("normalized misfit")
    ax.set_title(f"{w}: misfit of each observation vs temperature"); ax.legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.show()
""")

code(r"""
rows = []
for w in ("lg", "hp"):
    sc = load_scan(w)
    for oid, meta in sc["meta"].items():
        if meta["kind"] in ("lateral", "frontal", "thickness", "extent"):
            Tb, mb = best_T(sc, oid)
            Ts_ = [r["T"] for r in sc["rows"]]
            rows.append(dict(window=w, id=oid, kind=meta["kind"], group=meta["group"],
                             age_ka=(1950 - meta["year"]) / 1000, best_T=Tb, misfit_at_best=mb,
                             at_edge=Tb in (min(Ts_), max(Ts_))))
best_tab = pd.DataFrame(rows).sort_values("age_ka", ascending=False)
best_tab.round(2)
""")

md(r"""
## 5. Gradient inversion

Control: `T(t)` at 2000 m, nodes every 200 yr (`lg`, 24 nodes) or 100 yr
(`hp`, 42 nodes), linear in between; smoothness prior on the second
difference; for `hp`, an anchor `T(2017) = 1.0 ± 0.3 °C` from the stake
calibration.

**Optimizer.** Adam, 40 iterations. The learning rate decays on a cosine
from 0.15 to 0.01 °C. If a step makes the loss more than 5 % worse than the
best so far, it goes back to the best iterate and halves the rate.

The cell below is the same loop as `tools/run_argentiere_inversion.py`; it is
guarded because one window takes hours. The runs whose results are loaded
further down were started with

```
IGEM_DEVICE=cuda:0 python tools/run_argentiere_inversion.py --window lg --iters 40 --init_T 1.0
IGEM_DEVICE=cuda:1 python tools/run_argentiere_inversion.py --window hp --iters 40 --init_T 1.2
```
""")

code(r"""
if RUN_HEAVY_CELLS:
    from core.moraine_obs import invert_observations
    from core.utils import device
    torch.backends.cudnn.deterministic = True
    w = "lg"
    glacier, geo, obs, node_times, anchor = R.build(w, DX)
    T = torch.full((len(node_times), 1), R.WINDOWS[w]["init_T"], device=device, requires_grad=True)
    opt = torch.optim.Adam([T], lr=0.1)
    for i in range(20):
        opt.zero_grad()
        loss, data, terms, _ = invert_observations(
            T, glacier, obs, precip_tensor=R.BASE["precip"], T_s=R.BASE["T_s"],
            melt_factor=R.BASE["melt_factor"], dt_snap=R.WINDOWS[w]["dt_snap"],
            reg_lambda=1.0, anchor=anchor)
        loss.backward(); opt.step()
        print(i, float(loss), float(data))
else:
    print("RUN_HEAVY_CELLS is False: loading", RESULTS)
""")

code(r"""
def load_run(w):
    p = f"{RESULTS[w]}/history.json"
    if not os.path.exists(p):
        print(f"{w}: no history yet at {p}"); return None
    with open(p) as f:
        return json.load(f)

runs = {w: load_run(w) for w in ("lg", "hp")}
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, (w, run) in zip(axes, runs.items()):
    if run is None: continue
    h = run["history"]
    ax.plot(h["loss"], "o-", ms=3, label="loss"); ax.plot(h["data"], label="data term")
    ax.set_yscale("log"); ax.set_xlabel("iteration")
    d0, db = h["data"][0], min(h["data"])
    ax.set_title(f"{w}: best at iter {run['best']['iter']}, data term -{(1 - db / d0) * 100:.0f}%")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw=dict(width_ratios=[1, 1]))
for ax, w in zip(axes, ("lg", "hp")):
    run = runs[w]
    if run is None: continue
    t = np.array(run["node_times"])
    x = (1950 - t) / 1000 if w == "lg" else t
    h = run["history"]
    ax.plot(x, h["T"][0], c="grey", lw=1, ls="--", label="initial")
    ax.plot(x, run["best"]["T"], c="C3", lw=2, label=f"best (iter {run['best']['iter']})")
    bt = best_tab[(best_tab.window == w) & best_tab.kind.isin(["lateral", "frontal"])]
    if len(bt):
        bx = bt.age_ka if w == "lg" else 1950 - 1000 * bt.age_ka
        ax.scatter(bx, bt.best_T, c=np.where(bt.group == "dated", "k", "w"), edgecolor="k", zorder=5,
                   label="best constant T per moraine (scan)")
    ax.axhline(R.BASE["T_present"], c="C0", lw=.8, label="2012-2021 (stakes)")
    if w == "lg":
        ax.axvspan(12.9, 11.7, color="C0", alpha=.08, label="Younger Dryas")
        ax.invert_xaxis(); ax.set_xlabel("ka BP")
    else:
        ax.axvspan(1300, 1850, color="C0", alpha=.08, label="LIA"); ax.set_xlabel("year CE")
    ax.set_ylabel("T at 2000 m (C)"); ax.legend(fontsize=7); ax.set_title(w)
plt.tight_layout(); plt.show()
""")

md(r"""
### 5.1 How well is each observation fitted?

Misfits are normalized: about 1 means "fitted to within its σ" for the
surface constraints; 0 means a one-sided constraint is satisfied.
""")

code(r"""
tabs = []
for w in ("lg", "hp"):
    run = runs[w]
    if run is None: continue
    h = run["history"]; b = run["best"]["iter"]
    for oid, meta in run["meta"].items():
        tabs.append(dict(window=w, id=oid, kind=meta["kind"], group=meta["group"], weight=meta["weight"],
                         age_ka=round((1950 - meta["year"]) / 1000, 2),
                         misfit_init=h["terms"][0][oid], misfit_best=h["terms"][b][oid]))
fit = pd.DataFrame(tabs)
fit.sort_values(["window", "misfit_best"], ascending=[True, False]).round(2).head(40) if len(fit) else "no inversion results yet"
""")

md(r"""
### 5.2 Simulated glaciers at the dated stands

Thickness at the saved snapshot closest to each dated moraine, with the
moraine lines on top.
""")

code(r"""
nc = netCDF4.Dataset(f"data/Argentiere/argentiere_{DX}m.nc")
xg, yg = nc["x"][:], nc["y"][:]
bed = np.asarray(nc["topg"][:])
lines = {v[5:]: np.asarray(nc[v][:]) for v in nc.variables if v.startswith("line_")}
nc.close()
ext = [xg[0], xg[-1], yg[0], yg[-1]]

def show_stand(w, years, titles):
    p = f"{RESULTS[w]}/final_fields.npz"
    if not os.path.exists(p):
        print("no final fields for", w); return
    f = np.load(p); times = f["times"]
    fig, axes = plt.subplots(1, len(years), figsize=(4.2 * len(years), 5.2))
    for ax, yr, ttl in zip(np.atleast_1d(axes), years, titles):
        k = int(np.argmin(np.abs(times - yr)))
        H = np.where(f["H"][k] > 1, f["H"][k], np.nan)
        ax.contour(xg, yg, bed, levels=np.arange(1000, 4800, 250), colors="k", linewidths=.2)
        im = ax.imshow(H[::-1], extent=ext, cmap="Blues", vmin=0, vmax=500)
        for key, L in lines.items():
            ax.contour(xg, yg, L, levels=[.5], colors="r", linewidths=.8)
        ax.set_title(f"{ttl}\nsnapshot {times[k]:.0f} CE"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=.6, label="ice thickness (m)"); plt.show()

show_stand("lg", [1950 - 12500, 1950 - 11660, 1950 - 10900, 1950 - 10350],
           ["12.5 ka (YD)", "L1 11.66 ka", "J2 La Joux 10.9 ka", "L2/L3 10.35 ka"])
show_stand("hp", [1150, 1850, 2012.7, 2021.7], ["1150 CE (Medieval)", "1850 LIA", "2012", "2021"])
""")

md(r"""
## 6. What the runs show

**Both runs converged.** By iteration 30 the gradient and the loss had
flattened (loss changes < 0.1 % per iteration). The step-halving fired twice
in each run, both times early. The data
term fell 61 % in `hp` (1.24 → 0.48) but only 20 % in `lg` (1.16 → 0.92).
More iterations would not change either number. What is left is the model
disagreeing with itself, not the optimizer stopping early.

**`hp` (2000 BCE → 2021)**

* The Mer de Glace wood is fitted. Every ice-free interval is satisfied, and
  the Medieval "buried by the next advance" constraints are at ≤ 0.4.
* The shape is plausible: a warm Roman period, a cold dip about 600 CE, warm
  about 1000 CE, a cold 1300–1500 CE, and a cold LIA end.
* The level is not. The 2012/2021 thickness misfits *rise* (3.0 → 3.2,
  3.5 → 3.7), the LIA-surface misfit rises too (0.7 → 1.2), and 1850 sits
  0.6 °C *warmer* than 2017. The simulated 2012 glacier is still bigger than the
  1850 one.

**`lg` (13.6 → 9.0 ka)**

* The Lognan laterals L1 (0.4 → 0.3) and L2 (0.9 → 0.06) and the Mer de Glace
  bedrock ages are now fitted.
* La Joux (J1, the undated La Joux lines, J2 frontal) and the Talèfre laterals
  (9.0, 3.7) stay misfitted.
* The curve sits at 1.2–1.6 °C throughout, warmer than 2012–2021, with no
  Younger Dryas cooling.

**Diagnosis.** For a
given temperature the model's glaciers are too large, so the inversion warms
everything:

* Its equilibrium at the 2012–2021 temperature is about the LIA glacier.
* Its Lateglacial moraines at Lognan are reached at present-day temperature,
  while La Joux (3 km beyond the LIA) and Talèfre cannot be fitted at the same
  time.

This is not something the optimizer can fix. The likely culprit is
accumulation: precipitation calibrated only on ablation-zone stakes, applied
uniformly up to 4800 m, plus ice on steep headwalls.

**Next step:** calibrate precipitation on the observed 1850 → 2012 retreat.
Run 1850–2021 with a station temperature anomaly and choose P (or an
accumulation cap) so that the 2012 geometry is reached. Then re-run both
windows. Until then, read only the *shape* of the `hp` curve, not its level,
and treat `lg` as a pipeline test.
""")

nb.cells = C
nbf.write(nb, "notebooks/Argentiere_temperature_inversion.ipynb")
print("wrote notebooks/Argentiere_temperature_inversion.ipynb")

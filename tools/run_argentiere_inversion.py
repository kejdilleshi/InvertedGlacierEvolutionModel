"""Invert the temperature history of the upper Arve glaciers from dated landforms.

Two windows, each its own inversion (a single differentiable run from 13.6 ka to
today is ~3x longer than either and has no data over most of the middle):

``lg``  Lateglacial to early Holocene, 13.6 -> 9.0 ka BP. Lateral and frontal
        moraines of Argentière (Lognan, La Joux) and Talèfre, bedrock exposures
        and erratics (ice-free after their exposure age).
``hp``  Late Holocene to present, 2000 BCE -> 2021 CE. Subfossil wood in the
        Mer de Glace lateral moraine (ice-free intervals and the advances that
        buried them), the dated Talèfre LIA moraine, the LIA extent and surface
        (1850), and Argentière's 2012 and 2021 thickness. The 2012-2021
        temperature is anchored to the stake-SMB calibration.

    IGEM_DEVICE=cuda:0 python tools/run_argentiere_inversion.py --window lg --iters 30
    python tools/run_argentiere_inversion.py --window lg --scan 0 -1 -2 -3 -4

``--scan`` runs the forward model at constant temperatures instead (no
gradient), which is how the initial guess is chosen.

``--valley`` scores every landform against its source valley's flowline
(``core/valley.py``): frontal moraines and mapped termini by the along-valley
distance of the simulated terminus, lateral moraines, bedrock and wood by the
ice surface on the valley axis. It also switches to the grouped, robust data
term (each landform counts once, summed) and a summed smoothness prior.

``--adjoint_slope godunov`` differentiates the transport as first-order upwind
(the superbee adjoint is unstable). ``--no_emulator_grad`` also holds the
emulated velocity constant within each step when differentiating: the full
adjoint is accurate where it stays bounded (-39 against -42 m/C by finite
differences) but at one Lateglacial iterate it grew to 3e24 in a single cell
holding 6 m of ice, so the gradient overflowed. The frozen-velocity gradient
stays bounded, halves the backward pass, and points the same way as the full one
(cosine 0.93 where both exist) at about 2.7 times the size, a scale Adam ignores.
A non-finite gradient makes the loop go back to the best iterate with half the
step (and write ``nonfinite_<i>.json``) instead of repeating the same step.

``--climate phys`` replaces the stake-only mass-balance calibration (4.7 m/yr of
precipitation) by literature values (1.8 m/yr, see ``CLIMATES``); ``--t_bounds``
keeps the temperature inside the literature envelope ``T_ENVELOPE``.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from config.read_config import parse_arguments
from core.emulator import load_emulator
from core.forward_schemes.igm_emulator_step import use_adjoint_transport, use_compiled_transport
from core.glacier import GlacierDynamicsCheckpointed
from core.moraine_obs import (describe, evaluate, invert_observations, select,
                              terminus_position)
from core.utils import device
from data.loader import load_argentiere

# Stake calibration (notebooks/Argentiere_temperature_inversion.ipynb, section 2):
# with the temperature at 2000 m pinned to 1.0 C for 2012-2021, the stakes give
# melt factor and precipitation.
# Precipitation follows the temperature control by Clausius-Clapeyron,
# P(t) = precip * (1 + precip_cc) ** (T(t) - T_present).
BASE = dict(melt_factor=0.26, precip=4.7, T_s=10.0, z_ref=2000.0, T_present=1.0,
            precip_cc=0.07, tau_ref=0.2154, arrhenius=78.0)

# ``--climate phys``: the stake-only calibration needs 4.7 m/yr of precipitation, which
# gives a glacier-wide accumulation of 3.2 m w.e./yr, twice the measured glacier-wide winter
# balance (1.47 m w.e., Pleiades 2012-2020, Davaze et al. 2020). Here the precipitation is
# set from the literature instead: 1.8 m/yr (Chamonix 1.28 m/yr at 1042 m; Col du Dome
# ~2.5 m w.e./yr at 4250 m), which gives 1.40 m w.e./yr of accumulation over the glacier.
# T_s = 9.4 C is Chamonix's seasonal half-range (January -2.2, July 16.5 C, 1981-2010).
# The temperature and the degree-day factor (bounded to 10 mm w.e./K/d) are then fitted to
# the Pleiades glacier-wide balance, -0.66 m w.e./yr: -0.89 C at 2000 m, 2 C below the
# Chamonix value extrapolated with the lapse rate. This is an effective on-glacier
# temperature for this monthly PDD scheme; the stakes are fitted to rmse 0.66 m w.e.
# (bias +0.46: the ablation zone melts less than measured).
# Calibration script: scratchpad phys/calib_final.py, shown in the notebook (section 9).
CLIMATES = {
    "stakes": BASE,
    "phys": dict(BASE, melt_factor=0.3044, precip=1.8, T_s=9.4, T_present=-0.89),
}

# Literature envelope for the temperature, as anomalies (C) from pre-industrial
# (1871-1900), by age in ka BP: (older bound, younger bound, low, high). The present
# (2012-2021) is taken as pre-industrial + 2.0 C (MeteoSwiss: about 2 C to 1991-2020).
# Lateglacial: Protin et al. 2019 (Argentiere: Younger Dryas / early Holocene summers 3.0-5.5 C
# below present), Kerschner & Ivy-Ochs (Egesen stadial: summer -3.5 C), Heiri et al. 2015 and
# Samartin et al. 2012 (chironomids: Younger Dryas ~2 C below the Allerod, early Holocene up to
# 1-1.5 C above pre-industrial). Late Holocene: Alpine summer reconstructions stay within about
# -2/+1 C of pre-industrial before 1900.
PRESENT_MINUS_PREINDUSTRIAL = 2.0
T_ENVELOPE = [
    (13.6, 12.9, -4.0, +0.5),     # Allerod
    (12.9, 11.7, -6.5, -1.0),     # Younger Dryas
    (11.7, 10.0, -4.0, +1.0),     # Preboreal, early Holocene
    (10.0, 4.0, -2.5, +1.5),      # early-mid Holocene
    (4.0, 0.05, -2.0, +1.0),      # late Holocene, LIA (to 1900 CE)
    (0.05, -0.1, -1.0, +3.0),     # 1900 CE to today
]


def temperature_bounds(node_times, T_present):
    """Lower and upper bound (C at z_ref) of every node from :data:`T_ENVELOPE`."""
    shift = T_present - PRESENT_MINUS_PREINDUSTRIAL
    lo, hi = [], []
    for t in node_times:
        ka = (1950.0 - t) / 1000.0
        # first period containing the age; older than the envelope -> its oldest period
        a, b = next(((l, h) for old, young, l, h in T_ENVELOPE if young <= ka <= old),
                    T_ENVELOPE[0][2:] if ka > T_ENVELOPE[0][0] else T_ENVELOPE[-1][2:])
        lo.append(a + shift); hi.append(b + shift)
    return np.array(lo), np.array(hi)

WINDOWS = {
    "lg": dict(t_start=1950.0 - 13600.0, t_end=1950.0 - 9000.0, node_dt=200.0,
               dt_snap=100.0, mask="catchment_lateglacial", init_T=0.0,
               anchor=False),
    "hp": dict(t_start=-2000.0, t_end=2021.8, node_dt=100.0, dt_snap=20.0,
               mask="catchment_holocene", init_T=1.2, anchor=True),
}


def build(window, dx=200, compile_net=True, chunk_steps=0, max_velbar=300.0,
          cfl=0.3, smb_freq=10.0, morphostrat=True, base=BASE, valley=False,
          adjoint_slope=None, emulator_grad=True):
    """Glacier model, observations and node times for one window."""
    w = WINDOWS[window]
    args = parse_arguments([])
    args.dx = args.dy = int(dx)
    args.t_start = torch.tensor(w["t_start"])
    args.ttot = torch.tensor(w["t_end"])
    args.vis_freq = w["node_dt"]
    args.smb_freq = smb_freq
    args.chunk_steps = chunk_steps
    args.z_ref = base["z_ref"]
    args.max_velbar = max_velbar
    args.emulator_grad = emulator_grad
    args.cfl = cfl
    args.precip_cc = base["precip_cc"]
    args.precip_T_ref = base["T_present"]
    emulator = load_emulator(args.emulator_path, device=device)
    if compile_net:
        # network + transport compiled: ~2.8x faster forward + backward on this
        # grid, gradients equal to 1e-4 relative (the first iteration compiles).
        emulator.network = torch.compile(emulator.network)
        use_compiled_transport(True)
    use_adjoint_transport(adjoint_slope)
    geo, obs = load_argentiere(dx, mask=w["mask"], thk="thk_lia", morphostrat=morphostrat,
                               valley=valley)
    glacier = GlacierDynamicsCheckpointed(
        geo["topg_t"], geo["H_init"], ice_mask=geo["ice_mask"], device=device,
        args=args, emulator=emulator, tau_ref=base["tau_ref"],
        arrhenius=base["arrhenius"])
    obs = select(obs, w["t_start"], w["t_end"], w["dt_snap"])
    n_nodes = int(np.ceil((w["t_end"] - w["t_start"]) / w["node_dt"])) + 1
    node_times = [w["t_start"] + i * w["node_dt"] for i in range(n_nodes)]
    anchor = (2017.0, base["T_present"], 0.3) if w["anchor"] else None
    return glacier, geo, obs, node_times, anchor


def forward_terms(glacier, obs, T_nodes, base=BASE, extra_times=()):
    with torch.no_grad():
        times = sorted({t for o in obs for t in o["times"]} | set(extra_times))
        snaps, H = glacier(precip_tensor=base["precip"], T_m_lowest=T_nodes,
                           T_s=base["T_s"], melt_factor=base["melt_factor"],
                           smb_method="PDD", save_times=times)
        terms = evaluate(obs, snaps, glacier.Z_topo)
    return {k: float(v) for k, v in terms.items()}, snaps, H


def termini(geo, snaps):
    """Soft terminus position (m) along every flowline at every saved time."""
    out = {}
    with torch.no_grad():
        for name, fl in geo.get("flowlines", {}).items():
            path = torch.tensor(fl.active[fl.start:], device=device)
            s = torch.tensor(fl.s[fl.start: fl.stop], dtype=torch.float32, device=device)
            out[name] = {float(t): float(terminus_position(H, path, s))
                         for t, H in sorted(snaps.items())}
    return out


def profiles(geo, snaps, every=1):
    """Ice thickness along every flowline's active part, at every ``every``-th saved time."""
    out = {}
    times = sorted(snaps)[::every]
    for name, fl in geo.get("flowlines", {}).items():
        idx = torch.tensor(fl.active, device=device)
        out[name] = dict(s=fl.s[: fl.stop].tolist(), t=times,
                         H=[snaps[t].reshape(-1)[idx].clamp(min=0).cpu().numpy().round(1).tolist()
                            for t in times])
    return out


def install_adjoint_probe(clip=None):
    """Watch the adjoint of the ice thickness after every time step.

    A hook on the thickness each step returns records the largest |dL/dH| and
    whether it is finite; with ``clip`` it also replaces a non-finite adjoint by
    zero and bounds the rest to +-clip, so one blown-up cell cannot turn the
    whole gradient into NaN. Returns the record, read with :func:`adjoint_report`.
    """
    rec = dict(t=[], gmax=[], ok=[])
    step = GlacierDynamicsCheckpointed._step

    def probed(self, scheme, H_ice, Z_surf, smb, time_):
        H_ice, Z_surf, time_ = step(self, scheme, H_ice, Z_surf, smb, time_)
        if H_ice.requires_grad:
            k = len(rec["t"])
            rec["t"].append(float(time_)); rec["gmax"].append(None); rec["ok"].append(None)

            def hook(g, k=k):
                rec["gmax"][k] = torch.nan_to_num(g.abs(), nan=0.0, posinf=0.0).max()
                rec["ok"][k] = torch.isfinite(g).all()
                if clip:
                    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).clamp(-clip, clip)
            H_ice.register_hook(hook)
        return H_ice, Z_surf, time_

    GlacierDynamicsCheckpointed._step = probed
    return rec


def adjoint_report(rec):
    """Summary of the last backward pass; empties the record."""
    got = [(t, m, o) for t, m, o in zip(rec["t"], rec["gmax"], rec["ok"]) if m is not None]
    for k in rec:
        rec[k].clear()
    if not got:
        return {}
    t = np.array([a for a, _, _ in got])
    m = torch.stack([b for _, b, _ in got]).cpu().numpy()
    ok = torch.stack([c for _, _, c in got]).cpu().numpy()
    return dict(max=float(m.max()), max_at=float(t[m.argmax()]), n_steps=int(len(t)),
                n_nonfinite=int((~ok).sum()),
                first_nonfinite_t=float(t[~ok].max()) if (~ok).any() else None)


def initial_T(path):
    """Temperature nodes of an earlier run (its best iterate), or of a ``{"T": [...]}`` file."""
    d = json.load(open(path))
    return d["best"]["T"] if "best" in d else d["T"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=list(WINDOWS), required=True)
    ap.add_argument("--dx", type=int, default=200)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.15,
                    help="initial Adam learning rate (deg C); cosine-decayed to --lr_min")
    ap.add_argument("--lr_min", type=float, default=0.01)
    ap.add_argument("--no_cc", action="store_true",
                    help="constant precipitation instead of Clausius-Clapeyron scaling")
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--init_T", type=float, default=None)
    ap.add_argument("--no_morphostrat", action="store_true")
    ap.add_argument("--scan", type=float, nargs="*", default=None,
                    help="constant temperatures (C at z_ref) to run forward only")
    ap.add_argument("--chunk_steps", type=int, default=0,
                    help="two-level checkpointing block (0 = off; ~0.26 GB per 1000 "
                         "steps at 200 m without it)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--valley", action="store_true",
                    help="valley-referenced operators, grouped robust data term, summed prior")
    ap.add_argument("--max_velbar", type=float, default=300.0,
                    help="cap on depth-averaged speed (m/yr), 0 = none")
    ap.add_argument("--init_from", default=None,
                    help="history.json of an earlier run (its best iterate) or a {\"T\": [...]} file "
                         "to start from instead of a constant --init_T")
    ap.add_argument("--reg_slope", type=float, default=0.0,
                    help="weight of the summed squared first differences of the nodes (stops nodes "
                         "without data from drifting)")
    ap.add_argument("--adjoint_clip", type=float, default=None,
                    help="bound |dL/dH| per step to this value and zero non-finite entries")
    ap.add_argument("--probe_adjoint", action="store_true",
                    help="report the size of the adjoint every iteration (cheap)")
    ap.add_argument("--no_emulator_grad", action="store_true",
                    help="treat the velocity as constant within each step when differentiating: the "
                         "adjoint then stays bounded where the full one explodes, but it overstates "
                         "the thickness sensitivities")
    ap.add_argument("--cfl", type=float, default=0.3, help="CFL number of the time step")
    ap.add_argument("--adjoint_slope", choices=["godunov"], default=None,
                    help="differentiate the transport with this limiter (forward unchanged); "
                         "godunov passes the finite-difference gradient check, superbee does not")
    ap.add_argument("--climate", choices=list(CLIMATES), default="stakes",
                    help="mass-balance parameters: 'stakes' (P 4.7 m/yr, stake fit only) or 'phys' "
                         "(P 1.8 m/yr from the literature, fitted to the geodetic balance)")
    ap.add_argument("--t_bounds", action="store_true",
                    help="keep every node inside the literature envelope T_ENVELOPE (projected after each step)")
    ap.add_argument("--scan_relative", action="store_true",
                    help="--scan temperatures are anomalies from the climate's T_present")
    opts = ap.parse_args()
    torch.backends.cudnn.deterministic = True     # chunked recompute must replay
    torch.backends.cudnn.benchmark = False
    out = opts.out_dir or f"results/argentiere_{opts.window}_{opts.dx}m"
    os.makedirs(out, exist_ok=True)

    clim = CLIMATES[opts.climate]
    base = dict(clim, precip_cc=None if opts.no_cc else clim["precip_cc"])
    glacier, geo, obs, node_times, anchor = build(
        opts.window, opts.dx, chunk_steps=opts.chunk_steps,
        morphostrat=not opts.no_morphostrat, base=base, valley=opts.valley,
        max_velbar=opts.max_velbar, adjoint_slope=opts.adjoint_slope,
        emulator_grad=not opts.no_emulator_grad, cfl=opts.cfl)
    weighting = dict(weighting="groups", reg_sum=True) if opts.valley else {}
    w = WINDOWS[opts.window]
    print(f"{opts.window}: {w['t_start']:.0f} -> {w['t_end']:.0f} CE, {len(node_times)} nodes, "
          f"{len(obs)} observations, {sum(len(o['times']) for o in obs)} scored snapshots")
    meta = {o["id"]: dict(kind=o["kind"], group=o["group"], t0=o["t0"], t1=o["t1"],
                          year=o["year"], weight=o["weight"], name=o["name"],
                          **{k: o[k] for k in ("valley", "s_obs", "s_req", "group_id",
                                               "sigma_L", "sigma_z") if k in o})
            for o in obs}
    every = [w["t_start"] + w["dt_snap"] * (i + 1)
             for i in range(int((w["t_end"] - w["t_start"]) / w["dt_snap"]))]

    if opts.scan is not None:
        rows = []
        for T in opts.scan:
            T = T + base["T_present"] if opts.scan_relative else T
            t0 = time.time()
            Tn = torch.full((len(node_times), 1), T, device=device)
            terms, snaps, H = forward_terms(glacier, obs, Tn, base, extra_times=every)
            area = float((H > 1).sum()) * opts.dx ** 2 / 1e6
            rows.append(dict(T=T, terms=terms, area_km2=area, seconds=time.time() - t0,
                             describe=describe(obs, snaps, glacier.Z_topo),
                             termini=termini(geo, snaps),
                             profiles=profiles(geo, snaps, every=max(1, len(snaps) // 25))))
            print(f"T={T:+5.1f}: end area {area:6.1f} km2, mean misfit "
                  f"{np.mean(list(terms.values())):8.2f} ({time.time() - t0:.0f} s)", flush=True)
        with open(f"{out}/scan.json", "w") as f:
            json.dump(dict(window=opts.window, meta=meta, rows=rows), f, indent=1)
        return

    if opts.init_from:
        T = torch.tensor(initial_T(opts.init_from), dtype=torch.float32, device=device).view(-1, 1)
        assert T.shape[0] == len(node_times), "--init_from has a different number of nodes"
        T.requires_grad_(True)
    else:
        init_T = w["init_T"] if opts.init_T is None else opts.init_T
        T = torch.full((len(node_times), 1), init_T, device=device, requires_grad=True)
    bounds = None
    if opts.t_bounds:
        lo, hi = temperature_bounds(node_times, base["T_present"])
        bounds = (torch.tensor(lo, dtype=T.dtype, device=device).view(-1, 1),
                  torch.tensor(hi, dtype=T.dtype, device=device).view(-1, 1))
        with torch.no_grad():
            T.copy_(torch.maximum(torch.minimum(T, bounds[1]), bounds[0]))
        print("temperature bounds (C at z_ref):", " ".join(f"[{a:+.1f},{b:+.1f}]" for a, b in zip(lo, hi)))
    optimizer = torch.optim.Adam([T], lr=opts.lr)
    probe = install_adjoint_probe(opts.adjoint_clip) if (opts.probe_adjoint or opts.adjoint_clip) else None
    hist = dict(loss=[], data=[], T=[], T_next=[], grad=[], grad_norm=[], seconds=[], terms=[],
                lr=[], adjoint=[])
    n_bad = 0             # consecutive non-finite gradients
    best = dict(loss=np.inf)
    lr_scale = 1.0        # halved each time a step makes the loss clearly worse
    for i in range(opts.iters):
        t0 = time.time()
        # cosine decay from lr to lr_min over the run, times the backtracking factor
        lr = lr_scale * (opts.lr_min + 0.5 * (opts.lr - opts.lr_min)
                         * (1 + np.cos(np.pi * i / max(1, opts.iters - 1))))
        for gr in optimizer.param_groups:
            gr["lr"] = lr
        optimizer.zero_grad()
        loss, data, terms, _ = invert_observations(
            T, glacier, obs, precip_tensor=base["precip"], T_s=base["T_s"],
            melt_factor=base["melt_factor"], dt_snap=w["dt_snap"],
            reg_lambda=opts.reg_lambda, anchor=anchor, reg_slope=opts.reg_slope, **weighting)
        loss.backward()
        g = float(T.grad.norm())
        adj = adjoint_report(probe) if probe is not None else {}
        if not (np.isfinite(g) and np.isfinite(float(loss))):
            # The same T gives the same gradient, so skipping the step would repeat it
            # for ever: keep the evidence, go back to the best iterate, take smaller steps.
            n_bad += 1
            with open(f"{out}/nonfinite_{i}.json", "w") as f:
                json.dump(dict(iter=i, T=T.detach().flatten().tolist(), loss=float(loss),
                               grad=T.grad.flatten().tolist(), adjoint=adj), f)
            print(f"iter {i}: non-finite {'loss' if not np.isfinite(float(loss)) else 'gradient'} "
                  f"({n_bad} in a row; adjoint {adj}); T kept in nonfinite_{i}.json", flush=True)
            if n_bad >= 3 or not np.isfinite(best["loss"]):
                print("giving up: three non-finite iterates in a row", flush=True)
                break
            lr_scale *= 0.5
            with torch.no_grad():
                T.copy_(torch.tensor(best["T"], device=device).view(-1, 1))
            optimizer = torch.optim.Adam([T], lr=lr * 0.5)
            continue
        n_bad = 0
        hist["loss"].append(float(loss)); hist["data"].append(float(data))
        hist["T"].append(T.detach().flatten().tolist()); hist["grad_norm"].append(g)
        hist["grad"].append(T.grad.flatten().tolist()); hist["adjoint"].append(adj)
        hist["terms"].append({k: float(v) for k, v in terms.items()}); hist["lr"].append(lr)
        if float(loss) < best["loss"]:
            best = dict(loss=float(loss), iter=i, T=T.detach().flatten().tolist())
        elif float(loss) > 1.05 * best["loss"]:
            # overshoot: go back to the best iterate and take smaller steps
            lr_scale *= 0.5
            with torch.no_grad():
                T.copy_(torch.tensor(best["T"], device=device).view(-1, 1))
            optimizer = torch.optim.Adam([T], lr=lr * 0.5)
            print(f"iter {i:3d}: loss {float(loss):.3f} > 1.05 x best {best['loss']:.3f}: "
                  f"back to iter {best['iter']}, lr x {lr_scale:.3g}", flush=True)
            hist["seconds"].append(time.time() - t0)
            continue
        optimizer.step()
        if bounds is not None:
            with torch.no_grad():
                T.copy_(torch.maximum(torch.minimum(T, bounds[1]), bounds[0]))
        hist["seconds"].append(time.time() - t0)
        hist["T_next"].append(T.detach().flatten().tolist())
        print(f"iter {i:3d}: loss {float(loss):8.3f} data {float(data):8.3f} |g| {g:7.3f} "
              f"lr {lr:.3f} T [{T.min():+.2f}, {T.max():+.2f}] ({time.time() - t0:.0f} s)"
              + (f" adjoint max {adj['max']:.2e}" if adj else ""), flush=True)
        with open(f"{out}/history.json", "w") as f:
            json.dump(dict(window=opts.window, settings=vars(opts), base=base,
                           node_times=node_times, meta=meta, history=hist, best=best,
                           bounds=None if bounds is None else
                           [bounds[0].flatten().tolist(), bounds[1].flatten().tolist()]),
                      f, indent=1)

    # final forward pass at the best iterate: per-observation misfits and fields
    T_best = torch.tensor(best["T"], device=device).view(-1, 1)
    terms, snaps, H = forward_terms(glacier, obs, T_best, base, extra_times=every)
    keep = sorted(snaps)[:: max(1, len(snaps) // 60)]
    keep = sorted(set(keep) | {t for t in snaps if t in (1850.0, 2012.7, 2021.7)})
    np.savez_compressed(f"{out}/final_fields.npz", times=np.array(keep),
                        H=np.stack([snaps[t].cpu().numpy() for t in keep]),
                        H_end=H.cpu().numpy())
    with open(f"{out}/final_terms.json", "w") as f:
        json.dump(dict(T=best["T"], node_times=node_times, terms=terms, meta=meta,
                       describe=describe(obs, snaps, glacier.Z_topo),
                       termini=termini(geo, snaps)), f, indent=1)
    print(f"best iter {best['iter']}, loss {best['loss']:.3f}; wrote {out}")


if __name__ == "__main__":
    main()

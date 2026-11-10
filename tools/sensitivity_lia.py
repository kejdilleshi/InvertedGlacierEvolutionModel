"""How much does a wrong melt factor / precipitation / tau_ref cost?

Runs the historical Aletsch experiment -- 1700 to 2017, started from the modern
geometry, scored against the dated extents -- and perturbs one fixed parameter
at a time.  Each parameter's effect is then converted into the temperature
change that would move the glacier the same distance, so the answer reads
directly as a bias on an inverted T: if a +10% melt factor is worth +0.4 C,
then a melt factor 10% too high biases the inverted temperature 0.4 C low.

Stage A scans constant T to locate a reference and measure dTerminus/dT.
Stage B perturbs each parameter at that reference.

    python tools/sensitivity_lia.py                  # full run, ~25 min
    python tools/sensitivity_lia.py --fractions 0.1  # cheaper
"""

import argparse
import json
import os
import time

import netCDF4
import numpy as np
import torch

from config.read_config import parse_arguments
from core.emulator import load_emulator
from core.glacier import GlacierDynamicsCheckpointed
from core.utils import device

EPOCHS = [1880, 1926, 1957, 1980, 2017]
# arrhenius is held fixed (not perturbed -- dropped from the sensitivity sweep
# as below-resolution: see the +-25% result in the notebook history) but the
# forward model still needs a value.
BASE = dict(melt_factor=2.7 / 12, precip=1.8, arrhenius=78.0, tau_ref=0.2154)


def load_observations(path="data/geology.nc"):
    nc = netCDF4.Dataset(path)
    topg = torch.tensor(nc["topg"][:], dtype=torch.float32, device=device)
    obs = {y: torch.tensor(nc[f"surf_{y}"][:], dtype=torch.float32, device=device) - topg
           for y in EPOCHS}
    out = dict(
        topg=topg,
        icemask=torch.tensor(nc["icemask"][:], dtype=torch.float32, device=device),
        H_init=torch.tensor(nc["surf_2017"][:], dtype=torch.float32, device=device) - topg,
        y=np.asarray(nc["y"][:]),
        obs=obs,
    )
    nc.close()
    return out


def score(saved, data):
    """Extent agreement per epoch, plus the terminus position."""
    per_epoch, ious, termini = {}, [], []
    for year in EPOCHS:
        H = saved[year]
        sim, ref = H > 1.0, data["obs"][year] > 1.0
        inter = float((sim & ref).sum())
        union = float((sim | ref).sum())
        iou = inter / union if union else 0.0
        rows = torch.nonzero(sim.any(dim=1)).flatten()
        term = float(data["y"][int(rows.min())]) if rows.numel() else float("nan")
        obs_rows = torch.nonzero(ref.any(dim=1)).flatten()
        obs_term = float(data["y"][int(obs_rows.min())])
        per_epoch[year] = dict(iou=iou, area_km2=float(sim.sum()) * 0.01,
                               obs_area_km2=float(ref.sum()) * 0.01,
                               terminus=term, obs_terminus=obs_term)
        ious.append(iou)
        termini.append(term)
    return dict(per_epoch=per_epoch, mean_iou=float(np.mean(ious)),
                terminus_2017=per_epoch[2017]["terminus"],
                terminus_1880=per_epoch[1880]["terminus"],
                mean_terminus=float(np.mean(termini)))


def run(data, emulator, args, T, params):
    glacier = GlacierDynamicsCheckpointed(
        data["topg"], data["H_init"], ice_mask=data["icemask"], device=device,
        args=args, emulator=emulator,
        tau_ref=params["tau_ref"], arrhenius=params["arrhenius"])
    with torch.no_grad():
        saved, _ = glacier(precip_tensor=params["precip"],
                           T_m_lowest=torch.tensor(T, device=device),
                           T_s=10.0, melt_factor=params["melt_factor"],
                           smb_method="PDD",
                           save_times=[float(y) for y in EPOCHS])
    missing = [y for y in EPOCHS if saved[float(y)] is None]
    if missing:
        raise RuntimeError(f"no snapshot saved for {missing}")
    return score({y: saved[float(y)] for y in EPOCHS}, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", type=float, nargs="+",
                    default=[6.5, 7.0, 7.5, 8.0])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.25])
    ap.add_argument("--dT", type=float, default=0.5,
                    help="step used to measure dTerminus/dT at the reference")
    ap.add_argument("--out", default="results/sensitivity_lia.json")
    opts = ap.parse_args()

    args = parse_arguments([])
    args.t_start = torch.tensor(1500.0)  # matches the corrected spin-up (see notebook Sec. 2)
    args.ttot = torch.tensor(2017.0)
    args.vis_freq = 1e9            # constant T: never resample the control
    emulator = load_emulator(args.emulator_path, device=device)
    data = load_observations()

    print("observed:  " + "  ".join(
        f"{y} {float((data['obs'][y] > 1).sum()) * 0.01:.1f} km2 "
        f"term {data['y'][int(torch.nonzero((data['obs'][y] > 1).any(1)).min())] / 1000:.2f}"
        for y in EPOCHS))

    # ---- Stage A: constant-T scan -> reference point and dTerminus/dT ----
    print(f"\nStage A: constant-T scan, {int(args.t_start)}-2017 from the 2017 geometry")
    scan = []
    for T in opts.temps:
        t0 = time.time()
        s = run(data, emulator, args, T, dict(BASE))
        s["T"] = T
        scan.append(s)
        print(f"  T={T:4.1f}  mean IoU {s['mean_iou']:.3f}  "
              f"terminus 1880 {s['terminus_1880'] / 1000:.2f} km  "
              f"2017 {s['terminus_2017'] / 1000:.2f} km  ({time.time() - t0:.0f}s)")

    best = max(scan, key=lambda s: s["mean_iou"])
    T_ref = best["T"]
    print(f"  reference T = {T_ref} C (best mean IoU {best['mean_iou']:.3f})")

    # dTerminus/dT, measured symmetrically about the reference
    lo = run(data, emulator, args, T_ref - opts.dT, dict(BASE))
    hi = run(data, emulator, args, T_ref + opts.dT, dict(BASE))
    dterm_dT = (hi["mean_terminus"] - lo["mean_terminus"]) / (2 * opts.dT)
    diou_dT = (hi["mean_iou"] - lo["mean_iou"]) / (2 * opts.dT)
    print(f"  dTerminus/dT = {dterm_dT:+.0f} m/C   dIoU/dT = {diou_dT:+.4f} /C")

    # ---- Stage B: one parameter at a time ----
    print(f"\nStage B: perturbations at T={T_ref} C "
          f"(terminus is the mean over the 5 epochs)")
    print(f"{'parameter':12s} {'change':>8s} {'value':>10s} {'mean IoU':>9s} "
          f"{'d terminus':>11s} {'= dT':>8s}")
    rows = [dict(parameter="baseline", fraction=0.0, value=None,
                 mean_iou=best["mean_iou"], d_terminus=0.0, dT_equivalent=0.0,
                 **{k: v for k, v in best.items() if k in ("terminus_1880", "terminus_2017")})]
    print(f"{'baseline':12s} {'-':>8s} {'-':>10s} {best['mean_iou']:9.3f} "
          f"{0.0:11.0f} {0.0:8.2f}")

    for name in ("melt_factor", "precip", "tau_ref"):
        for frac in sorted(set([f for f in opts.fractions] + [-f for f in opts.fractions])):
            params = dict(BASE)
            params[name] = BASE[name] * (1.0 + frac)
            s = run(data, emulator, args, T_ref, params)
            d_term = s["mean_terminus"] - best["mean_terminus"]
            dT_eq = d_term / dterm_dT if dterm_dT else float("nan")
            rows.append(dict(parameter=name, fraction=frac, value=params[name],
                             mean_iou=s["mean_iou"], d_terminus=d_term,
                             dT_equivalent=dT_eq,
                             terminus_1880=s["terminus_1880"],
                             terminus_2017=s["terminus_2017"]))
            print(f"{name:12s} {frac * 100:+7.0f}% {params[name]:10.4f} "
                  f"{s['mean_iou']:9.3f} {d_term:+11.0f} {dT_eq:+8.2f}")

    os.makedirs(os.path.dirname(opts.out), exist_ok=True)
    with open(opts.out, "w") as f:
        json.dump(dict(settings=vars(opts), base=BASE, T_ref=T_ref,
                       dterm_dT=dterm_dT, diou_dT=diou_dT,
                       scan=scan, perturbations=rows), f, indent=2)
    print("\nwrote", opts.out)
    print("Read the last column as the bias a wrong parameter puts on an "
          "inverted temperature, with the sign flipped:")
    print("  a melt factor 10% too high makes the model too warm, so the "
          "inversion compensates by returning a temperature that much too low.")


if __name__ == "__main__":
    main()

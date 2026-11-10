"""Forward scan: which constant temperature puts the ice surface on each moraine.

Holds precipitation and the melt factor fixed, as the synthetic temperature test
did, and runs the glacier to near steady state for a range of ``T_m_lowest``.
For every dated moraine it reports the median mismatch between the simulated ice
surface and the moraine elevation, so the temperature that reproduces each stand
can be read off directly.  This is the map the gradient inversion then descends.

    python tools/scan_yd_temperature.py --temps 3 4 5 6 7 --years 400
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from config.read_config import parse_arguments
from core.emulator import load_emulator
from core.glacier import GlacierDynamicsCheckpointed
from core.utils import device
from data.loader import load_yd_aletsch


def surface_misfit(H, topo, moraine):
    """Median and spread of (simulated surface - moraine elevation), in metres.

    Scored on the inboard companion cells, not on the moraine itself: an
    ice-free moraine cell has surface == bed == moraine elevation and would
    otherwise score a perfect fit.  See data/YD-Aletsch/README.md.
    """
    m = moraine["inboard"]
    resid = ((topo + H)[m] - moraine["z_target"][m]).detach().cpu().numpy()
    return dict(median=float(np.median(resid)),
                p10=float(np.percentile(resid, 10)),
                p90=float(np.percentile(resid, 90)),
                frac_ice=float((H[m] > 1.0).float().mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", type=float, nargs="+",
                    default=[3.0, 4.0, 5.0, 6.0, 7.0],
                    help="T_m_lowest values to scan (C at the domain minimum)")
    ap.add_argument("--years", type=float, default=400.0)
    ap.add_argument("--precip", type=float, default=1.8)
    ap.add_argument("--melt_factor", type=float, default=2.7 / 12)
    ap.add_argument("--T_s", type=float, default=10.0)
    ap.add_argument("--mask", default="catchment",
                    choices=["catchment", "icemask"])
    ap.add_argument("--max_velbar", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    opts = ap.parse_args()
    if opts.out is None:
        opts.out = f"results/yd_temperature_scan_{opts.mask}.json"

    args = parse_arguments([])
    args.max_velbar = opts.max_velbar
    args.vis_freq = 1e9          # constant climate: never resample the control
    args.t_start = torch.tensor(0.0)
    args.ttot = torch.tensor(opts.years)

    emulator = load_emulator(args.emulator_path, device=device)
    topo, H_init, ice_mask, moraines = load_yd_aletsch(mask=opts.mask)
    glacier = GlacierDynamicsCheckpointed(topo, H_init, ice_mask=ice_mask,
                                          device=device, args=args,
                                          emulator=emulator)

    save_times = [opts.years * f for f in (0.25, 0.5, 1.0)]
    results = []
    for T in opts.temps:
        t0 = time.time()
        with torch.no_grad():
            snaps, H = glacier(precip_tensor=opts.precip,
                               T_m_lowest=torch.tensor(T, device=device),
                               T_s=opts.T_s, melt_factor=opts.melt_factor,
                               smb_method="PDD", save_times=save_times)
        entry = dict(T_m_lowest=T,
                     area_km2=float((H > 1).sum()) * 0.01,
                     volume_km3=float(H.sum()) * 1e4 / 1e9,
                     max_thk_m=float(H.max()),
                     seconds=time.time() - t0)
        # How far from steady state: volume change over the last quarter.
        vols = [float(s.sum()) * 1e4 / 1e9 for s in snaps.values() if s is not None]
        entry["volume_km3_snapshots"] = vols
        for name, mor in moraines.items():
            entry[name] = surface_misfit(H, topo, mor)
        results.append(entry)

        print(f"T={T:4.1f} C  {entry['area_km2']:6.1f} km2  "
              f"{entry['volume_km3']:6.1f} km3  max {entry['max_thk_m']:5.0f} m  "
              f"({entry['seconds']:.0f} s)")
        for name in ("LIA_W", "LIA_E", "YD1", "YD2"):
            d = entry[name]
            print(f"      {name:6s} surface - target: {d['median']:+6.0f} m "
                  f"[{d['p10']:+5.0f},{d['p90']:+5.0f}]   ice on "
                  f"{d['frac_ice'] * 100:3.0f}% of its cells")

    os.makedirs(os.path.dirname(opts.out), exist_ok=True)
    with open(opts.out, "w") as f:
        json.dump(dict(settings=vars(opts), results=results), f, indent=2)
    print("\nwrote", opts.out)


if __name__ == "__main__":
    main()

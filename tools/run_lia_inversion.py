"""The POC experiment: invert for T_m_lowest(t), 1600-2017, extent-only, 5 epochs.

Starts from the modern (2017) geometry and integrates forward. A 180-year
spin-up (starting the run in 1700) undershoots the observed 1880 volume by 9%
and misplaces the terminus by 1 km (see spin_up_diagnostic.png); starting in
1600 recovers 97% of the volume and the terminus to within 100 m, so the run
starts there. Climate is free from 1600, but only the 1880-2017 observations
constrain it -- the 1600-1850 nodes are shaped by the smoothness prior alone,
not data, and should be read as a warm-up, not a reconstruction.

Precipitation, melt factor, tau_ref and the Arrhenius factor are held at their
calibrated values throughout (see results/sensitivity_lia.json for what
getting them wrong would cost).

    python tools/run_lia_inversion.py --iters 50
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
from core.inversion import invert_extent
from core.utils import device
from data.loader import analysis_mask

EPOCHS = [1880, 1926, 1957, 1980, 2017]
BASE = dict(melt_factor=2.7 / 12, precip=1.8, T_s=10.0,
           arrhenius=78.0, tau_ref=0.2154)
OUT_DIR = "results/lia_1700_2017"


def load_observations(path="data/geology.nc"):
    nc = netCDF4.Dataset(path)
    topg = torch.tensor(nc["topg"][:], dtype=torch.float32, device=device)
    obs = {y: torch.tensor(nc[f"surf_{y}"][:], dtype=torch.float32, device=device) - topg
           for y in EPOCHS}
    out = dict(
        topg=topg,
        icemask=torch.tensor(nc["icemask"][:], dtype=torch.float32, device=device),
        H_init=torch.tensor(nc["surf_2017"][:], dtype=torch.float32, device=device) - topg,
        x=np.asarray(nc["x"][:]), y=np.asarray(nc["y"][:]),
        obs=obs,
    )
    nc.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_start", type=float, default=1600.0)
    ap.add_argument("--vis_freq", type=float, default=50.0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reg_lambda", type=float, default=0.01)
    ap.add_argument("--init_T", type=float, default=7.0)
    ap.add_argument("--clip", type=float, default=0.0,
                    help="max gradient norm; 0 disables clipping")
    ap.add_argument("--elev_cut", type=float, default=None,
                    help="bed elevation (m) at and above which the loss weight is 0; "
                         "unset = no mask (the original full-domain loss)")
    ap.add_argument("--mask_width", type=float, default=50.0,
                    help="metres over which the mask rises from 0 to ~1 below elev_cut")
    ap.add_argument("--out_dir", default=OUT_DIR)
    opts = ap.parse_args()
    os.makedirs(opts.out_dir, exist_ok=True)

    args = parse_arguments([])
    args.t_start = torch.tensor(opts.t_start)
    args.ttot = torch.tensor(2017.0)
    args.vis_freq = opts.vis_freq
    emulator = load_emulator(args.emulator_path, device=device)
    data = load_observations()

    glacier = GlacierDynamicsCheckpointed(
        data["topg"], data["H_init"], ice_mask=data["icemask"], device=device,
        args=args, emulator=emulator,
        tau_ref=BASE["tau_ref"], arrhenius=BASE["arrhenius"])

    mask = None
    if opts.elev_cut is not None:
        mask = analysis_mask(data["topg"], opts.elev_cut, opts.mask_width)
        np.save(f"{opts.out_dir}/analysis_mask.npy", mask.cpu().numpy())
        print(f"analysis mask: weight 0 for bed >= {opts.elev_cut:.0f} m, "
              f"{int((mask > 0.5).sum())} of {mask.numel()} cells above 0.5")

    n_nodes = int((float(args.ttot) - opts.t_start) // opts.vis_freq) + 1
    node_times = [opts.t_start + i * opts.vis_freq for i in range(n_nodes)]
    print(f"control: {n_nodes} nodes at {node_times}")

    T = torch.full((n_nodes, 1), opts.init_T, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([T], lr=opts.lr)

    history = dict(loss=[], data=[], T=[], grad_norm=[], seconds=[],
                   iou={y: [] for y in EPOCHS}, area={y: [] for y in EPOCHS})
    weights = {y: 1.0 for y in EPOCHS}

    best_loss, best_state, skipped = float("inf"), None, 0
    for i in range(opts.iters):
        t0 = time.time()
        optimizer.zero_grad()
        sim, loss, data_term, metrics = invert_extent(
            T, data["obs"], glacier, precip_tensor=BASE["precip"],
            T_s=BASE["T_s"], melt_factor=BASE["melt_factor"],
            weights=weights, reg_lambda=opts.reg_lambda,
            analysis_mask=mask)
        loss.backward()
        grad_norm = float(T.grad.norm())

        # A non-finite gradient must never reach Adam: it writes NaN into T and
        # every later forward pass is NaN, which looks like a converged-then-
        # dead run rather than a failure. Drop the step instead.
        if not (np.isfinite(grad_norm) and np.isfinite(float(loss))):
            optimizer.zero_grad()
            skipped += 1
            print(f"iter {i:3d}/{opts.iters}: non-finite gradient "
                  f"(|grad| {grad_norm}) -- step skipped ({skipped} so far)",
                  flush=True)
            continue
        if opts.clip > 0:
            torch.nn.utils.clip_grad_norm_([T], opts.clip)
        optimizer.step()
        dt = time.time() - t0

        history["loss"].append(float(loss))
        history["data"].append(float(data_term))
        history["T"].append(T.detach().cpu().numpy().flatten().tolist())
        history["grad_norm"].append(grad_norm)
        history["seconds"].append(dt)
        for y in EPOCHS:
            history["iou"][y].append(float(metrics[y]["iou"]))
            history["area"][y].append(float(metrics[y]["area"]))

        if float(loss) < best_loss:
            best_loss = float(loss)
            best_state = dict(T=T.detach().cpu().numpy().flatten().tolist(), iter=i)

        print(f"iter {i:3d}/{opts.iters}: loss {float(loss):.5f}  "
              f"data {float(data_term):.5f}  |grad| {grad_norm:.4f}  "
              f"mean IoU {np.mean([history['iou'][y][-1] for y in EPOCHS]):.3f}  "
              f"({dt:.0f}s)", flush=True)

        # Checkpoint every 5 iterations so a killed run still has usable output.
        if i % 5 == 0 or i == opts.iters - 1:
            with open(f"{opts.out_dir}/inversion_history.json", "w") as f:
                json.dump(dict(settings=vars(opts), base=BASE, node_times=node_times,
                               epochs=EPOCHS, history=history, best=best_state),
                         f, indent=2)

    # ---- final forward pass at the best-loss T, save full fields for plotting ----
    T_best = torch.tensor(best_state["T"], device=device).view(-1, 1)
    with torch.no_grad():
        sim_final, loss_f, data_f, metrics_f = invert_extent(
            T_best, data["obs"], glacier, precip_tensor=BASE["precip"],
            T_s=BASE["T_s"], melt_factor=BASE["melt_factor"],
            weights=weights, reg_lambda=opts.reg_lambda,
            analysis_mask=mask)

    np.savez(f"{opts.out_dir}/final_fields.npz",
            x=data["x"], y=data["y"], topg=data["topg"].cpu().numpy(),
            icemask=data["icemask"].cpu().numpy(),
            **{f"sim_{y}": sim_final[y].cpu().numpy() for y in EPOCHS},
            **{f"obs_{y}": data["obs"][y].cpu().numpy() for y in EPOCHS})

    print(f"\nbest iter {best_state['iter']}: loss {best_loss:.5f}")
    if skipped:
        print(f"{skipped} of {opts.iters} iterations skipped on a non-finite gradient")
    print("T at best iter:", [round(v, 3) for v in best_state["T"]])
    print(f"wrote {opts.out_dir}/inversion_history.json and final_fields.npz")


if __name__ == "__main__":
    main()

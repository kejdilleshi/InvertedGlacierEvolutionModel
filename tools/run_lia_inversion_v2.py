"""V2 of the LIA temperature inversion -- three fixes, each with evidence.

V1 (tools/run_lia_inversion.py) recovered ~4% of the temperature trend the data
demands. The diagnosis, in the V2 notebook:

1. **The loss was dominated by misfit no temperature could remove.** 36% of the
   extent disagreement -- 60% of it at 1880 -- sat above 2944 m, the ELA at the
   warmest temperature in the scan. Those cells are in the accumulation zone for
   every value of the control, on 34-degree slopes with tens of metres of ice.
   Masking the loss to bed < 3000 m keeps 100% of the reducible signal and drops
   96% of the irreducible part, lifting signal purity from 64% to 98%.

2. **It started 1.2 C from the answer, and flat.** Inverting the constant-T
   terminus curve epoch by epoch gives an implied history of 6.14 C (1880)
   rising to 7.50 C (2017). V1 began at a flat 7.0 C and, with Adam's
   normalized steps, marched every node upward together instead of tilting them.
   V2 starts from that implied curve.

3. **Four of eleven nodes had no data to fit.** Nothing constrains 1500-1650:
   the first observation is 1880 and the glacier forgets its initial state in
   ~100-200 years. V2 ties them to the 1700 node, leaving 7 free parameters
   for 5 observations instead of 11.

4. **One bad gradient can freeze Adam.** At V1 iteration 27 the gradient norm
   jumped from 0.0013 to 7.47 -- four orders of magnitude -- while the loss went
   on falling smoothly, so it is an adjoint artifact, not a real feature of the
   objective. (The CFL step length is read through ``.item()``, so autograd sees
   it as a constant; when the limiter switches the step, the adjoint takes a
   jolt.) Adam divides by a running sqrt of the squared gradient, so a spike
   that large inflates the denominator and throttles every later step, with a
   half-life of ~700 iterations at beta2=0.999. V2 clips the gradient norm.

    python tools/run_lia_inversion_v2.py --iters 25
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

EPOCHS = [1880, 1926, 1957, 1980, 2017]
BASE = dict(melt_factor=2.7 / 12, precip=1.8, T_s=10.0,
            arrhenius=78.0, tau_ref=0.2154)
OUT_DIR = "results/lia_v2"
# Implied constant-T per epoch, from inverting the terminus-vs-T scan in
# results/sensitivity_lia.json (see the V2 notebook for the derivation).
IMPLIED_T = {1880: 6.14, 1926: 6.64, 1957: 6.86, 1980: 7.14, 2017: 7.50}


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


def warm_start(node_times):
    """The implied-T curve sampled onto the control nodes, flat before 1880."""
    ys = np.array(sorted(IMPLIED_T))
    vs = np.array([IMPLIED_T[y] for y in ys])
    return np.interp(np.asarray(node_times), ys, vs)   # np.interp clamps at the ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_start", type=float, default=1500.0)
    ap.add_argument("--vis_freq", type=float, default=50.0)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--reg_lambda", type=float, default=0.01)
    ap.add_argument("--elev_cut", type=float, default=3000.0,
                    help="loss is averaged over cells with bed below this")
    ap.add_argument("--n_tied", type=int, default=4,
                    help="leading nodes tied to the first free node (spin-up)")
    ap.add_argument("--clip", type=float, default=0.01,
                    help="max gradient norm")
    ap.add_argument("--warm", choices=["implied", "flat"], default="implied",
                    help="'implied' starts from the equilibrium-inverted curve, "
                         "'flat' from a uniform init_T as V1 did")
    ap.add_argument("--init_T", type=float, default=7.0,
                    help="the uniform value used by --warm flat")
    ap.add_argument("--param", choices=["nodes", "level_dev"], default="nodes",
                    help="'nodes' optimizes the node values directly, as V1 did. "
                         "'level_dev' splits them into one scalar level plus "
                         "zero-mean deviations, so Adam gives the shape its own "
                         "step size instead of leaving it to emerge from small "
                         "differences between correlated node gradients")
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

    n_nodes = int((float(args.ttot) - opts.t_start) // opts.vis_freq) + 1
    node_times = [opts.t_start + i * opts.vis_freq for i in range(n_nodes)]

    # ---- fix 1: the loss only sees elevations the control can act on ----
    mask = data["topg"] < opts.elev_cut
    print(f"analysis mask: {int(mask.sum())} of {mask.numel()} cells "
          f"(bed < {opts.elev_cut:.0f} m)")

    # ---- fix 2 + 3: warm start, and tie the unconstrained spin-up nodes ----
    if opts.warm == "implied":
        init = warm_start(node_times)
    else:
        init = np.full(len(node_times), opts.init_T, dtype=float)
    n_tied = opts.n_tied
    free_init = init[n_tied:]

    print(f"control: {n_nodes} nodes at {node_times}")
    print(f"         {n_tied} tied to the {node_times[n_tied]:.0f} node, "
          f"{len(free_init)} free")
    print(f"parameterization: {opts.param};  warm start: {opts.warm}")
    print("initial T:", [round(v, 2) for v in init])

    if opts.param == "nodes":
        T_free = torch.tensor(free_init, dtype=torch.float32,
                              device=device, requires_grad=True)
        params = [T_free]
        def nodes():
            return T_free
    else:
        # One scalar for the level, one zero-mean vector for the shape. Adam
        # keeps a separate second-moment estimate for each, so the shape steps
        # at the full learning rate from iteration 1 instead of having to
        # emerge from the residue of near-identical node gradients.
        level = torch.tensor(float(free_init.mean()), dtype=torch.float32,
                             device=device, requires_grad=True)
        dev = torch.tensor(free_init - free_init.mean(), dtype=torch.float32,
                           device=device, requires_grad=True)
        params = [level, dev]
        def nodes():
            return level + (dev - dev.mean())

    optimizer = torch.optim.Adam(params, lr=opts.lr)
    history = dict(loss=[], data=[], T=[], grad_norm=[], seconds=[],
                   iou={y: [] for y in EPOCHS})
    best_loss, best_state = float("inf"), None

    for i in range(opts.iters):
        t0 = time.time()
        optimizer.zero_grad()
        T_nodes = nodes()
        T_full = torch.cat([T_nodes[:1].expand(n_tied), T_nodes]).unsqueeze(1)
        sim, loss, data_term, metrics = invert_extent(
            T_full, data["obs"], glacier, precip_tensor=BASE["precip"],
            T_s=BASE["T_s"], melt_factor=BASE["melt_factor"],
            weights={y: 1.0 for y in EPOCHS}, reg_lambda=opts.reg_lambda,
            analysis_mask=mask)
        loss.backward()
        grad_norm = float(torch.cat([p.grad.flatten() for p in params]).norm())
        clipped = grad_norm > opts.clip
        if opts.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, opts.clip)
        optimizer.step()
        dt = time.time() - t0

        T_now = T_full.detach().cpu().numpy().flatten().tolist()
        history["loss"].append(float(loss))
        history["data"].append(float(data_term))
        history["T"].append(T_now)
        history["grad_norm"].append(grad_norm)
        history["seconds"].append(dt)
        for y in EPOCHS:
            history["iou"][y].append(float(metrics[y]["iou"]))

        if float(loss) < best_loss:
            best_loss = float(loss)
            best_state = dict(T=T_now, iter=i)

        print(f"iter {i:3d}/{opts.iters}: loss {float(loss):.5f}  "
              f"|grad| {grad_norm:.4f}{' CLIPPED' if clipped else ''}  "
              f"mean IoU {np.mean([history['iou'][y][-1] for y in EPOCHS]):.3f}  "
              f"T {min(T_now):.2f}-{max(T_now):.2f}  ({dt:.0f}s)", flush=True)

        if i % 5 == 0 or i == opts.iters - 1:
            with open(f"{opts.out_dir}/inversion_history.json", "w") as f:
                json.dump(dict(settings=vars(opts), base=BASE,
                               node_times=node_times, epochs=EPOCHS,
                               warm_start=init.tolist(), n_tied=n_tied,
                               history=history, best=best_state), f, indent=2)

    T_best = torch.tensor(best_state["T"], device=device).view(-1, 1)
    with torch.no_grad():
        sim_final, _, _, _ = invert_extent(
            T_best, data["obs"], glacier, precip_tensor=BASE["precip"],
            T_s=BASE["T_s"], melt_factor=BASE["melt_factor"],
            weights={y: 1.0 for y in EPOCHS}, reg_lambda=opts.reg_lambda,
            analysis_mask=mask)
    np.savez(f"{opts.out_dir}/final_fields.npz",
             x=data["x"], y=data["y"], topg=data["topg"].cpu().numpy(),
             analysis_mask=mask.cpu().numpy(),
             **{f"sim_{y}": sim_final[y].cpu().numpy() for y in EPOCHS},
             **{f"obs_{y}": data["obs"][y].cpu().numpy() for y in EPOCHS})

    print(f"\nbest iter {best_state['iter']}: loss {best_loss:.5f}")
    print("T at best iter:", [round(v, 3) for v in best_state["T"]])
    print(f"wrote {opts.out_dir}/")


if __name__ == "__main__":
    main()

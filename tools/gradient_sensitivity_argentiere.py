"""Sensitivity of the inverted Lateglacial temperature to every fixed parameter, from gradients.

``tools/sensitivity_argentiere.py`` finds, for one perturbed parameter at a time, the uniform
temperature shift ``u`` that restores the fit (three or more forward runs per value). To first
order the same answer comes from derivatives at the optimum (implicit function theorem):

    du*/dtheta = - L_u,theta / L_uu

``L_u = sum_i dL/dT_i`` is the gradient along a uniform shift. Two forward + backward runs, at
``T* - d`` and ``T* + d``, with every parameter a leaf tensor, give ``L_u,theta`` for all
parameters at once (central difference in ``u`` of ``dL/dtheta``) and ``L_uu`` (of ``L_u``).
Taking both from the same adjoint cancels its overall scale (the frozen-velocity gradient is
~2.7x the full one).

Parameters: precipitation, melt factor, seasonal half-range, lapse rate, the Clausius-Clapeyron
reference temperature (``T_present``), and with ``--emulator_grad`` (full adjoint through the
ice-flow emulator) the flow-law factor and the sliding stress. The speed cap is not
differentiable.

    IGEM_DEVICE=cuda:0 python tools/gradient_sensitivity_argentiere.py --u -0.2 --emulator_grad
    IGEM_DEVICE=cuda:1 python tools/gradient_sensitivity_argentiere.py --u +0.2 --emulator_grad
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import core.climate as climate
from core.moraine_obs import invert_observations
from core.utils import device
from tools.run_argentiere_inversion import (CLIMATES, WINDOWS, adjoint_report, build,
                                            install_adjoint_probe)

RUN = "results/argentiere_lg_100m_phys"
OUT = f"{RUN}/sensitivity/gradients"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", type=float, required=True, help="uniform shift of the best T (C)")
    ap.add_argument("--emulator_grad", action="store_true",
                    help="full adjoint through the emulator (needed for arrhenius, tau_ref)")
    ap.add_argument("--adjoint_clip", type=float, default=0.05)
    opts = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    s = json.load(open(f"{RUN}/history.json"))["settings"]
    base = CLIMATES["phys"]
    best = json.load(open(f"{RUN}/final_terms.json"))
    glacier, geo, obs, node_times, _ = build(
        "lg", s["dx"], chunk_steps=s["chunk_steps"], base=base, valley=True,
        max_velbar=s["max_velbar"], cfl=s["cfl"], adjoint_slope="godunov",
        emulator_grad=opts.emulator_grad)
    probe = install_adjoint_probe(opts.adjoint_clip)

    leaf = lambda v: torch.tensor(float(v), device=device, requires_grad=True)
    th = dict(precip=leaf(base["precip"]), melt_factor=leaf(base["melt_factor"]), T_s=leaf(base["T_s"]),
              lapse=leaf(7.0), T_present=leaf(base["T_present"]), arrhenius=leaf(base["arrhenius"]),
              tau_ref=leaf(base["tau_ref"]))
    climate.LAPSE_RATE = th["lapse"] / 1000.0
    glacier.precip_T_ref = th["T_present"]
    glacier.arrhenius = th["arrhenius"] * torch.ones_like(glacier.Z_topo)
    glacier.tau_ref = th["tau_ref"] * torch.ones_like(glacier.Z_topo)
    T = torch.tensor(np.array(best["T"]) + opts.u, dtype=torch.float32, device=device).view(-1, 1)
    T.requires_grad_(True)

    t0 = time.time()
    loss, data, terms, _ = invert_observations(
        T, glacier, obs, precip_tensor=th["precip"], T_s=th["T_s"], melt_factor=th["melt_factor"],
        dt_snap=WINDOWS["lg"]["dt_snap"], reg_lambda=s["reg_lambda"], weighting="groups", reg_sum=True)
    t1 = time.time()
    loss.backward()
    adj = adjoint_report(probe)
    g = {k: (None if v.grad is None else float(v.grad)) for k, v in th.items()}
    out = dict(u=opts.u, emulator_grad=opts.emulator_grad, loss=float(loss), data=float(data),
               grad_T=T.grad.flatten().tolist(), L_u=float(T.grad.sum()), grad_params=g,
               params={k: float(v) for k, v in th.items()}, adjoint=adj,
               terms={k: float(v) for k, v in terms.items()},
               seconds_forward=t1 - t0, seconds_backward=time.time() - t1)
    tag = f"u{opts.u:+.2f}_{'full' if opts.emulator_grad else 'frozen'}"
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"u {opts.u:+.2f}: loss {float(loss):.3f}, L_u {out['L_u']:+.3f}, "
          + ", ".join(f"dL/d{k} {v:+.4g}" for k, v in g.items() if v is not None)
          + f" ({out['seconds_forward']:.0f} + {out['seconds_backward']:.0f} s), adjoint {adj}", flush=True)


if __name__ == "__main__":
    main()

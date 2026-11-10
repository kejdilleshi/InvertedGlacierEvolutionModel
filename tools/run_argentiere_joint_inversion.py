"""Joint inversion of the Lateglacial temperature T(t) and five fixed parameters.

Controls: the 24 temperature nodes (as anomalies to 2012-2021) and precipitation P, seasonal
half-range T_s, degree-day factor DDF, Glen's flow-law factor A and the sliding stress tau_ref.

Every control is optimized as a fraction of its allowed range, z in [0, 1] (ranges below; the
temperature nodes use the literature envelope), with one Adam learning rate and no prior on the
parameters, so no control is favoured. Adam scales each control's step by its own gradient history,
so a step moves every control by about the same fraction of its range.

The present-day temperature follows the mass-balance parameters: it is re-fitted to the geodetic
2012-2021 balance at every iterate (``present_T``), and T(t) = T_present + anomaly. Its derivative is
carried to first order, so the gradient of P, T_s and DDF includes the re-fit.

Gradients: P, T_s, DDF and the nodes from the frozen-velocity upwind adjoint (as the temperature
inversion); A and tau_ref enter the loss only through the ice-flow emulator, whose input gradient was
found unreliable (tools/gradient_sensitivity_argentiere.py), so theirs are one-sided finite
differences: one extra forward run each per iteration.

    IGEM_DEVICE=cuda:0 python -u tools/run_argentiere_joint_inversion.py --iters 30
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from core.moraine_obs import invert_observations
from core.utils import device
from tools.run_argentiere_inversion import (CLIMATES, PRESENT_MINUS_PREINDUSTRIAL, WINDOWS, T_ENVELOPE,
                                            adjoint_report, build, install_adjoint_probe)
from tools.sensitivity_argentiere import DAY, LAPSE, present_T

START = "results/argentiere_lg_100m_phys"        # best T(t) of the fixed-parameter inversion
# allowed range of each parameter: the tested ranges (presentation slide 4), except P: 1.0-1.8 m/yr (user, 2026-10-30)
# and tau_ref widened downward,
# 0.10-0.25 MPa, inside the emulator's training range
RANGES = dict(precip=(1.0, 1.8), T_s=(7.9, 10.9), ddf=(8.0, 12.0), arrhenius=(52.0, 117.0), tau_ref=(0.10, 0.25))
FD_KEYS = ("arrhenius", "tau_ref")


def anomaly_bounds(node_times):
    """Envelope of each node as an anomaly to 2012-2021 (C)."""
    lo, hi = [], []
    for t in node_times:
        ka = (1950.0 - t) / 1000.0
        a, b = next(((l, h) for old, young, l, h in T_ENVELOPE if young <= ka <= old), T_ENVELOPE[0][2:])
        lo.append(a - PRESENT_MINUS_PREINDUSTRIAL); hi.append(b - PRESENT_MINUS_PREINDUSTRIAL)
    return np.array(lo), np.array(hi)


class Controls:
    """z in [0, 1] <-> anomaly nodes and parameters."""

    def __init__(self, node_times, base):
        self.lo_a, self.hi_a = anomaly_bounds(node_times)
        self.n = len(node_times)
        self.keys = list(RANGES)
        self.lo = np.concatenate([self.lo_a, [RANGES[k][0] for k in self.keys]])
        self.hi = np.concatenate([self.hi_a, [RANGES[k][1] for k in self.keys]])
        self.base = base
        self.Tp0 = present_T(base["precip"], base["melt_factor"], base["T_s"], LAPSE)

    def to_z(self, anomaly, params):
        x = np.concatenate([anomaly, [params[k] for k in self.keys]])
        return (x - self.lo) / (self.hi - self.lo)

    def physical(self, z):
        lo = torch.tensor(self.lo, dtype=z.dtype, device=z.device)
        hi = torch.tensor(self.hi, dtype=z.dtype, device=z.device)
        x = lo + z * (hi - lo)
        p = {k: x[self.n + i] for i, k in enumerate(self.keys)}
        p["melt_factor"] = p.pop("ddf") * DAY
        return x[: self.n], p

    def present(self, p):
        """T_present re-fitted to the geodetic balance, with its first-order derivative (a tensor)."""
        v = {k: float(p[k]) for k in ("precip", "melt_factor", "T_s")}
        Tp = self.base["T_present"] + present_T(v["precip"], v["melt_factor"], v["T_s"], LAPSE) - self.Tp0
        out = torch.tensor(Tp, device=p["precip"].device)
        for k, h in (("precip", 0.01), ("melt_factor", 0.001), ("T_s", 0.05)):
            up, dn = dict(v), dict(v); up[k] += h; dn[k] -= h
            d = (present_T(up["precip"], up["melt_factor"], up["T_s"], LAPSE)
                 - present_T(dn["precip"], dn["melt_factor"], dn["T_s"], LAPSE)) / (2 * h)
            out = out + d * (p[k] - p[k].detach())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.03, help="Adam step, fraction of each control's range")
    ap.add_argument("--lr_min", type=float, default=0.005)
    ap.add_argument("--fd_step", type=float, default=0.03, help="finite-difference step, fraction of range")
    ap.add_argument("--out_dir", default="results/argentiere_lg_100m_joint")
    opts = ap.parse_args()
    out = opts.out_dir
    os.makedirs(out, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    s = json.load(open(f"{START}/history.json"))["settings"]
    base = CLIMATES["phys"]
    glacier, geo, obs, node_times, _ = build(
        "lg", s["dx"], chunk_steps=s["chunk_steps"], base=base, valley=True, max_velbar=s["max_velbar"],
        cfl=s["cfl"], adjoint_slope="godunov", emulator_grad=False)
    probe = install_adjoint_probe(s["adjoint_clip"])
    ctl = Controls(node_times, base)
    T0 = np.array(json.load(open(f"{START}/final_terms.json"))["T"])
    p0 = dict(precip=base["precip"], T_s=base["T_s"], ddf=base["melt_factor"] / DAY,
              arrhenius=base["arrhenius"], tau_ref=base["tau_ref"])
    z = torch.tensor(ctl.to_z(T0 - base["T_present"], p0), dtype=torch.float32, device=device).clamp(0, 1)
    z.requires_grad_(True)
    idx_fd = [ctl.n + ctl.keys.index(k) for k in FD_KEYS]

    def run(z, grad=True):
        """Loss at z; with grad, the backward pass for the nodes, P, T_s and DDF."""
        with torch.set_grad_enabled(grad):
            a, p = ctl.physical(z)
            Tp = ctl.present(p)
            glacier.precip_T_ref = Tp
            glacier.arrhenius = p["arrhenius"].detach() * torch.ones_like(glacier.Z_topo)
            glacier.tau_ref = p["tau_ref"].detach() * torch.ones_like(glacier.Z_topo)
            T = (Tp + a).view(-1, 1)
            loss, data, terms, _ = invert_observations(
                T, glacier, obs, precip_tensor=p["precip"], T_s=p["T_s"], melt_factor=p["melt_factor"],
                dt_snap=WINDOWS["lg"]["dt_snap"], reg_lambda=s["reg_lambda"], weighting="groups", reg_sum=True)
        return loss, data, terms, T, p, Tp

    def describe(T, p, Tp):
        return dict(T=T.detach().flatten().tolist(), T_present=float(Tp),
                    params=dict(precip=float(p["precip"]), T_s=float(p["T_s"]), ddf=float(p["melt_factor"]) / DAY,
                                arrhenius=float(p["arrhenius"]), tau_ref=float(p["tau_ref"])))

    opt = torch.optim.Adam([z], lr=opts.lr)
    hist = dict(loss=[], data=[], z=[], grad_z=[], seconds=[], terms=[], lr=[], state=[], adjoint=[])
    best = dict(loss=np.inf)
    lr_scale = 1.0
    for i in range(opts.iters):
        t0 = time.time()
        lr = lr_scale * (opts.lr_min + 0.5 * (opts.lr - opts.lr_min) * (1 + np.cos(np.pi * i / max(1, opts.iters - 1))))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()
        loss, data, terms, T, p, Tp = run(z)
        loss.backward()
        adj = adjoint_report(probe)
        L = float(loss)
        # A and tau_ref: one-sided finite differences, in the same units (d loss / d z)
        with torch.no_grad():
            for j in idx_fd:
                zh = z.detach().clone()
                h = opts.fd_step if zh[j] + opts.fd_step <= 1 else -opts.fd_step
                zh[j] += h
                Lh = float(run(zh, grad=False)[0])
                z.grad[j] = (Lh - L) / h
        g = z.grad.detach().clone()
        if not (np.isfinite(L) and torch.isfinite(g).all()):
            print(f"iter {i}: non-finite loss/gradient, back to the best iterate with half the step", flush=True)
            if not np.isfinite(best["loss"]):
                break
            lr_scale *= 0.5
            with torch.no_grad():
                z.copy_(torch.tensor(best["z"], device=device))
            opt = torch.optim.Adam([z], lr=lr * 0.5)
            continue
        st = describe(T, p, Tp)
        for k, v in (("loss", L), ("data", float(data)), ("z", z.detach().tolist()), ("grad_z", g.tolist()),
                     ("terms", {k: float(v) for k, v in terms.items()}), ("lr", lr), ("state", st), ("adjoint", adj)):
            hist[k].append(v)
        if L < best["loss"]:
            best = dict(loss=L, iter=i, z=z.detach().tolist(), **st)
        elif L > 1.05 * best["loss"]:
            lr_scale *= 0.5
            with torch.no_grad():
                z.copy_(torch.tensor(best["z"], device=device))
            opt = torch.optim.Adam([z], lr=lr * 0.5)
            hist["seconds"].append(time.time() - t0)
            print(f"iter {i:3d}: loss {L:.3f} > 1.05 x best {best['loss']:.3f}: back to iter {best['iter']}", flush=True)
            continue
        opt.step()
        with torch.no_grad():
            z.clamp_(0, 1)
        hist["seconds"].append(time.time() - t0)
        pp = st["params"]
        print(f"iter {i:3d}: loss {L:8.3f} data {float(data):8.3f} | P {pp['precip']:.3f} T_s {pp['T_s']:.2f} "
              f"DDF {pp['ddf']:.2f} A {pp['arrhenius']:.1f} tau {pp['tau_ref']:.4f} T_present {st['T_present']:+.2f} | "
              f"YD anomaly {np.mean(st['T'][4:10]) - st['T_present']:+.2f} | lr {lr:.4f} ({time.time() - t0:.0f} s)", flush=True)
        with open(f"{out}/history.json", "w") as f:
            json.dump(dict(settings=vars(opts), start=START, ranges=RANGES, node_times=node_times,
                           z_lo=ctl.lo.tolist(), z_hi=ctl.hi.tolist(), n_nodes=ctl.n, keys=ctl.keys,
                           history=hist, best=best), f, indent=1)

    # final forward run at the best iterate: fields and per-landform distances
    from core.moraine_obs import describe as describe_obs
    from tools.run_argentiere_inversion import termini
    zb = torch.tensor(best["z"], device=device)
    loss, data, terms, T, p, Tp = run(zb, grad=False)
    with torch.no_grad():
        every = [WINDOWS["lg"]["t_start"] + 25.0 * (k + 1) for k in range(int(4600 / 25))]
        snaps, H = glacier(precip_tensor=p["precip"], T_m_lowest=T, T_s=p["T_s"], melt_factor=p["melt_factor"],
                           smb_method="PDD", save_times=sorted(set(every) | {t for o in obs for t in o["times"]}))
    keep = sorted(snaps)
    np.savez_compressed(f"{out}/final_fields.npz", times=np.array(keep),
                        H=np.stack([snaps[t].cpu().numpy() for t in keep]))
    with open(f"{out}/final_terms.json", "w") as f:
        json.dump(dict(best, node_times=node_times, terms={k: float(v) for k, v in terms.items()},
                       describe=describe_obs(obs, snaps, glacier.Z_topo), termini=termini(geo, snaps)), f, indent=1)
    print(f"best iter {best['iter']}, loss {best['loss']:.3f}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()

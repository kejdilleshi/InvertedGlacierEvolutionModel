"""How much does each fixed parameter move the inverted Lateglacial temperature?

Starts from the 100 m Lateglacial inversion (``results/argentiere_lg_100m_phys``,
literature climate) and asks, one parameter at a time: if this parameter were
different, which temperature history would fit the landforms as well? A full
re-inversion per parameter costs ~17 h at 100 m, so the answer is restricted to a
uniform shift of the inverted history: the forward model is run with the
perturbed parameter at ``T* + dT`` for a few ``dT``, and a parabola through the
loss gives the best shift and the loss at that shift.

Mass-balance parameters (precipitation, degree-day factor, seasonality, lapse
rate) are calibrated jointly with the present-day temperature on the geodetic
2012-2021 balance (-0.66 m w.e./yr), so for those the present temperature is
re-fitted first (``recal``), as the calibration would have done. What carries
over to the paleoclimate is then the *anomaly* to the present, ``dT - dT_present``.
The paleo-precipitation factor scales the precipitation without recalibration
(the Lateglacial precipitation relative to the Clausius-Clapeyron scaling of
today's). Flow-law factor, sliding and speed cap do not enter the calibration.

The ``block_*`` cases shift only the nodes of one period with the base
parameters: the curvature of the loss there gives the data-only uncertainty of
that period (the loss is twice a negative log-likelihood, so sigma^2 = 2 / L'').

The guess for each case's shift is the mass-balance-equivalent shift: the dT that
keeps the glacier-wide balance of the inverted glacier (its Younger Dryas
geometry, from ``video_fields.npz``) unchanged.

    IGEM_DEVICE=cuda:1 python tools/sensitivity_argentiere.py --cases base P- P+ ...
    python tools/sensitivity_argentiere.py --list
"""

import argparse
import json
import os
import time

import netCDF4
import numpy as np
import torch

import core.climate as climate
from core.climate import compute_integral_positive_temperature, compute_negative_temperature_ratio
from core.moraine_obs import data_term, describe, evaluate
from core.utils import device
from tools.run_argentiere_inversion import CLIMATES, build

RUN = "results/argentiere_lg_100m_phys"
OUT = f"{RUN}/sensitivity"
DAY = 30.44 / 1000.0                  # melt factor per month <-> degree-day factor in mm/K/d
GEODETIC = -0.66                      # m w.e./yr, Pleiades 2012-2020 (Davaze et al. 2020)
LAPSE = 7.0                           # C/km, the model's
PERIODS = {                           # ka BP, for the block_* cases
    "allerod": (13.6, 12.9), "yd": (12.9, 11.7), "preboreal": (11.7, 10.0), "early_holocene": (10.0, 9.0)}

# name: (label, overrides, recalibrate the present temperature?)
CASES = {
    "base":     ("base parameters", {}, False),
    "P-":       ("precipitation 1.5 m/yr", dict(precip=1.5), True),
    "P+":       ("precipitation 2.1 m/yr", dict(precip=2.1), True),
    "DDF-":     ("degree-day factor 8 mm/K/d", dict(ddf=8.0), True),
    "DDF+":     ("degree-day factor 12 mm/K/d", dict(ddf=12.0), True),
    "Ts-":      ("seasonal half-range 7.9 C", dict(T_s=7.9), True),
    "Ts+":      ("seasonal half-range 10.9 C", dict(T_s=10.9), True),
    "lapse-":   ("lapse rate 6 C/km", dict(lapse=6.0), True),
    "lapse+":   ("lapse rate 8 C/km", dict(lapse=8.0), True),
    "Ppaleo-":  ("Lateglacial precipitation x0.75", dict(paleo_p=0.75), False),
    "Ppaleo+":  ("Lateglacial precipitation x1.25", dict(paleo_p=1.25), False),
    "A-":       ("flow-law factor 52 MPa^-3/yr", dict(arrhenius=52.0), False),
    "A+":       ("flow-law factor 117 MPa^-3/yr", dict(arrhenius=117.0), False),
    "tau-":     ("sliding tau_ref 0.15 MPa (more sliding)", dict(tau_ref=0.15), False),
    "cap+":     ("speed cap 1000 m/yr", dict(max_velbar=1000.0), False),
}
CASES.update({f"block_{p}": (f"base parameters, {a}-{b} ka only", {}, False)
              for p, (a, b) in PERIODS.items()})


def params(name, base):
    """Model parameters of a case: precip, melt_factor, T_s, lapse, T_present, ..."""
    _, ov, recal = CASES[name]
    p = dict(precip=base["precip"], melt_factor=base["melt_factor"], T_s=base["T_s"],
             lapse=LAPSE, T_present=base["T_present"], arrhenius=base["arrhenius"],
             tau_ref=base["tau_ref"], max_velbar=600.0, dT_present=0.0)
    for k, v in ov.items():
        if k == "ddf":
            p["melt_factor"] = v * DAY
        elif k == "paleo_p":
            p["precip"] = base["precip"] * v
        else:
            p[k] = v
    if recal:
        # same calibration as the base: the shift of the fitted present temperature
        # is added to the base value (whose fit tolerance this keeps)
        p["dT_present"] = (present_T(p["precip"], p["melt_factor"], p["T_s"], p["lapse"])
                           - present_T(base["precip"], base["melt_factor"], base["T_s"], LAPSE))
        p["T_present"] = base["T_present"] + p["dT_present"]
    return p


_ZS = None


def present_T(precip, melt_factor, T_s, lapse):
    """Temperature at 2000 m giving the geodetic 2012-2021 glacier-wide balance."""
    global _ZS
    if _ZS is None:
        nc = netCDF4.Dataset("data/Argentiere/argentiere_100m.nc")
        bed = np.asarray(nc["topg"][:])
        surf = bed + 0.5 * (np.asarray(nc["thk_2012"][:]) + np.asarray(nc["thk_2021"][:]))
        _ZS = surf[np.asarray(nc["argentiere_2012"][:]) > 0.5]
        nc.close()
    lo, hi = -6.0, 4.0
    for _ in range(50):
        T = 0.5 * (lo + hi)
        b = mean_balance(_ZS, T, precip, melt_factor, T_s, lapse)
        lo, hi = (lo, T) if b < GEODETIC else (T, hi)
    return 0.5 * (lo + hi)


def mean_balance(z, T, precip, melt_factor, T_s, lapse):
    """Mean PDD balance (m w.e./yr) of the surface cells ``z`` at ``T`` (C at 2000 m)."""
    Tm = torch.as_tensor(T, dtype=torch.float64) - lapse / 1000.0 * (
        torch.as_tensor(z, dtype=torch.float64) - 2000.0)
    Ts = torch.tensor(float(T_s), dtype=torch.float64)
    acc = precip * compute_negative_temperature_ratio(Tm, Ts)
    abl = melt_factor * compute_integral_positive_temperature(Tm, Ts)
    return float((acc - abl).mean())


def smb_equivalent_shift(p, base, T_best, node_times):
    """dT keeping the balance of the inverted Younger Dryas glacier unchanged."""
    f = np.load(f"{RUN}/video_fields.npz")
    nc = netCDF4.Dataset("data/Argentiere/argentiere_100m.nc")
    bed = np.asarray(nc["topg"][:]); nc.close()
    ka = (1950.0 - f["times"]) / 1000.0
    out = []
    for i in np.flatnonzero((ka <= 12.9) & (ka >= 11.7))[::4]:
        H = f["H"][i]
        z = (bed + H)[H > 1.0]
        T = float(np.interp(f["times"][i], node_times, T_best))

        def bal(T, q, dT=0.0):
            P = q["precip"] * (1 + base["precip_cc"]) ** (T + dT - q["T_present"])
            return mean_balance(z, T + dT, P, q["melt_factor"], q["T_s"], q["lapse"])

        b0 = bal(T, dict(precip=base["precip"], melt_factor=base["melt_factor"], T_s=base["T_s"],
                         lapse=LAPSE, T_present=base["T_present"]))
        lo, hi = -6.0, 6.0          # the balance falls with dT
        for _ in range(40):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if bal(T, p, m) > b0 else (lo, m)
        out.append(0.5 * (lo + hi))
    return float(np.mean(out))


def gradient_shift(p, base):
    """First-order dT from the two gradient runs (tools/gradient_sensitivity_argentiere.py)."""
    G = [json.load(open(f"{OUT}/gradients/u{u:+.2f}_full.json")) for u in (-0.2, 0.2)]
    L_uu = (G[1]["L_u"] - G[0]["L_u"]) / 0.4
    du = {k: -(G[1]["grad_params"][k] - G[0]["grad_params"][k]) / 0.4 / L_uu for k in G[0]["grad_params"]}
    dtheta = dict(precip=p["precip"] - base["precip"], melt_factor=p["melt_factor"] - base["melt_factor"],
                  T_s=p["T_s"] - base["T_s"], lapse=p["lapse"] - LAPSE, T_present=p["dT_present"],
                  arrhenius=p["arrhenius"] - base["arrhenius"], tau_ref=p["tau_ref"] - base["tau_ref"])
    return float(sum(du[k] * v for k, v in dtheta.items()))


def apply(glacier, p):
    climate.LAPSE_RATE = p["lapse"] / 1000.0
    glacier.precip_T_ref = p["T_present"]
    glacier.tau_ref = glacier._as_field(p["tau_ref"], "tau_ref")
    glacier.arrhenius = glacier._as_field(p["arrhenius"], "arrhenius")
    glacier.max_velbar = float(p["max_velbar"])


def loss_at(glacier, obs, T_nodes, p):
    """Loss of the inversion (grouped Huber data term + summed curvature prior)."""
    with torch.no_grad():
        times = sorted({t for o in obs for t in o["times"]})
        snaps, _ = glacier(precip_tensor=p["precip"], T_m_lowest=T_nodes, T_s=p["T_s"],
                           melt_factor=p["melt_factor"], smb_method="PDD", save_times=times)
        terms = evaluate(obs, snaps, glacier.Z_topo)
        data = float(data_term(obs, terms, "groups"))
        flat = T_nodes.flatten()
        reg = float(((flat[2:] - 2 * flat[1:-1] + flat[:-2]) ** 2).sum())
        return dict(loss=data + reg, data=data, reg=reg,
                    terms={k: float(v) for k, v in terms.items()},
                    describe=describe(obs, snaps, glacier.Z_topo))


def parabola(rows):
    """Vertex (dT, loss) and curvature of the parabola through the three lowest-loss-centred rows."""
    rows = sorted(rows, key=lambda r: r["dT"])
    i = int(np.argmin([r["loss"] for r in rows]))
    i = min(max(i, 1), len(rows) - 2)
    x = np.array([r["dT"] for r in rows[i - 1: i + 2]]); y = np.array([r["loss"] for r in rows[i - 1: i + 2]])
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return dict(dT=float(rows[int(np.argmin([r["loss"] for r in rows]))]["dT"]),
                    loss=float(min(y)), curv=float(2 * a))
    xv = -b / (2 * a)
    return dict(dT=float(xv), loss=float(c - b * b / (4 * a)), curv=float(2 * a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--step", type=float, default=0.4, help="spacing of the dT points (C)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--gradient_guess", action="store_true",
                    help="centre the dT points on the gradient estimate (needs the gradient runs)")
    opts = ap.parse_args()
    if opts.list:
        for k, (lab, ov, rec) in CASES.items():
            print(f"{k:22s} {lab:45s} {'recalibrated' if rec else ''}")
        return
    os.makedirs(OUT, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with open(f"{RUN}/history.json") as f:
        s = json.load(f)["settings"]
    base = CLIMATES["phys"]
    best = json.load(open(f"{RUN}/final_terms.json"))
    T_best = np.array(best["T"]); node_times = best["node_times"]
    glacier, geo, obs, nt, _ = build("lg", s["dx"], chunk_steps=s["chunk_steps"], base=base,
                                     valley=True, max_velbar=s["max_velbar"], cfl=s["cfl"])
    assert np.allclose(nt, node_times)
    ka = (1950.0 - np.array(node_times)) / 1000.0

    for name in opts.cases:
        path = f"{OUT}/{name}.json"
        rec = json.load(open(path)) if os.path.exists(path) else None
        p = params(name, base)
        if name.startswith("block_"):
            a, b = PERIODS[name[6:]]
            mask = ((ka <= a + 1e-6) & (ka >= b - 1e-6)).astype(float)
            guess = 0.0
        else:
            mask = np.ones_like(T_best)
            guess = round(gradient_shift(p, base), 2) if opts.gradient_guess else \
                0.0 if name == "base" or not any(
                k in CASES[name][1] for k in ("precip", "ddf", "T_s", "lapse", "paleo_p")) \
                else round(smb_equivalent_shift(p, base, T_best, node_times), 2)
        if rec is None:
            rec = dict(name=name, label=CASES[name][0], params=p, guess=guess,
                       mask=mask.tolist(), rows=[])
        if name.startswith("block_") and os.path.exists(f"{OUT}/base.json"):
            # dT = 0 of a block is the base run itself
            rec["rows"] += [r for r in json.load(open(f"{OUT}/base.json"))["rows"]
                            if abs(r["dT"]) < 1e-6 and not any(abs(q["dT"]) < 1e-6 for q in rec["rows"])]
        apply(glacier, p)
        todo = [guess, guess - opts.step, guess + opts.step]
        for _ in range(4):      # extend while the minimum sits at an end
            for dT in todo:
                if any(abs(r["dT"] - dT) < 1e-6 for r in rec["rows"]):
                    continue
                t0 = time.time()
                Tn = torch.tensor(T_best + dT * mask, dtype=torch.float32, device=device).view(-1, 1)
                r = dict(dT=round(float(dT), 4), **loss_at(glacier, obs, Tn, p))
                r["seconds"] = time.time() - t0
                rec["rows"].append(r)
                print(f"{name:22s} dT {dT:+.2f}: loss {r['loss']:8.3f} data {r['data']:8.3f} "
                      f"({r['seconds']:.0f} s)", flush=True)
                with open(path, "w") as f:
                    json.dump(rec, f, indent=1)
            xs = sorted(r["dT"] for r in rec["rows"])
            i = int(np.argmin([r["loss"] for r in sorted(rec["rows"], key=lambda r: r["dT"])]))
            if 0 < i < len(xs) - 1:
                break
            todo = [xs[0] - opts.step] if i == 0 else [xs[-1] + opts.step]
        rec["fit"] = parabola(rec["rows"])
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"{name:22s} best dT {rec['fit']['dT']:+.2f} C, loss {rec['fit']['loss']:.3f} "
              f"(guess {guess:+.2f}, dT_present {p['dT_present']:+.2f})", flush=True)
    climate.LAPSE_RATE = LAPSE / 1000.0


if __name__ == "__main__":
    main()

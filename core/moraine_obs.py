"""Observation operators for dated glacial landforms, and the inversion objective.

Each observation is a dict (built by :func:`data.loader.load_argentiere` from the
table that ``tools/prepare_argentiere_data.py`` writes) with at least

``id``, ``kind``, ``t0``, ``t1`` (calendar years CE), ``weight``, ``cells``
(boolean grid of the cells it is scored on), and kind-specific fields.

Kinds and what each one asks of the simulated thickness ``H`` (bed ``topo``):

``lateral``      ice surface at ``cells`` equals ``z`` (grid), sigma ``sigma``,
                 at *some* time in [t0, t1] -- a soft minimum over snapshots,
                 because the moraine was built at one moment inside its age
                 uncertainty, not at the nominal age.
``frontal``      ice on ``cells`` (upstream of the ridge), none on ``outside``,
                 at some time in [t0, t1].
``icefree``      no ice above the sample elevation ``z`` at *every* time from
                 t0 (the young end of the exposure age) to the end of the run.
``icefree_any``  the same, at some time in [t0, t1] (subfossil wood: the tree
                 grew at that spot during the calibrated range).
``covered_any``  ice surface at or above ``z`` at some time in [t0, t1] (the
                 wood was overridden by the next advance).
``thickness``    mean squared thickness misfit to ``H_obs`` on ``cells`` at the
                 single time ``t0`` (2012, 2021, LIA surface).
``extent``       soft-indicator extent misfit to ``H_obs`` on ``cells`` at ``t0``.

All misfits are normalized (squared residual over sigma squared, or an extent
indicator in [0, 1]), so the weights are the only thing that sets their
relative importance.

Valley-referenced kinds
-----------------------
The kinds above score the ice *at the landform's own cells*. That fails in
two ways in steep mountain valleys: a glacier that stops kilometres short of
a moraine scores a constant misfit with zero gradient (nothing there, nothing
to move), and ice draped on the valley side next to a lateral moraine or a
bedrock sample -- not the glacier that built it -- is scored as if it were.
The kinds below refer every landform to its source valley instead
(``core/valley.py``): a flowline ``path`` (flat cell indices down the valley
axis) with arc length ``s`` (m), and for every landform cell its
cross-section ``axis`` on that line. Two quantities of the simulated glacier
are measured there, both differentiable and both defined however far the
glacier is from the landform:

* the terminus position ``L`` (m along the flowline): a soft maximum of the
  arc length over flowline cells weighted by a soft ice indicator, see
  :func:`terminus_position` -- a signed distance along the valley rather than
  an overlap;
* the ice surface on the valley axis at each landform cell's cross-section.

``terminus``     frontal moraine or mapped terminus: ``L = s_obs`` (sigma_L),
                 at some time in [t0, t1].
``lateral_v``    lateral moraine: the glacier reaches its cross-section
                 (``L >= s_req``) and its axis surface stands at the crest
                 elevation plus ``delta`` (sigma_z), at some time in [t0, t1].
``icefree_v``    bedrock/erratic: axis surface below the sample plus
                 ``delta`` at every time from t0 on.
``icefree_any_v`` the same at some time in [t0, t1] (subfossil wood).
``covered_any_v`` the glacier reached the wood's cross-section and its axis
                 surface stood above the wood plus ``delta``, at some time in
                 [t0, t1] (without the reach term a glacier that stops short
                 of the site would again feel nothing).
``preserve``     a moraine survived, so no later advance overrode it: after
                 t1, ``L <= s_obs`` (frontal) or axis surface <= crest +
                 ``delta`` (lateral), at every time.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

import torch

SOFTMIN_TAU = 1.0       # temperature of the soft minimum over snapshots
EXTENT_THRESH = 1.0     # m of ice that counts as "ice"
EXTENT_WIDTH = 5.0      # m, width of the soft extent indicator
TERMINUS_TAU = 100.0    # m, smoothness of the soft maximum that locates the terminus
ICE_SCALE = 30.0        # m, thickness at which the soft ice indicator reaches 1 - 1/e
GAP_THICKNESS = 5.0     # m, ice thinner than this only partly connects the snout to the glacier
HUBER_C = 3.0           # residuals beyond this many sigma count linearly, not squared


def soft_min(values: torch.Tensor, tau: float = SOFTMIN_TAU) -> torch.Tensor:
    """Smooth minimum: ``-tau * log(mean(exp(-v / tau)))``.

    Equals the plain value for a single snapshot and tends to the minimum for
    small ``tau``. Using the mean (not the sum) keeps it from going below the
    minimum merely because there are many snapshots.
    """
    return -tau * (torch.logsumexp(-values / tau, dim=0) - math.log(values.numel()))


def _extent(H: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid((H - EXTENT_THRESH) / EXTENT_WIDTH)


def terminus_position(H: torch.Tensor, path: torch.Tensor, s: torch.Tensor,
                      tau: float = TERMINUS_TAU, scale: float = ICE_SCALE,
                      gap: float = GAP_THICKNESS, volume_scale=None) -> torch.Tensor:
    """Soft position (m along a flowline) of the snout of the ice that starts at ``path[0]``.

    ``tau * logsumexp(s / tau + log m(H) + sum of log g(H) upstream)`` over the
    flowline cells: a soft maximum of the arc length, weighted by the soft ice
    indicator ``m(H) = 1 - exp(-H / scale)`` and by contiguity -- every cell
    between ``path[0]`` (a point the source glacier always covers) and the cell
    must hold ice, with ``g`` a smoothstep that is 1 from ``gap`` metres up and
    0 without ice. So another glacier's tongue lying further down the same
    valley floor never counts as this one's.

    The position is within a fraction of a cell of the snout, moves
    continuously as the snout cell fills or empties, and is defined however far
    the glacier is from any landform. ``path[0]`` always counts, so it is
    defined without any ice.

    Its gradient alone is far too small: on a grid a glacier advances by
    filling the next cell, an event the adjoint cannot see, so only the
    snout cell's own thickness reaches the derivative (about 1/50 of the
    finite-difference sensitivity on the Argentière). With ``volume_scale``
    (m) the value is unchanged but the gradient is taken from the ice volume
    per unit width along the connected glacier, divided by ``volume_scale``:
    the thickening that feeds an advance, scaled to dV/dL of the glacier
    (a straight-through surrogate gradient; -675 against -968 m/C by finite
    differences there, with the upwind adjoint of
    ``core.forward_schemes.igm_emulator_step.use_adjoint_transport``).
    """
    h = H.reshape(-1)[path]
    has_ice = h > 0
    safe = torch.where(has_ice, h, torch.ones_like(h))         # no NaN gradients
    logm = torch.where(has_ice, torch.log(-torch.expm1(-safe / scale)),
                       torch.full_like(h, -math.inf))
    u = (safe / gap).clamp(max=1.0)
    logg = torch.where(has_ice, torch.log(u * u * (3.0 - 2.0 * u)),
                       torch.full_like(h, -math.inf))
    upstream = torch.cat([torch.zeros_like(h[:1]), torch.cumsum(logg, dim=0)[:-1]])
    z = torch.cat([s[:1] / tau, s / tau + logm + upstream])
    L = tau * torch.logsumexp(z, dim=0)
    if volume_scale is not None:
        ds = torch.cat([s[1:] - s[:-1], s[-1:] - s[-2:-1]])
        connected = torch.exp(upstream + logg).detach()          # 1 along the glacier, 0 beyond a gap
        V = (torch.relu(h) * ds * connected).sum()
        L = L + (V - V.detach()) / volume_scale
    return L


def axis_surface(H: torch.Tensor, topo: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Ice surface (bed where there is no ice) at flat cell indices on a flowline."""
    return topo.reshape(-1)[cells] + torch.relu(H.reshape(-1)[cells])


def huber(v: torch.Tensor, c: float = HUBER_C) -> torch.Tensor:
    """Robust version of a squared normalized residual ``v = r**2``.

    ``v`` up to ``c**2``, then ``2 c |r| - c**2``: one constraint the model
    cannot fit pulls on the solution like ``|r|``, not like ``r**2``.
    """
    v = v.clamp(min=0.0)               # a soft minimum can sit a rounding error below 0
    r = torch.sqrt(v + 1e-12)
    return torch.where(r <= c, v, 2.0 * c * r - c * c)


def obs_times(o: dict, t_start: float, t_end: float, dt_snap: float) -> List[float]:
    """Snapshot times at which observation ``o`` is scored inside a run window."""
    if o["kind"] in ("thickness", "extent"):
        return [float(o["t0"])] if t_start < o["t0"] <= t_end else []
    if o["kind"] == "terminus" and o["t0"] == o["t1"]:
        return [float(o["t0"])] if t_start < o["t0"] <= t_end else []
    lo = max(float(o["t0"]), t_start + dt_snap)
    hi = min(float(o["t1"]), t_end)
    if hi < lo:
        # Window shorter than the spacing, or partly outside the run: score the
        # one admissible instant, if any.
        mid = 0.5 * (float(o["t0"]) + min(float(o["t1"]), t_end))
        return [mid] if t_start < mid <= t_end else []
    if hi - lo < dt_snap:
        return [0.5 * (lo + hi)]
    n =max(1, int(math.floor((hi - lo) / dt_snap)) + 1)
    return [lo + i * dt_snap for i in range(n)]


def select(observations: Iterable[dict], t_start: float, t_end: float,
           dt_snap: float) -> List[dict]:
    """Observations that have at least one snapshot inside the run window."""
    out = []
    for o in observations:
        times = obs_times(o, t_start, t_end, dt_snap)
        if times and o.get("use", True):
            out.append(dict(o, times=times))
    return out


def misfit_at(o: dict, H: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
    """Normalized misfit of one observation for one thickness field."""
    c = o.get("cells")
    kind = o["kind"]
    if kind == "lateral":
        r = (topo + H)[c] - o["z"][c]
        return ((r / o["sigma"]) ** 2).mean()
    if kind == "frontal":
        e = _extent(H)
        m = ((1.0 - e[c]) ** 2).mean()
        if o.get("outside") is not None and o["outside"].any():
            m = m + (e[o["outside"]] ** 2).mean()
        return m
    if kind in ("icefree", "icefree_any"):
        # ice thickness above the sample: surface - max(sample, bed)
        r = torch.relu(H[c] - torch.relu(o["z_pt"] - topo[c]))
        return ((r / o["sigma"]) ** 2).mean()
    if kind == "covered_any":
        r = torch.relu(o["z_pt"] - (topo + H)[c])
        return ((r / o["sigma"]) ** 2).mean()
    if kind == "thickness":
        r = H[c] - o["H_obs"][c]
        return ((r / o["sigma"]) ** 2).mean()
    if kind == "extent":
        return ((_extent(H[c]) - _extent(o["H_obs"][c])) ** 2).mean()
    return _misfit_valley(o, H, topo)


def _misfit_valley(o: dict, H: torch.Tensor, topo: torch.Tensor) -> torch.Tensor:
    kind = o["kind"]
    if kind == "terminus" or "s_req" in o or (kind == "preserve" and "s_obs" in o):
        L = terminus_position(H, o["path"], o["s"], volume_scale=o.get("volume_scale"))
    if kind == "terminus":
        return ((L - o["s_obs"]) / o["sigma_L"]) ** 2
    if kind == "preserve":
        if "s_obs" in o:
            return (torch.relu(L - o["s_obs"]) / o["sigma_L"]) ** 2
        r = torch.relu(axis_surface(H, topo, o["axis"]) - o["z"] - o["delta"])
        return ((r / o["sigma_z"]) ** 2).mean()
    zs = axis_surface(H, topo, o["axis"])
    if kind == "lateral_v":
        reach = torch.relu(o["s_req"] - L) / o["sigma_L"]
        height = (zs - o["z"] - o["delta"]) / o["sigma_z"]
        return reach ** 2 + (height ** 2).mean()
    if kind in ("icefree_v", "icefree_any_v"):
        return ((torch.relu(zs - o["z"] - o["delta"]) / o["sigma_z"]) ** 2).mean()
    if kind == "covered_any_v":
        # overridden by an advance: the glacier reached the site and stood above it
        reach = torch.relu(o["s_req"] - L) / o["sigma_L"]
        return reach ** 2 + ((torch.relu(o["z"] + o["delta"] - zs) / o["sigma_z"]) ** 2).mean()
    raise ValueError(f"unknown observation kind {kind!r}")


def combine(o: dict, per_time: torch.Tensor) -> torch.Tensor:
    """Aggregate a misfit over its snapshots: all times, or the best time."""
    if o["kind"] in ("icefree", "icefree_v", "preserve"):
        return per_time.mean()
    return soft_min(per_time)


def evaluate(observations: List[dict], snaps: Dict[float, torch.Tensor],
             topo: torch.Tensor):
    """Per-observation misfit (dict id -> scalar tensor) from saved snapshots."""
    terms = {}
    for o in observations:
        vals = torch.stack([misfit_at(o, snaps[t], topo) for t in o["times"]])
        terms[o["id"]] = combine(o, vals)
    return terms


def describe(observations: List[dict], snaps: Dict[float, torch.Tensor],
             topo: torch.Tensor) -> Dict[str, dict]:
    """What the valley-referenced misfits mean in metres, per observation.

    At the snapshot that scores best (worst, for the kinds that must hold at
    every time): the terminus position ``L_m`` and its offset from the
    landform (``dL_m``, positive = beyond it), the shortfall in reaching a
    lateral moraine's cross-section (``reach_deficit_m``), and the mean axis
    surface minus the landform elevation and ``delta`` (``dz_m``).
    """
    out = {}
    with torch.no_grad():
        for o in observations:
            if "path" not in o and "axis" not in o:
                continue
            vals = [float(misfit_at(o, snaps[t], topo)) for t in o["times"]]
            pick = max if o["kind"] in ("icefree_v", "preserve") else min
            t_pick = o["times"][vals.index(pick(vals))]
            H = snaps[t_pick]
            d = dict(t=t_pick, misfit=pick(vals))
            if "path" in o:
                L = float(terminus_position(H, o["path"], o["s"]))
                d["L_m"] = L
                if "s_obs" in o:
                    d["dL_m"] = L - o["s_obs"]
                if "s_req" in o:
                    d["reach_deficit_m"] = max(0.0, o["s_req"] - L)
            if "axis" in o and "z" in o:
                zs = axis_surface(H, topo, o["axis"])
                d["dz_m"] = float((zs - o["z"] - o["delta"]).mean())
            out[o["id"]] = d
    return out


def data_term(observations: List[dict], terms: Dict[str, torch.Tensor],
              weighting: str = "mean") -> torch.Tensor:
    """Combine per-observation misfits into one number.

    ``"mean"``    weighted mean over observations (the original scheme). Every
                  satisfied one-sided constraint adds a zero and dilutes the rest.
    ``"groups"``  each landform counts once: the misfits of the observations
                  sharing a ``group`` are averaged (weighted), passed through
                  :func:`huber`, multiplied by the group's weight (the largest
                  of its members') and *summed* over groups -- a sum, so a
                  satisfied constraint adds nothing rather than diluting.
    """
    if weighting == "mean":
        w = torch.tensor([o["weight"] for o in observations], device=terms[observations[0]["id"]].device)
        v = torch.stack([terms[o["id"]] for o in observations])
        return (w * v).sum() / w.sum()
    if weighting != "groups":
        raise ValueError(f"unknown weighting {weighting!r}")
    groups: Dict[str, List[dict]] = {}
    for o in observations:
        groups.setdefault(o.get("group_id", o["id"]), []).append(o)
    total = 0.0
    for members in groups.values():
        w = [o["weight"] for o in members]
        v = sum(wi * terms[o["id"]] for wi, o in zip(w, members)) / sum(w)
        total = total + max(w) * huber(v)
    return total


def invert_observations(T_nodes, glacier, observations, *, precip_tensor, T_s,
                        melt_factor, dt_snap, reg_lambda=1.0, anchor=None,
                        smb_method="PDD", weighting="mean", reg_sum=False,
                        reg_slope=0.0):
    """Forward run + weighted misfit of every observation inside the window.

    Parameters
    ----------
    T_nodes : Tensor ``[n_nodes, 1]``
        Temperature at ``glacier.z_ref`` at nodes every ``glacier.vis_freq``
        years from ``glacier.t_start``; linearly interpolated in between.
    observations : list of dict
        Already passed through :func:`select` for this window.
    reg_lambda : float
        Weight of the smoothness prior, the mean squared *second* difference of
        the nodes (deg C^2): penalizes curvature, not trends.
    anchor : (time, value, sigma), optional
        A temperature known independently at one time (e.g. 2012-2021 from
        the stake calibration), added as ((T(time) - value) / sigma)^2.
    weighting : {"mean", "groups"}
        How the observations are combined, see :func:`data_term`.
    reg_sum : bool
        Sum (not mean) the squared second differences, so the prior keeps its
        strength relative to a summed (``"groups"``) data term.
    reg_slope : float
        Weight of the summed squared *first* differences of the nodes. The
        second-difference prior lets a node without data drift: Adam moves it
        by the learning rate every iteration as long as the (tiny) gradient
        keeps its sign, and a trend costs nothing. This penalizes the trend.

    Returns
    -------
    loss, data, terms, snaps
        ``terms`` maps each observation id to its (unweighted) misfit.
    """
    times = sorted({t for o in observations for t in o["times"]})
    snaps, H_end = glacier(precip_tensor=precip_tensor, T_m_lowest=T_nodes,
                           T_s=T_s, melt_factor=melt_factor,
                           smb_method=smb_method, save_times=times)
    missing = [t for t in times if snaps.get(t) is None]
    if missing:
        raise RuntimeError(f"no snapshot saved for {missing[:5]} ...")
    terms = evaluate(observations, snaps, glacier.Z_topo)
    data = data_term(observations, terms, weighting)

    flat = T_nodes.flatten()
    curv = (flat[2:] - 2 * flat[1:-1] + flat[:-2]) ** 2
    reg = ((curv.sum() if reg_sum else curv.mean())
           if flat.numel() > 2 else torch.zeros_like(data))
    loss = data + reg_lambda * reg
    if reg_slope and flat.numel() > 1:
        loss = loss + reg_slope * ((flat[1:] - flat[:-1]) ** 2).sum()
    if anchor is not None:
        t_a, val, sig = anchor
        T_a = glacier._control_at(T_nodes, torch.tensor(float(t_a), device=T_nodes.device))
        loss = loss + ((T_a - val) / sig) ** 2
    return loss, data, terms, snaps

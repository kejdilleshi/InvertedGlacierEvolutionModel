import torch
import netCDF4
from core.utils import device

def load_geology(path_nc):
    nc = netCDF4.Dataset(path_nc)
    topo = torch.tensor(nc.variables['topg'][:], device=device)
    # thk1880 = torch.tensor(nc.variables['thk_1880'][:], device=device)
    thk1880 = torch.tensor(nc.variables['surf_1999'][:], device=device) -topo
    mask = torch.tensor(nc.variables['icemask'][:], device=device)
    # mask = torch.ones(topo.shape, device=device)
    nc.close()
    return topo,thk1880, mask

def load_daily_data(file_path, accumulate=False, device='cpu'):
    """
    Parse a whitespace-delimited file with columns:
        year  jd  hour  temp  prec
    (first two lines are headers), and pack into a tensor of shape:
        (num_years, 366, 2), where [:, :, 0] = temp, [:, :, 1] = prec

    Parameters
    ----------
    file_path : str
        Path to the input file (e.g., "temp_prec.dat").
    accumulate : bool, optional
        If multiple rows map to the same (year, day), whether to sum them (True)
        or let the last occurrence overwrite (False). Default False.
    device : str or torch.device, optional
        Device to place the resulting tensors on.

    Returns
    -------
    data : torch.FloatTensor
        Shape (num_years, 366, 2): temperature and precipitation grids.
    years : torch.LongTensor
        Shape (num_years,): sorted unique calendar years aligned with data[year_idx].
    """
    # --- Read & parse file without NumPy/pandas ---
    years_f, jds_f, temps, precs = [], [], [], []

    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()

    # Skip first 2 header rows
    for line in lines[2:]:
        if not line.strip():
            continue
        cols = line.split()
        if len(cols) < 5:
            continue
        y, jd, hr, te, pr = cols[:5]
        years_f.append(float(y))
        jds_f.append(float(jd))
        temps.append(float(te))
        precs.append(float(pr))

    if not years_f:
        # No data rows parsed; return empty tensors
        empty = torch.zeros((0, 366, 2), dtype=torch.float32, device=device)
        return empty, torch.zeros((0,), dtype=torch.int64, device=device)

    # --- Convert to tensors ---
    year = torch.tensor(years_f, dtype=torch.float32, device=device)
    jd   = torch.tensor(jds_f,   dtype=torch.float32, device=device)
    temp = torch.tensor(temps,   dtype=torch.float32, device=device)
    prec = torch.tensor(precs,   dtype=torch.float32, device=device)

    # Cast year & jd to int
    year_i = year.to(torch.int64)
    jd_i   = jd.to(torch.int64)

    # Optional: sort by (year, jd) for tidiness
    sort_keys = year_i * 1000 + jd_i
    order = torch.argsort(sort_keys)
    year_i = year_i[order]
    jd_i   = jd_i[order]
    temp   = temp[order]
    prec   = prec[order]

    # Unique sorted years and mapping (yidx)
    years = torch.unique(year_i, sorted=True)
    yidx  = torch.searchsorted(years, year_i)

    # Day index: clamp to 1..366, then make 0-based
    didx = torch.clamp(jd_i, 1, 366) - 1

    num_years = years.numel()

    # --- Scatter into grids ---
    temp_grid = torch.zeros((num_years, 366), dtype=torch.float32, device=device)
    prec_grid = torch.zeros((num_years, 366), dtype=torch.float32, device=device)

    # If you have multiple rows per (year, day) and want to SUM them, set accumulate=True
    temp_grid.index_put_((yidx, didx), temp, accumulate=accumulate)
    prec_grid.index_put_((yidx, didx), prec, accumulate=accumulate)

    # Stack to (num_years, 366, 2)
    data = torch.stack((temp_grid, prec_grid), dim=-1)

    return data, years



def load_yd_aletsch(path_nc="data/YD-Aletsch/aletsch_yd_100m.nc",
                    mask="catchment", thk="thk_1880"):
    """Load the prepared YD-Aletsch geometry and the dated moraine targets.

    Written by ``tools/prepare_yd_data.py``, which joins the 30 m landform masks
    in ``data/YD-Aletsch/`` to the 100 m model grid of ``data/geology.nc``.

    Parameters
    ----------
    mask : {'catchment', 'icemask'}
        Where ice is allowed to accumulate.  ``catchment`` is the Massa basin
        (227 km2) and is the one to use when the glacier is larger than today;
        ``icemask`` is the modern/LIA mask carried over from ``geology.nc``.
    thk : {'thk_1880', 'thk'} or None
        Initial ice thickness.  ``None`` starts from an ice-free bed.

    Returns
    -------
    topo, H_init, ice_mask : Tensor
        The usual ``load_geology`` triple, on ``core.utils.device``.
    moraines : dict
        ``name -> {'mask', 'inboard', 'z_target', 'z_cell', 'z_crest', 'age'}``.

        The data term is ``surface[inboard] - z_target[inboard]``: ``inboard``
        holds the moraine cells stepped a few cells toward the ice, and
        ``z_target`` the elevation the simulated surface must reach there.
        Sampling at the moraine cells themselves (``mask`` / ``z_cell``) is
        degenerate -- an ice-free cell has surface == bed == moraine elevation
        and fits perfectly -- so score on ``inboard``.  ``z_crest`` is the 3x3
        DEM maximum, biased about 80 m high and kept only for reference.
        ``age`` is the ``(min, max)`` exposure-age bounds in years BP.

        The file's ``obs_operator_sigma_m`` attribute gives the spread of this
        operator measured against the 1880 surface at the LIA laterals.
    """
    nc = netCDF4.Dataset(path_nc)
    topo = torch.tensor(nc.variables["topg"][:], device=device)
    ice_mask = torch.tensor(nc.variables[mask][:], device=device)
    H_init = (torch.zeros_like(topo) if thk is None
              else torch.tensor(nc.variables[thk][:], device=device))

    moraines = {}
    for name in ("YD2", "YD1", "LIA_W", "LIA_E"):
        v = nc.variables[f"moraine_{name}"]
        age = ((float(v.age_min_yr_BP), float(v.age_max_yr_BP))
               if hasattr(v, "age_min_yr_BP") else None)
        moraines[name] = {
            "mask": torch.tensor(v[:], device=device) > 0.5,
            "inboard": torch.tensor(nc.variables[f"inboard_{name}"][:],
                                    device=device) > 0.5,
            "z_target": torch.tensor(nc.variables[f"z_target_{name}"][:],
                                     device=device),
            "z_cell": torch.tensor(nc.variables[f"z_cell_{name}"][:],
                                   device=device),
            "z_crest": torch.tensor(nc.variables[f"z_crest_{name}"][:],
                                    device=device),
            "age": age,
        }
    sigma = float(nc.obs_operator_sigma_m)
    nc.close()
    for m in moraines.values():
        m["sigma"] = sigma
    return topo, H_init, ice_mask, moraines


def analysis_mask(topg, elev_cut=2900.0, width=50.0):
    """Loss weights in [0, 1]: exactly 0 from ``elev_cut`` up, smooth below it.

    Ice above the equilibrium line for every temperature the inversion could
    try (headwalls fed by avalanches and snow slopes) cannot be changed by the
    control, so it is removed from the misfit.  The cut is a sigmoid rescaled to
    start at zero: ``tanh((elev_cut - z) / (2 width))``, clamped at 0, which is
    ``2 * sigmoid - 1``.  It is 0 at ``elev_cut``, 0.46 at one ``width`` below,
    0.96 at four, and continuous throughout.

    Parameters
    ----------
    topg : Tensor
        Bed elevation (m).
    elev_cut : float
        Elevation (m) at and above which the weight is 0.
    width : float
        Metres over which the weight rises toward 1.
    """
    return torch.tanh((elev_cut - topg) / (2.0 * width)).clamp(min=0.0)


# Valley-referenced scoring (``valley=True``, see core/valley.py and the
# valley kinds in core/moraine_obs.py). Uncertainties in metres:
VALLEY = dict(
    sigma_L=300.0,          # dated frontal moraine vs the terminus that built it
    sigma_L_mapped=200.0,   # mapped terminus (LIA, 2012, 2021 outlines): one cell
    sigma_z=60.0,           # axis surface vs a margin elevation: the scatter of
                            # (axis minus margin) along the LIA Argentière, 61 m
    delta=10.0,             # median (axis minus margin) surface on the LIA Argentière
    preserve_weight=0.5,    # a moraine's survival: partly redundant with its own fit
)
# Gradient scale of the terminus (m), dV/dL of each glacier: ice volume per unit
# width along its flowline against terminus position, across the constant-T
# scans of both windows (results/argentiere_*_200m_valley_scan_*). See
# core.moraine_obs.terminus_position: without it the adjoint sees only the
# snout cell and underestimates the terminus sensitivity about 50-fold.
VOLUME_SCALE = {"Argentiere": 690.0, "Mer de Glace": 530.0, "Tour": 230.0, "Talefre": 500.0}
# flowline -> (LIA_ID whose valley it climbs into, landform site names on it)
ARGENTIERE_VALLEYS = {
    "Argentiere": (4053, ("Lognan", "La Joux", "Cheserys", "Remuaz")),
    "Talefre": (967, ("Jardin de Talefre",)),
    "Mer de Glace": (967, ("Mer de Glace", "Mont Blanc Tête de Trélaporte",
                           "Aiguille du Moine")),
}
TOUR_LIA_ID = 4056


def load_argentiere(dx=200, mask="catchment_lateglacial", thk="thk_lia",
                    fields=True, morphostrat=True, valley=False):
    """Load the prepared Argentière / upper-Arve grid and every dated constraint.

    Written by ``tools/prepare_argentiere_data.py``; see that script and
    ``data/Argentiere/README.md`` for where each piece comes from, and
    ``core/moraine_obs.py`` for what each observation kind asks of the model.

    Parameters
    ----------
    dx : {100, 200}
        Grid spacing of the prepared file to read.
    mask : str
        Where ice may accumulate: ``catchment_lateglacial`` (upper Arve, for
        glaciers larger than the LIA) or ``catchment_holocene``.
    thk : str or None
        Initial thickness: ``thk_lia``, ``thk_2012``, ``thk_2021`` or ``None``.
    fields : bool
        Add the 1850 (LIA), 2012 and 2021 thickness/extent observations.
    morphostrat : bool
        Keep the undated ice margins assigned to the YD/early-Holocene by
        morphostratigraphy (weight 0.5) -- set False to use dated ones only.
    valley : bool
        Score every landform against its source valley's flowline instead of
        at its own cells (see :func:`_valley_observations`); adds the mapped
        LIA, 2012 and 2021 termini to the fields. ``geo["flowlines"]`` then
        holds the flowlines.

    Returns
    -------
    geo : dict
        ``topg``, ``H_init``, ``ice_mask`` (tensors on ``core.utils.device``)
        plus ``x``, ``y`` (numpy) and every 2D field of the file as numpy.
    observations : list of dict
        One dict per observation, ready for ``core.moraine_obs.select``.
    """
    import numpy as np
    import pandas as pd

    path = f"data/Argentiere/argentiere_{int(dx)}m.nc"
    nc = netCDF4.Dataset(path)
    t = lambda a: torch.tensor(np.nan_to_num(np.asarray(a, dtype=np.float32)),
                               device=device)
    geo = {name: np.asarray(nc[name][:]) for name in nc.variables
           if nc[name].ndim == 2 and not name.startswith(("obs", "line_"))}
    geo["x"], geo["y"] = np.asarray(nc["x"][:]), np.asarray(nc["y"][:])
    geo["topg_t"] = t(geo["topg"])
    geo["ice_mask"] = t(geo[mask])
    geo["H_init"] = (torch.zeros_like(geo["topg_t"]) if thk is None
                     else t(geo[thk]) * geo["ice_mask"])
    geo["lateral_sigma_m"] = float(nc.lateral_sigma_m)

    df = pd.read_csv(f"data/Argentiere/argentiere_observations_{int(dx)}m.csv")
    df = df[df.in_catchment]
    if not morphostrat:
        df = df[df.group != "morphostrat"]
    observations = []
    for _, r in df.iterrows():
        o = dict(id=r["id"], kind=r["kind"], group=r["group"], t0=float(r["t0"]),
                 t1=float(r["t1"]), year=float(r["year"]), weight=float(r["weight"]),
                 source=r["source"], name=r["name"])
        o["cells"] = t(nc[f"obs_{r['key']}"][:]) > 0.5
        if r["kind"] == "lateral":
            o["z"] = t(nc[f"obsz_{r['key']}"][:])
            o["sigma"] = float(r["sigma"])
        elif r["kind"] == "frontal":
            o["outside"] = t(nc[f"obsout_{r['key']}"][:]) > 0.5
        else:
            o["z_pt"] = float(r["z"])
            o["sigma"] = 30.0
        observations.append(o)

    if fields:
        valid = t(geo["present_valid"]) > 0.5
        lia_cells = t(geo["lia_arve"]) > 0.5
        hol = t(geo["catchment_holocene"]) > 0.5
        for year, name in ((2012.7, "thk_2012"), (2021.7, "thk_2021")):
            observations.append(dict(
                id=f"{name}", kind="thickness", group="field", t0=year, t1=year,
                year=year, weight=1.0, cells=valid, H_obs=t(geo[name]), sigma=30.0,
                source="Argentière radar/DEM products", name="Argentière"))
        observations.append(dict(
            id="lia_surface", kind="thickness", group="field", t0=1850.0, t1=1850.0,
            year=1850.0, weight=1.0, cells=lia_cells, H_obs=t(geo["thk_lia"]),
            sigma=50.0, source="Reinthaler & Paul 2025", name="LIA glaciers"))
        observations.append(dict(
            id="lia_extent", kind="extent", group="field", t0=1850.0, t1=1850.0,
            year=1850.0, weight=1.0, cells=hol, H_obs=t(geo["thk_lia"]),
            source="Reinthaler & Paul 2025", name="LIA glaciers"))
    if valley:
        observations = _valley_observations(geo, nc, df, observations, dx, fields)
    nc.close()
    return geo, observations


def _valley_observations(geo, nc, df, observations, dx, fields):
    """Re-express the Argentière landforms against their valleys' flowlines.

    One flowline per valley in ``ARGENTIERE_VALLEYS`` (built from that valley's
    moraines, or from all its landforms where it has none), plus the Tour for
    its LIA terminus. Every landform cell is referred to its nearest flowline
    cell. Frontal moraines become ``terminus``; lateral moraines ``lateral_v``
    (crest elevation = bed at the mapped line); bedrock and wood the ``_v``
    kinds at the sample elevation; dated moraines also get ``preserve``. The
    thickness and extent fields are kept, and the mapped termini added.
    """
    import math
    import numpy as np
    from core.valley import build_flowline, d8_receivers

    f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=device)
    i64 = lambda a: torch.tensor(np.asarray(a, dtype=np.int64), device=device)
    bed = geo["topg"].astype(float)
    lia = geo["lia_id"].astype(int)
    recv, zz = d8_receivers(bed)
    rows = {r["id"]: r for _, r in df.iterrows()}

    def cells_of(r):
        var = f"line_{r['key']}"
        return np.asarray(nc[var if var in nc.variables else f"obs_{r['key']}"][:]) > 0.5

    cells = {oid: cells_of(r) for oid, r in rows.items()}
    members = {name: [oid for oid, r in rows.items() if r["name"] in names]
               for name, (_, names) in ARGENTIERE_VALLEYS.items()}
    stray = set(rows) - {oid for ids in members.values() for oid in ids}
    if stray:
        raise RuntimeError(f"no valley assigned to {sorted(stray)}")
    flowlines = {}
    for name, (pid, _) in ARGENTIERE_VALLEYS.items():
        ids = members[name]
        built_from = [i for i in ids if rows[i]["kind"] in ("lateral", "frontal")] or ids
        flowlines[name] = build_flowline(
            name, {i: cells[i] for i in built_from}, bed, dx, recv, zz,
            prefer_mask=lia == pid, other_ice=(lia > 0) & (lia != pid))
    tour = lia == TOUR_LIA_ID
    low = np.zeros(bed.size, dtype=bool)
    low[np.flatnonzero(tour.ravel())[np.argmin(zz[tour.ravel()])]] = True
    flowlines["Tour"] = build_flowline("Tour", {"tour": low.reshape(bed.shape)}, bed, dx,
                                       recv, zz, prefer_mask=tour,
                                       other_ice=(lia > 0) & ~tour)
    geo["flowlines"] = flowlines
    # The terminus counts ice contiguous with the glacier from a point it always
    # covers: the first flowline cell with >= 50 m of LIA ice in its LIA polygon.
    for name, fl in flowlines.items():
        pid = ARGENTIERE_VALLEYS[name][0] if name in ARGENTIERE_VALLEYS else TOUR_LIA_ID
        thick = ((lia == pid) & (geo["thk_lia"] >= 50.0)).ravel()[fl.active]
        fl.start = int(np.flatnonzero(thick)[0]) if thick.any() else 0
    line_t = {name: (i64(fl.active[fl.start:]), f32(fl.s[fl.start: fl.stop]))
              for name, fl in flowlines.items()}
    valley_of = {oid: name for name, ids in members.items() for oid in ids}

    out = []
    for o in observations:
        r = rows.get(o["id"])
        if r is None:                                     # a field: kept as it is
            out.append(dict(o, group_id=o["id"]))
            continue
        name = valley_of[o["id"]]
        fl = flowlines[name]
        k, d = fl.project(cells[o["id"]])
        path, s = line_t[name]
        v = dict(id=o["id"], group=o["group"], t0=o["t0"], t1=o["t1"], year=o["year"],
                 weight=o["weight"], source=o["source"], name=o["name"], valley=name,
                 path=path, s=s, axis=i64(fl.active[k]), sigma_L=VALLEY["sigma_L"],
                 sigma_z=VALLEY["sigma_z"], delta=VALLEY["delta"], group_id=o["id"],
                 s_cells=fl.s[k], axis_dist_m=d, volume_scale=VOLUME_SCALE[name])
        kind = r["kind"]
        stadial = str(r["stadial"]) if str(r["stadial"]) not in ("-", "nan") else None
        if kind == "frontal":
            v.update(kind="terminus", s_obs=float(np.median(fl.s[k])))
        elif kind == "lateral":
            v.update(kind="lateral_v", z=f32(bed[cells[o["id"]]]), s_req=float(fl.s[k].max()))
        elif kind == "icefree":
            v.update(kind="icefree_v", z=f32(np.full(k.size, float(r["z"]))))
        elif kind in ("icefree_any", "covered_any"):
            v.update(kind=kind + "_v", z=f32(np.full(k.size, float(r["z"]))))
            if kind == "covered_any":
                v["s_req"] = float(fl.s[k].max())
            if stadial:            # one ice-free interval / one advance counts once
                v["group_id"] = f"{stadial} | {kind}"
        else:
            raise ValueError(f"{o['id']}: no valley version of kind {kind!r}")
        out.append(v)
        if kind in ("frontal", "lateral") and o["group"] == "dated":
            pv = dict(v, id=o["id"] + "-preserve", kind="preserve", t0=o["t1"], t1=math.inf,
                      weight=VALLEY["preserve_weight"], group_id=o["id"] + "-preserve")
            pv.pop("s_req", None)
            out.append(pv)

    if fields:
        targets = [("Argentiere", "lia_id", 4053, 1850.0), ("Mer de Glace", "lia_id", 967, 1850.0),
                   ("Tour", "lia_id", TOUR_LIA_ID, 1850.0),
                   ("Argentiere", "thk_2012", None, 2012.7), ("Argentiere", "thk_2021", None, 2021.7)]
        for name, var, pid, year in targets:
            fl = flowlines[name]
            m = (geo[var] == pid) if pid is not None else (geo[var] > 1.0)
            path, s = line_t[name]
            tag = "lia" if var == "lia_id" else var[4:]
            out.append(dict(
                id=f"terminus_{tag}_{name.replace(' ', '')}", kind="terminus", group="field",
                t0=year, t1=year, year=year, weight=1.0, valley=name, path=path, s=s,
                s_obs=fl.position_of(m), sigma_L=VALLEY["sigma_L_mapped"],
                volume_scale=VOLUME_SCALE[name],
                group_id=f"terminus_{tag}_{name.replace(' ', '')}",
                source="Reinthaler & Paul 2025" if pid else "Argentière radar/DEM products",
                name=name))
    return out

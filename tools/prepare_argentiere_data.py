"""Put the Argentière / upper-Arve data onto one model grid, with every dated constraint.

Reads ``data/Argentiere/sources/`` (written by ``tools/fetch_argentiere_sources.py``)
and writes, for a chosen resolution,

* ``data/Argentiere/argentiere_<dx>m.nc`` -- bed, present (2012, 2021) and LIA
  (1850) geometry, the masks ice may occupy, and one set of target cells per
  observation;
* ``data/Argentiere/argentiere_observations_<dx>m.csv`` -- the same observations
  as a table: kind, age, time window, target elevation, weight, source.

The grid covers the whole upper Arve (Argentière, Tour, Mer de Glace with Talèfre
and Leschaux, the valley floor down to below La Joux), because the Lateglacial
tongue at La Joux sits on the Arve valley floor and the Talèfre and Mer de Glace
records belong to the glacier that joins it 2 km further down.

Bed: Argentière's own bed (radar-based, ``present/bedrock_100m_utm32n.tif``)
under the 2012 glacier, blended over 300 m into the ice-free Alps bed of Henz et
al. (2025) everywhere else. The radar product carries the *ice surface* under
the neighbouring glaciers inside its box, so it is only used under Argentière.

Observation kinds (see ``core/moraine_obs.py`` for the operators):

``lateral``   lateral moraine crest -> the simulated ice surface at the cells
              ``k`` steps downslope (toward the ice) must reach the crest
              elevation, at some time inside the age window.
``frontal``   frontal moraine -> ice on the cells just upstream, none on the
              cells just downstream, at some time inside the age window.
``icefree``   bedrock exposure or erratic boulder -> no ice above the sample at
              any time after its exposure age.
``icefree_any`` radiocarbon-dated wood in a lateral moraine (Mer de Glace,
              Le Roy et al. 2015) -> the glacier surface was below the wood at
              some time inside the calibrated range.
``covered_any`` the same wood was overridden by the next advance -> the surface
              rose to the wood within ``COVER_LAG`` years after the range.

Ages: TCN ages are relative to the sampling year (``20YY`` from the sample
name, e.g. ``ARG-16-9``), 14C ages are cal BP (1950). Model time is calendar
year CE throughout.

    python tools/prepare_argentiere_data.py --dx 100

Needs geopandas and rasterio (the ``igm`` env here).
"""

import argparse
import heapq
import os
import re

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy import ndimage

SRC = "data/Argentiere/sources"
OUT_DIR = "data/Argentiere"
# Working grid: the whole source crop (minus 1 km). The written grid is cut from
# it automatically -- the bounding box of everywhere ice may grow, plus MARGIN_M
# -- so the ice masks never reach IGM's 5-cell boundary ramp.
BOUNDS = (316000.0, 5060000.0, 370000.0, 5114000.0)   # x0, y0, x1, y1, EPSG:32632
MARGIN_M = 3000.0         # ice-free margin between the ice masks and the domain edge
CORRIDOR_M = 8000.0       # how far below the Arveyron confluence ice may advance (Lateglacial)
CORRIDOR_H_M = 3000.0     # ... and in the Holocene runs
CRS = "EPSG:32632"
BLEND_M = 300.0           # radar bed -> Henz bed transition around the 2012 glacier
INBOARD_K = 3             # default lateral-moraine offset (cells); recalibrated below
FRONTAL_K = 3             # frontal moraine: cells checked upstream and downstream
COVER_LAG = 400.0         # yr after an ice-free interval within which wood was buried
# Morphostratigraphic window for undated Lateglacial ice margins: younger than the
# erratic at La Joux (13.24 ka, ice-free in front of J1) and older than J2 (10.91 ka).
MORPHO_WINDOW_KA = (10.9, 13.2)

# Reinthaler & Paul (2025) LIA_ID of the three glaciers that matter here (Gardent
# 2014 outlines): checked against the 2012 Argentière glacier and the Talèfre and
# Mer de Glace samples, which fall inside them.
LIA_IDS = {"argentiere": 4053, "mer_de_glace": 967, "tour": 4056}


# --------------------------------------------------------------------------
# grid and raster helpers
# --------------------------------------------------------------------------
def make_grid(dx):
    x0, y0, x1, y1 = BOUNDS
    nx, ny = int(round((x1 - x0) / dx)), int(round((y1 - y0) / dx))
    x = x0 + dx * (np.arange(nx) + 0.5)
    y = y0 + dx * (np.arange(ny) + 0.5)           # ascending, row 0 = south
    transform = from_origin(x0, y1, dx, dx)        # rasterio wants north-up
    return dict(x=x, y=y, nx=nx, ny=ny, dx=dx, transform=transform)


def to_grid(path, grid, resampling=Resampling.average, nodata=None):
    """Resample a raster onto the model grid (returned with ascending y)."""
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float64)
        nd = src.nodata if nodata is None else nodata
        if nd is not None:
            a[a == nd] = np.nan
        dst = np.full((grid["ny"], grid["nx"]), np.nan)
        reproject(a, dst, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=np.nan, dst_transform=grid["transform"],
                  dst_crs=CRS, dst_nodata=np.nan, resampling=resampling)
    return dst[::-1]


def burn(geoms, grid, all_touched=True, values=None):
    """Rasterize geometries onto the grid (ascending y)."""
    if values is None:
        shapes = [(g, 1) for g in geoms]
    else:
        shapes = list(zip(geoms, values))
    if not shapes:
        return np.zeros((grid["ny"], grid["nx"]), dtype=np.int32)
    a = rasterize(shapes, out_shape=(grid["ny"], grid["nx"]),
                  transform=grid["transform"], all_touched=all_touched,
                  fill=0, dtype="int32")
    return a[::-1]


def cell_of(xp, yp, grid):
    ix = int(np.floor((xp - BOUNDS[0]) / grid["dx"]))
    iy = int(np.floor((yp - BOUNDS[1]) / grid["dx"]))
    if 0 <= ix < grid["nx"] and 0 <= iy < grid["ny"]:
        return iy, ix
    return None


# --------------------------------------------------------------------------
# drainage (same algorithms as tools/prepare_yd_data.py)
# --------------------------------------------------------------------------
def fill_depressions(z, eps=1e-3):
    """Priority-flood depression filling (Barnes et al. 2014) with an epsilon slope.

    The epsilon (mm per cell) keeps every filled flat draining, so a flow path
    never stalls on a lake surface or a flat valley floor.
    """
    ny, nx = z.shape
    filled = np.full_like(z, np.inf)
    done = np.zeros(z.shape, dtype=bool)
    heap = []
    for iy in range(ny):
        for ix in (0, nx - 1):
            heapq.heappush(heap, (z[iy, ix], iy, ix))
            done[iy, ix] = True
            filled[iy, ix] = z[iy, ix]
    for ix in range(nx):
        for iy in (0, ny - 1):
            if not done[iy, ix]:
                heapq.heappush(heap, (z[iy, ix], iy, ix))
                done[iy, ix] = True
                filled[iy, ix] = z[iy, ix]
    while heap:
        zc, iy, ix = heapq.heappop(heap)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                jy, jx = iy + dy, ix + dx
                if dy == dx == 0 or not (0 <= jy < ny and 0 <= jx < nx):
                    continue
                if done[jy, jx]:
                    continue
                done[jy, jx] = True
                filled[jy, jx] = max(z[jy, jx], zc + eps)
                heapq.heappush(heap, (filled[jy, jx], jy, jx))
    return filled


def d8_receivers(z, fill=True):
    """Steepest-descent receiver (flat index) of every cell; self if none."""
    ny, nx = z.shape
    zz = fill_depressions(z) if fill else z
    flat_idx = np.arange(z.size).reshape(z.shape)
    recv = np.arange(z.size)
    best = np.zeros(z.size)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            dist = np.hypot(dy, dx)
            shifted = np.full(z.shape, np.inf)
            idx = np.full(z.shape, -1)
            ys = slice(max(0, dy), ny + min(0, dy))
            xs = slice(max(0, dx), nx + min(0, dx))
            ys_src = slice(max(0, -dy), ny + min(0, -dy))
            xs_src = slice(max(0, -dx), nx + min(0, -dx))
            shifted[ys_src, xs_src] = zz[ys, xs]
            idx[ys_src, xs_src] = flat_idx[ys, xs]
            slope = (zz - shifted).ravel() / dist
            better = slope > best
            best[better] = slope[better]
            recv[better] = idx.ravel()[better].astype(int)
    return recv, zz.ravel()


def upstream_of(seeds, recv, flat):
    reaches = seeds.ravel().copy()
    for cell in np.argsort(flat):
        if not reaches[cell] and reaches[recv[cell]]:
            reaches[cell] = True
    return reaches.reshape(seeds.shape)


def downstream_of(seeds, recv):
    path = np.zeros(seeds.size, dtype=bool)
    for cell in np.flatnonzero(seeds.ravel()):
        while not path[cell]:
            path[cell] = True
            nxt = recv[cell]
            if nxt == cell:
                break
            cell = nxt
    return path.reshape(seeds.shape)


def walk(cells, recv, k):
    """Follow receivers ``k`` steps from each flat index; returns flat indices."""
    out = np.asarray(cells).copy()
    for _ in range(k):
        out = recv[out]
    return out


# --------------------------------------------------------------------------
# observation operators on the grid
# --------------------------------------------------------------------------
def lateral_cells(moraine, bed, recv_raw, k):
    """Moraine cells stepped ``k`` cells downslope, with the crest elevation to reach.

    Downslope on the bed points from a valley-side ridge toward the valley axis,
    i.e. toward the ice that built it, whichever glacier and direction that is.
    """
    src = np.flatnonzero(moraine.ravel())
    dst = walk(src, recv_raw, k)
    total = np.zeros(moraine.size)
    count = np.zeros(moraine.size)
    np.add.at(total, dst, bed.ravel()[src])
    np.add.at(count, dst, 1.0)
    inb = (count > 0).reshape(moraine.shape) & ~moraine
    z = np.where(inb, (total / np.maximum(count, 1)).reshape(moraine.shape), np.nan)
    return inb, z


def frontal_cells(moraine, recv, k):
    """Cells within ``k`` steps upstream (ice expected) and downstream (no ice)."""
    shape = moraine.shape
    flat = moraine.ravel()
    down = np.zeros(moraine.size, dtype=bool)
    cur = np.flatnonzero(flat)
    for _ in range(k):
        cur = recv[cur]
        down[cur] = True
    up = np.zeros(moraine.size, dtype=bool)
    hit = flat.copy()
    # cells whose path enters the moraine within k steps
    cur = np.arange(moraine.size)
    for _ in range(k):
        cur = recv[cur]
        up |= flat[cur]
    up &= ~flat
    down &= ~flat & ~up
    del hit
    return up.reshape(shape), down.reshape(shape)


def sample_year(names, default=2015):
    """Sampling year from names such as ``ARG-16-9`` or ``JDT-16-13``."""
    years = []
    for n in names:
        m = re.search(r"-(\d{2})-", str(n))
        if m:
            years.append(2000 + int(m.group(1)))
    return float(np.median(years)) if years else float(default)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=100.0)
    opts = ap.parse_args()
    grid = make_grid(opts.dx)
    dx = grid["dx"]
    ny, nx = grid["ny"], grid["nx"]
    gx, gy = grid["x"], grid["y"]
    X, Y = np.meshgrid(gx, gy)
    tag = f"{int(dx)}m"
    print(f"grid {ny} x {nx} at {dx:.0f} m, x {BOUNDS[0]:.0f}-{BOUNDS[2]:.0f}, "
          f"y {BOUNDS[1]:.0f}-{BOUNDS[3]:.0f}")

    # ---------------- bed ----------------
    pres = os.path.join(SRC, "present")
    bed_henz = to_grid(os.path.join(SRC, "bed_henz2025_50m_chamonix.tif"), grid)
    bed_local = to_grid(os.path.join(pres, "bedrock_100m_utm32n.tif"), grid,
                        Resampling.bilinear)
    thk12 = to_grid(os.path.join(pres, "thickness_2012_100m_utm32n.tif"), grid,
                    Resampling.bilinear)
    thk21 = to_grid(os.path.join(pres, "thickness_2021_100m_utm32n.tif"), grid,
                    Resampling.bilinear)
    dem12 = to_grid(os.path.join(pres, "dem_2012_100m_utm32n.tif"), grid,
                    Resampling.bilinear)
    dem21 = to_grid(os.path.join(pres, "dem_2021_100m_utm32n.tif"), grid,
                    Resampling.bilinear)
    # The Argentière products are ~52 m above AW3D30 / Henz on ice-free ground: an
    # ellipsoidal vs orthometric datum (the geoid is ~52 m above WGS84 here).
    # AlpIce-1 elevations and the LIA surface are orthometric, so the radar
    # products are shifted by a plane fitted on terrain that has been ice-free
    # since the LIA.
    lia_any = burn(gpd.read_file(os.path.join(SRC, "lia_outlines_chamonix.gpkg")).geometry,
                   grid) > 0
    free = (ndimage.distance_transform_edt(~lia_any) * dx > 300) & np.isfinite(dem12)
    A = np.c_[np.ones(free.sum()), X[free] - X.mean(), Y[free] - Y.mean()]
    coef, *_ = np.linalg.lstsq(A, (dem12 - bed_henz)[free], rcond=None)
    geoid = coef[0] + coef[1] * (X - X.mean()) + coef[2] * (Y - Y.mean())
    resid = (dem12 - bed_henz)[free] - geoid[free]
    print(f"  datum: radar DEM - AW3D30 on {free.sum()} ice-free cells = "
          f"{coef[0]:+.1f} m (+{coef[1] * 1e3:.2f}/{coef[2] * 1e3:.2f} m/km), "
          f"residual std {resid.std():.1f} m -> removed")
    bed_local, dem12, dem21 = bed_local - geoid, dem12 - geoid, dem21 - geoid
    radar_box = np.isfinite(bed_local) & np.isfinite(thk12)
    radar_box = ndimage.binary_erosion(radar_box, iterations=2)
    arg12 = radar_box & (np.nan_to_num(thk12) > 1.0)
    # largest connected component = Argentière itself
    lab, n = ndimage.label(arg12)
    arg12 = lab == (np.argmax(ndimage.sum(arg12, lab, range(1, n + 1))) + 1)
    dist_out = ndimage.distance_transform_edt(~arg12) * dx
    w = np.clip(1.0 - dist_out / BLEND_M, 0.0, 1.0) * radar_box
    bed = np.where(w > 0, w * np.nan_to_num(bed_local) + (1 - w) * bed_henz, bed_henz)
    if not np.isfinite(bed).all():
        raise RuntimeError("bed has holes")
    ring = (dist_out > 0) & (dist_out <= BLEND_M) & radar_box
    off = (bed_local - bed_henz)[ring]
    print(f"  bed: radar product vs Henz on the {BLEND_M:.0f} m ring around the "
          f"2012 glacier: median {np.nanmedian(off):+.0f} m, IQR "
          f"{np.nanpercentile(off, 25):+.0f}/{np.nanpercentile(off, 75):+.0f} m")
    under = arg12
    print(f"  bed under Argentière: radar {np.nanmean(bed_local[under]):.0f} m vs "
          f"Henz {np.nanmean(bed_henz[under]):.0f} m (mean)")

    thk12 = np.where(arg12, np.clip(np.nan_to_num(thk12), 0, None), 0.0)
    arg21 = radar_box & (np.nan_to_num(thk21) > 1.0) & ndimage.binary_dilation(arg12, iterations=2)
    thk21 = np.where(arg21, np.clip(np.nan_to_num(thk21), 0, None), 0.0)
    print(f"  Argentière: 2012 {arg12.sum() * dx * dx / 1e6:.1f} km2 "
          f"{thk12.sum() * dx * dx / 1e9:.2f} km3; 2021 {arg21.sum() * dx * dx / 1e6:.1f} km2 "
          f"{thk21.sum() * dx * dx / 1e9:.2f} km3")
    # where the present surface is known (radar box) -- the only place 2012/2021
    # thickness can be scored, ice or no ice.
    present_valid = radar_box & ~(np.nan_to_num(bed_local - bed_henz) > 30) | arg12
    present_valid &= radar_box

    # ---------------- LIA ----------------
    lia = gpd.read_file(os.path.join(SRC, "lia_outlines_chamonix.gpkg")).to_crs(CRS)
    lia_id = burn(lia.geometry, grid, all_touched=False, values=lia.LIA_ID.astype(int))
    surf_lia = to_grid(os.path.join(SRC, "lia_surface_30m_chamonix.tif"), grid, nodata=0)
    lia_mask = lia_id > 0
    thk_lia = np.where(lia_mask & np.isfinite(surf_lia),
                       np.clip(np.nan_to_num(surf_lia) - bed, 0, None), 0.0)
    pick = LIA_IDS
    lia_arg = lia_id == pick["argentiere"]
    lia_mdg = lia_id == pick["mer_de_glace"]
    lia_tour = lia_id == pick["tour"]
    share = (lia_arg & arg12).sum() / arg12.sum()
    print(f"  LIA polygons {pick}: {share * 100:.0f}% of the 2012 Argentière lies in "
          f"LIA_ID {pick['argentiere']}")
    if share < 0.8:
        raise RuntimeError("LIA_IDS no longer matches the Argentière glacier")
    print(f"  LIA Argentière {lia_arg.sum() * dx * dx / 1e6:.1f} km2, mean thk "
          f"{thk_lia[lia_arg].mean():.0f} m; Mer de Glace {lia_mdg.sum() * dx * dx / 1e6:.1f} km2, "
          f"mean thk {thk_lia[lia_mdg].mean():.0f} m; surface coverage of LIA cells "
          f"{np.isfinite(surf_lia[lia_mask]).mean() * 100:.0f}%")

    # ---------------- AlpIce-1 ----------------
    gpkg = os.path.join(SRC, "alpice1_chamonix.gpkg")
    im = gpd.read_file(gpkg, layer="ice_margin").to_crs(CRS)
    ev_pt = gpd.read_file(gpkg, layer="tcn_event_pt").to_crs(CRS)
    tcn = gpd.read_file(gpkg, layer="tcn_ages").to_crs(CRS)
    c14 = gpd.read_file(gpkg, layer="c14_ages").to_crs(CRS)

    recv_fill, flat = d8_receivers(bed, fill=True)
    recv_raw, _ = d8_receivers(bed, fill=False)

    obs = []          # list of dicts; masks kept separately
    masks = {}

    def add(oid, **kw):
        kw["id"] = oid
        obs.append(kw)

    # ---- dated and morphostratigraphic ice margins ----
    for _, r in im.iterrows():
        sid = r.site_ID
        info = " ".join(str(v) for v in tcn.loc[tcn.site_ID == sid, "site_info"].unique())
        stad = str(r.pgl_stad)
        age = num(r.e_age)
        if np.isfinite(age):
            group = "dated"
            pos, neg = num(r.e_pos_err), num(r.e_neg_err)
            yr = sample_year(tcn.loc[tcn.site_ID == sid, "samp_name"])
            centre = yr - age
            win = (yr - age - pos, yr - age + neg)
        elif "YD/EH" in stad or sid == "ICEM-Protin2019-D":
            group = "morphostrat"
            yr = 2017.0
            centre = yr - 1000 * np.mean(MORPHO_WINDOW_KA)
            win = (yr - 1000 * MORPHO_WINDOW_KA[1], yr - 1000 * MORPHO_WINDOW_KA[0])
            age, pos, neg = 1000 * np.mean(MORPHO_WINDOW_KA), np.nan, np.nan
        else:
            print(f"  skip {sid:22s} {r.site_names:18s} stadial {stad!r}: no age, no window")
            continue
        line = burn([r.geometry], grid) > 0
        if not line.any():
            continue
        frontal = ("La Joux" in str(r.site_names)) and ("J2" in info or
                                                         (sid == "ICEM-Protin2019-E"))
        base = sid if sid != "undated" else f"undated-{r.site_names.replace(' ', '')}"
        oid = base
        k = 2
        while any(o["id"] == oid for o in obs):
            oid = f"{base}-{k}"
            k += 1
        kind = "frontal" if frontal else "lateral"
        add(oid, kind=kind, site=sid, name=str(r.site_names), glacier=str(r.pgl_name),
            stadial=stad, group=group, age_yr=age, err_pos=pos, err_neg=neg,
            year=centre, t0=win[0], t1=win[1], info=info[:120],
            weight=1.0 if group == "dated" else 0.5, source=str(r.orig_ref))
        masks[oid] = dict(line=line)
        # positional check: DEM at the boulder vs the boulder's own elevation
        b = tcn[tcn.site_ID == sid]
        if len(b):
            d = []
            for _, s in b.iterrows():
                c = cell_of(s.geometry.x, s.geometry.y, grid)
                if c:
                    d.append(bed[c] - num(s.elevation))
            obs[-1]["dem_minus_boulder_m"] = float(np.nanmedian(d)) if d else np.nan

    # ---- bedrock exposures and erratics: ice-free after exposure ----
    for _, r in ev_pt.iterrows():
        if str(r.pc_cat) not in ("bedrock exposure", "ice free") or str(r.site_ID).startswith("MM-"):
            continue
        samples = tcn[tcn.site_ID == r.site_ID]
        age, pos, neg = num(r.e_age), num(r.e_pos_err), num(r.e_neg_err)
        yr = sample_year(samples.samp_name)
        z = float(np.nanmean([num(e) for e in samples.elevation])) if len(samples) else np.nan
        c = cell_of(r.geometry.x, r.geometry.y, grid)
        if c is None or not np.isfinite(z):
            continue
        cellmask = np.zeros((ny, nx), dtype=bool)
        cellmask[c] = True
        add(r.site_ID, kind="icefree", site=r.site_ID, name=str(r.site_names),
            glacier="-", stadial="-", group="dated", age_yr=age, err_pos=pos,
            err_neg=neg, year=yr - age, t0=yr - age + neg, t1=np.inf, z=z,
            info=str(samples.site_info.iloc[0])[:120] if len(samples) else "",
            weight=1.0, source=str(r.orig_ref))
        masks[r.site_ID] = dict(cell=cellmask)

    # ---- Mer de Glace subfossil wood (Le Roy et al. 2015) ----
    wood = c14[(c14.orig_ref.str.contains("Le Roy", na=False)) &
               (c14.confid_ra.isin(["high", "medium"]))]
    groups = wood.groupby(["site_ID", "recal_min", "recal_max"])
    for (sid, rmin, rmax), g in groups:
        rmin, rmax = num(rmin), num(rmax)
        z = float(g.elevation.astype(float).min())
        p = g.geometry.iloc[0]
        c = cell_of(p.x, p.y, grid)
        if c is None:
            continue
        cellmask = np.zeros((ny, nx), dtype=bool)
        cellmask[c] = True
        stad = "/".join(sorted(set(str(s) for s in g.pgl_stad if str(s) != "-"))) or "-"
        base = f"{sid}-{int(rmax)}"
        add(base + "-free", kind="icefree_any", site=sid, name="Mer de Glace",
            glacier="Mer de Glace glacier", stadial=stad, group="dated",
            age_yr=0.5 * (rmin + rmax), err_pos=0.5 * (rmax - rmin),
            err_neg=0.5 * (rmax - rmin), year=1950 - 0.5 * (rmin + rmax),
            t0=1950 - rmax, t1=1950 - rmin, z=z, info=f"{len(g)} wood samples",
            weight=1.0, source="Le Roy et al., 2015")
        masks[base + "-free"] = dict(cell=cellmask)
        if any("readvance" in str(s) for s in g.pc_subcat):
            add(base + "-cover", kind="covered_any", site=sid, name="Mer de Glace",
                glacier="Mer de Glace glacier", stadial=stad, group="dated",
                age_yr=0.5 * (rmin + rmax), err_pos=np.nan, err_neg=np.nan,
                year=1950 - rmin + 0.5 * COVER_LAG, t0=1950 - rmin,
                t1=1950 - rmin + COVER_LAG, z=z, info="overridden by the next advance",
                weight=0.5, source="Le Roy et al., 2015")
            masks[base + "-cover"] = dict(cell=cellmask)

    # ---- calibrate the lateral offset k on the dated LIA moraine (Talèfre T4) ----
    print("  lateral-moraine offset, checked against the Reinthaler & Paul LIA surface:")
    lia_lat = [o for o in obs if o["kind"] == "lateral" and o["group"] == "dated"
               and 1500 < o["year"] < 1900]
    best_k, best_score = INBOARD_K, np.inf
    for k in (1, 2, 3, 4, 5):
        res = []
        for o in lia_lat:
            inb, z = lateral_cells(masks[o["id"]]["line"], bed, recv_raw, k)
            have = inb & np.isfinite(surf_lia) & lia_mask
            res.append(surf_lia[have] - z[have])
        r = np.concatenate(res) if res else np.array([])
        if r.size:
            score = abs(np.median(r)) + 0.3 * r.std()
            print(f"    k={k}: median {np.median(r):+5.0f} m, std {r.std():4.0f} m, n={r.size}")
            if score < best_score:
                best_k, best_score, best_r = k, score, r
    sigma_lat = float(max(30.0, best_r.std())) if np.isfinite(best_score) else 30.0
    bias_lat = float(np.median(best_r)) if np.isfinite(best_score) else 0.0
    print(f"  -> k={best_k}, bias {bias_lat:+.0f} m, sigma {sigma_lat:.0f} m (floored at 30 m)")

    for o in obs:
        m = masks[o["id"]]
        if o["kind"] == "lateral":
            inb, z = lateral_cells(m["line"], bed, recv_raw, best_k)
            m.update(target=inb, z=z + bias_lat)
            o["sigma"] = sigma_lat
        elif o["kind"] == "frontal":
            up, down = frontal_cells(m["line"], recv_fill, FRONTAL_K)
            m.update(target=up, outside=down)
            o["sigma"] = np.nan
        else:
            m.update(target=m["cell"])
            o["sigma"] = 20.0
        o["n_cells"] = int(m["target"].sum())
        tgt = m["target"]
        o["x"] = float(X[tgt].mean()) if tgt.any() else np.nan
        o["y"] = float(Y[tgt].mean()) if tgt.any() else np.nan
        o["bed_m"] = float(np.nanmean(bed[tgt])) if tgt.any() else np.nan
        if o["kind"] == "lateral":
            o["z"] = float(np.nanmean(m["z"][tgt]))

    # ---------------- masks where ice may grow ----------------
    feats = np.zeros((ny, nx), dtype=bool)
    for o in obs:
        feats |= masks[o["id"]].get("line", masks[o["id"]]["target"])
    seeds_lg = lia_mask.copy()
    # only LIA glaciers that drain through the upper Arve (not Italy / Switzerland)
    # The upper Arve basin: everything draining to the Arve just below the Mer de
    # Glace (Arveyron) confluence. Glaciers that join further down (Bossons,
    # Taconnaz) or drain to Italy or Switzerland are left out.
    def terminus(m):
        cells = np.flatnonzero(m.ravel())
        t = np.zeros(m.size, dtype=bool)
        t[cells[np.argmin(flat[cells])]] = True
        return t.reshape(m.shape)
    path_a = downstream_of(terminus(lia_arg), recv_fill)
    path_m = downstream_of(terminus(lia_mdg), recv_fill)
    joint = np.flatnonzero((path_a & path_m).ravel())
    if joint.size == 0:
        raise RuntimeError("Argentière and Mer de Glace do not drain to a common point")
    pour = joint[np.argmax(flat[joint])]            # highest shared cell = confluence
    pour = walk([pour], recv_fill, 3)[0]
    seed = np.zeros(ny * nx, dtype=bool)
    seed[pour] = True
    arve_basin = upstream_of(seed.reshape(ny, nx), recv_fill, flat)
    py_, px_ = np.unravel_index(pour, (ny, nx))
    print(f"  Arve basin above ({gx[px_]:.0f}, {gy[py_]:.0f}): "
          f"{arve_basin.sum() * dx * dx / 1e6:.0f} km2")
    seeds_lg &= arve_basin
    seeds_lg |= feats & arve_basin
    def below_pour(max_m):
        """Flow path from the confluence down the Arve, ``max_m`` metres long."""
        tail = np.zeros(ny * nx, dtype=bool)
        cell, dist = pour, 0.0
        while dist < max_m:
            tail[cell] = True
            nxt = recv_fill[cell]
            if nxt == cell:
                break
            (ya, yb), (xa, xb) = np.unravel_index([cell, nxt], (ny, nx))
            dist += np.hypot(yb - ya, xb - xa) * dx
            cell = nxt
        return tail.reshape(ny, nx)

    # Downstream of the confluence the corridor is cut after CORRIDOR_M; the mask
    # (and its -10 m/yr outside it) then ends the tongue inside the domain
    # instead of letting it run into the boundary.
    corridor = ndimage.binary_dilation(
        (downstream_of(seeds_lg, recv_fill) & arve_basin) | below_pour(CORRIDOR_M), iterations=4)
    catch_lg = ndimage.binary_fill_holes(upstream_of(seeds_lg, recv_fill, flat) | corridor)
    seeds_h = (lia_mask | arg12) & arve_basin
    corridor_h = ndimage.binary_dilation(
        (downstream_of(seeds_h, recv_fill) & arve_basin) | below_pour(CORRIDOR_H_M), iterations=2)
    catch_h = ndimage.binary_fill_holes(
        ndimage.binary_dilation(seeds_h, iterations=3) & arve_basin | corridor_h)
    print(f"  catchment (Lateglacial) {catch_lg.sum() * dx * dx / 1e6:.0f} km2, "
          f"Holocene {catch_h.sum() * dx * dx / 1e6:.0f} km2, LIA ice in the Arve "
          f"basin {(lia_mask & arve_basin).sum() * dx * dx / 1e6:.0f} km2")
    # observations that fall outside the Lateglacial catchment cannot be fitted
    for o in obs:
        inside = catch_lg[masks[o["id"]]["target"]].mean() if o["n_cells"] else 0.0
        o["in_catchment"] = bool(inside > 0.5)

    # ---------------- crop to the ice masks plus a margin ----------------
    ice_any = catch_lg | catch_h
    iy, ix = np.nonzero(ice_any)
    x_lo = np.floor((gx[ix.min()] - MARGIN_M) / 1000.0) * 1000.0
    x_hi = np.ceil((gx[ix.max()] + MARGIN_M) / 1000.0) * 1000.0
    y_lo = np.floor((gy[iy.min()] - MARGIN_M) / 1000.0) * 1000.0
    y_hi = np.ceil((gy[iy.max()] + MARGIN_M) / 1000.0) * 1000.0
    if x_lo < BOUNDS[0] or y_lo < BOUNDS[1] or x_hi > BOUNDS[2] or y_hi > BOUNDS[3]:
        raise RuntimeError("ice masks plus margin exceed the source crop; enlarge BOX/BOUNDS")
    xs = slice(int(round((x_lo - BOUNDS[0]) / dx)), int(round((x_hi - BOUNDS[0]) / dx)))
    ys = slice(int(round((y_lo - BOUNDS[1]) / dx)), int(round((y_hi - BOUNDS[1]) / dx)))
    crop = lambda a: a[ys, xs]
    (bed, bed_henz, thk12, thk21, dem12, dem21, radar_box, present_valid, thk_lia,
     surf_lia, lia_id, lia_arg, lia_mdg, lia_mask, arve_basin, catch_lg, catch_h,
     arg12) = map(crop, (bed, bed_henz, thk12, thk21, dem12, dem21, radar_box,
                         present_valid, thk_lia, surf_lia, lia_id, lia_arg, lia_mdg,
                         lia_mask, arve_basin, catch_lg, catch_h, arg12))
    for m in masks.values():
        for k_ in list(m):
            if isinstance(m[k_], np.ndarray) and m[k_].shape == (ny, nx):
                m[k_] = crop(m[k_])
    gx, gy = gx[xs], gy[ys]
    ny, nx = bed.shape
    edge = np.ones((ny, nx), dtype=bool)
    edge[5:-5, 5:-5] = False
    print(f"  domain cropped to x {x_lo:.0f}-{x_hi:.0f}, y {y_lo:.0f}-{y_hi:.0f}: "
          f"{ny} x {nx} cells; nearest ice-mask cell to the edge "
          f"{ndimage.distance_transform_edt(~edge)[ice_any[ys, xs]].min() * dx:.0f} m inside the boundary ramp")

    # ---------------- write ----------------
    os.makedirs(OUT_DIR, exist_ok=True)
    out_nc = os.path.join(OUT_DIR, f"argentiere_{tag}.nc")
    out_csv = os.path.join(OUT_DIR, f"argentiere_observations_{tag}.csv")
    nc = netCDF4.Dataset(out_nc, "w")
    nc.createDimension("x", nx)
    nc.createDimension("y", ny)
    nc.Conventions = "CF-1.8"
    nc.crs = CRS
    nc.title = "Argentière / upper Arve: model grid, present and LIA geometry, dated constraints"
    nc.source = "tools/prepare_argentiere_data.py from data/Argentiere/sources"
    nc.lateral_offset_cells = best_k
    nc.lateral_sigma_m = sigma_lat
    nc.lateral_bias_m = bias_lat
    nc.lia_polygons = str(pick)

    def put(name, data, dtype="f4", dims=("y", "x"), **attrs):
        v = nc.createVariable(name, dtype, dims, zlib=True)
        v[:] = data
        for k_, val in attrs.items():
            setattr(v, k_, val)

    put("x", gx, "f8", ("x",), units="m", axis="X")
    put("y", gy, "f8", ("y",), units="m", axis="Y")
    put("topg", bed, units="m", long_name="bed: radar under Argentière, Henz et al. 2025 elsewhere")
    put("topg_henz", bed_henz, units="m", long_name="Henz et al. 2025 ice-free bed")
    put("thk_2012", thk12, units="m", long_name="Argentière 2012 thickness (radar box only)")
    put("thk_2021", thk21, units="m", long_name="Argentière 2021 thickness (radar box only)")
    put("usurf_2012_obs", np.where(radar_box, np.nan_to_num(dem12), np.nan), units="m")
    put("usurf_2021_obs", np.where(radar_box, np.nan_to_num(dem21), np.nan), units="m")
    put("present_valid", present_valid.astype("f4"), units="1",
        long_name="cells where 2012/2021 thickness is known (Argentière radar box minus other glaciers)")
    put("thk_lia", thk_lia, units="m", long_name="LIA thickness, Reinthaler & Paul surface minus bed")
    put("usurf_lia", np.where(np.isfinite(surf_lia), surf_lia, np.nan), units="m")
    put("lia_id", lia_id.astype("f4"), units="1", long_name="Reinthaler & Paul LIA_ID")
    put("lia_argentiere", lia_arg.astype("f4"), units="1")
    put("lia_merdeglace", lia_mdg.astype("f4"), units="1")
    put("lia_arve", (lia_mask & arve_basin).astype("f4"), units="1",
        long_name="all LIA ice draining to the upper Arve")
    put("catchment_lateglacial", catch_lg.astype("f4"), units="1",
        long_name="where ice may grow for the Lateglacial runs")
    put("catchment_holocene", catch_h.astype("f4"), units="1",
        long_name="where ice may grow for the Holocene / LIA runs")
    put("argentiere_2012", arg12.astype("f4"), units="1")

    for o in obs:
        m = masks[o["id"]]
        key = re.sub(r"[^A-Za-z0-9_]", "_", o["id"])
        o["key"] = key
        put(f"obs_{key}", m["target"].astype("f4"), units="1",
            long_name=f"{o['kind']} target cells of {o['id']}")
        if "outside" in m:
            put(f"obsout_{key}", m["outside"].astype("f4"), units="1")
        if "z" in m:
            put(f"obsz_{key}", np.where(m["target"], m["z"], np.nan), units="m")
        if "line" in m:
            put(f"line_{key}", m["line"].astype("f4"), units="1")
    nc.close()

    cols = ["id", "key", "kind", "group", "site", "name", "glacier", "stadial", "age_yr",
            "err_pos", "err_neg", "year", "t0", "t1", "z", "sigma", "weight", "n_cells",
            "in_catchment", "x", "y", "bed_m", "dem_minus_boulder_m", "source", "info"]
    df = pd.DataFrame(obs)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    df = df[cols].sort_values("year")
    df.to_csv(out_csv, index=False, float_format="%.1f")
    print(f"\nwrote {out_nc}\nwrote {out_csv}")
    with pd.option_context("display.width", 200, "display.max_colwidth", 28):
        print(df[["id", "kind", "group", "year", "t0", "t1", "z", "n_cells",
                  "in_catchment", "dem_minus_boulder_m"]].to_string(index=False))


if __name__ == "__main__":
    main()

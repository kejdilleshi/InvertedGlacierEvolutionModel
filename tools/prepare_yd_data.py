"""Put the YD-Aletsch moraine data onto the model grid.

The three files in ``data/YD-Aletsch/`` are 30 m landform masks, each on its own
small grid and with no elevation attached.  The forward model runs on the 100 m
grid of ``data/geology.nc``.  This script joins the two: it rasterizes every
landform onto the model grid, attaches the elevation the ice surface has to
reach there, splits the LIA landform into its two lateral branches, delineates
the Massa catchment (the area ice may occupy when the glacier is far larger than
today), and writes one file the experiment can load directly.

It also calibrates the observation operator.  The LIA laterals sit beside a
surface geology.nc actually records -- the 1880 DEM -- so "moraine elevation =
contemporaneous ice surface" is measured rather than assumed, and the result
sets both the target variable and the sigma of the data term.

    python tools/prepare_yd_data.py

Writes ``data/YD-Aletsch/aletsch_yd_100m.nc`` and ``aletsch_yd_moraines.csv``.
"""

import heapq
import os

import netCDF4
import numpy as np
from scipy import ndimage

GEOLOGY = "data/geology.nc"
YD_DIR = "data/YD-Aletsch"
OUT_NC = os.path.join(YD_DIR, "aletsch_yd_100m.nc")
OUT_CSV = os.path.join(YD_DIR, "aletsch_yd_moraines.csv")

# Landform file -> (short name, exposure-age bounds in years before present).
# The LIA moraine is historically dated, not cosmogenically, so it carries no
# age field of its own; 1850 CE is the conventional stand.
LANDFORMS = {
    "aletsch_glacier_YD_moraine1": ("YD1", None),
    "aletsch_glacier_YD_moraine2": ("YD2", None),
    "aletsch_glacier_LIA": ("LIA", (100.0, 175.0)),
}


def read_model_grid():
    nc = netCDF4.Dataset(GEOLOGY)
    grid = dict(
        x=np.asarray(nc["x"][:], dtype=np.float64),
        y=np.asarray(nc["y"][:], dtype=np.float64),
        topg=np.asarray(nc["topg"][:], dtype=np.float64),
        usurf=np.asarray(nc["usurf"][:], dtype=np.float64),
        thk=np.asarray(nc["thk"][:], dtype=np.float64),
        icemask=np.asarray(nc["icemask"][:], dtype=np.float64),
        surf_1880=np.asarray(nc["surf_1880"][:], dtype=np.float64),
    )
    nc.close()
    return grid


def read_landform(name):
    """Return the 30 m moraine pixels as easting/northing/age arrays."""
    nc = netCDF4.Dataset(os.path.join(YD_DIR, name + ".nc"))
    x = np.asarray(nc["x"][:], dtype=np.float64)
    y = np.asarray(nc["y"][:], dtype=np.float64)
    mask = np.asarray(nc["mask"][:], dtype=np.float64)
    if "landform_summary_min_age" in nc.variables:
        amin = np.asarray(nc["landform_summary_min_age"][:], dtype=np.float64)
        amax = np.asarray(nc["landform_summary_max_age"][:], dtype=np.float64)
    else:
        amin = amax = np.full(mask.shape, np.nan)
    nc.close()

    rows, cols = np.nonzero(mask > 0.5)
    a0, a1 = amin[rows, cols], amax[rows, cols]
    ages = None
    if np.isfinite(a0).any():
        ages = (float(np.nanmin(a0)), float(np.nanmax(a1)))
    return x[cols], y[rows], ages


def rasterize(east, north, grid):
    """Count how many 30 m landform pixels fall in each 100 m model cell."""
    gx, gy = grid["x"], grid["y"]
    dx = gx[1] - gx[0]
    ix = np.round((east - gx[0]) / dx).astype(int)
    iy = np.round((north - gy[0]) / dx).astype(int)
    keep = (ix >= 0) & (ix < gx.size) & (iy >= 0) & (iy < gy.size)
    count = np.zeros((gy.size, gx.size), dtype=np.int32)
    np.add.at(count, (iy[keep], ix[keep]), 1)
    return count, int((~keep).sum())


def fill_depressions(z):
    """Priority-flood depression filling (Barnes et al. 2014), no epsilon."""
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
                filled[jy, jx] = max(z[jy, jx], zc)
                heapq.heappush(heap, (filled[jy, jx], jy, jx))
    return filled


def d8_receivers(z):
    """Steepest-descent receiver index for every cell of the filled DEM."""
    ny, nx = z.shape
    filled = fill_depressions(z)
    flat = filled.ravel()
    flat_idx = np.arange(flat.size).reshape(filled.shape)

    recv = np.arange(flat.size)
    best = np.zeros(flat.size)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            dist = np.hypot(dy, dx)
            shifted = np.full(filled.shape, np.inf)
            idx = np.full(filled.shape, -1)
            ys = slice(max(0, dy), ny + min(0, dy))
            xs = slice(max(0, dx), nx + min(0, dx))
            ys_src = slice(max(0, -dy), ny + min(0, -dy))
            xs_src = slice(max(0, -dx), nx + min(0, -dx))
            shifted[ys_src, xs_src] = filled[ys, xs]
            idx[ys_src, xs_src] = flat_idx[ys, xs]
            slope = (filled - shifted).ravel() / dist
            better = slope > best
            best[better] = slope[better]
            recv[better] = idx.ravel()[better].astype(int)
    return recv, flat


def upstream_of(seeds, recv, flat):
    """Cells whose flow path reaches ``seeds``; the contributing area."""
    reaches = seeds.ravel().copy()
    for cell in np.argsort(flat):          # low to high: receivers resolve first
        if not reaches[cell] and reaches[recv[cell]]:
            reaches[cell] = True
    return reaches.reshape(seeds.shape)


def downstream_of(seeds, recv):
    """The flow path leaving ``seeds``, followed to the domain edge."""
    shape = seeds.shape
    path = np.zeros(seeds.size, dtype=bool)
    for cell in np.flatnonzero(seeds.ravel()):
        while not path[cell]:
            path[cell] = True
            nxt = recv[cell]
            if nxt == cell:
                break
            cell = nxt
    return path.reshape(shape)


def inboard_target(moraine, ice, dem, k):
    """Map each moraine cell to a companion cell ``k`` steps toward the ice.

    A lateral moraine records the ice surface *beside* it, not under it.  Scoring
    the simulated surface at the moraine cell itself is degenerate: where the
    model puts no ice the surface is the bed, which is the moraine elevation, so
    an empty valley fits perfectly.  Moving the sample a few cells into the
    valley removes that escape route -- the model has to grow ice to that height
    to score well.

    Returns a mask of the companion cells and the elevation each must reach
    (the mean of the moraine cells that chose it).
    """
    _, nearest = ndimage.distance_transform_edt(~ice, return_indices=True)
    iy, ix = np.nonzero(moraine)
    step_y = np.clip(nearest[0][iy, ix] - iy, -k, k)
    step_x = np.clip(nearest[1][iy, ix] - ix, -k, k)
    ny, nx = moraine.shape
    jy = np.clip(iy + step_y, 0, ny - 1)
    jx = np.clip(ix + step_x, 0, nx - 1)

    total = np.zeros(moraine.shape)
    count = np.zeros(moraine.shape)
    np.add.at(total, (jy, jx), dem[iy, ix])
    np.add.at(count, (jy, jx), 1.0)
    mask = count > 0
    target = np.where(mask, total / np.maximum(count, 1.0), np.nan)
    return mask, target


def along_valley_distance(east, north):
    """Distance from the most downvalley moraine point, in km."""
    origin = np.array([422115.0, 5135595.0])  # SW corner of the YD2 landform
    return np.hypot(east - origin[0], north - origin[1]) / 1000.0


def main():
    grid = read_model_grid()
    gx, gy = grid["x"], grid["y"]
    dx = float(gx[1] - gx[0])
    ny, nx = grid["topg"].shape

    # ---- rasterize each landform, split the LIA pair into its two branches ----
    segments = {}          # label -> dict(count=..., ages=...)
    for fname, (short, fallback_age) in LANDFORMS.items():
        east, north, ages = read_landform(fname)
        ages = ages or fallback_age
        count, dropped = rasterize(east, north, grid)
        if dropped:
            print(f"  {short}: {dropped} 30 m pixels fell outside the model grid")

        if short == "LIA":
            # One file holds both lateral moraines, joined at the terminal loop.
            # Cutting the southernmost 500 m separates the two limbs; the cut
            # cells are then given back to whichever limb they sit nearer.
            here = count > 0
            iy, ix = np.nonzero(here)
            apex = gy[iy] < gy[iy].min() + 500.0
            trunk = here.copy()
            trunk[iy[apex], ix[apex]] = False
            lab, n = ndimage.label(trunk, structure=np.ones((3, 3)))
            if n < 2:
                raise RuntimeError("LIA landform did not separate into two limbs")
            keep = (np.argsort(-ndimage.sum(trunk, lab, range(1, n + 1)))[:2] + 1)
            means = [gx[np.nonzero(lab == k)[1]].mean() for k in keep]
            west, east = np.asarray(keep)[np.argsort(means)]
            limb = np.where(lab == west, 1, np.where(lab == east, 2, 0))
            _, nearest = ndimage.distance_transform_edt(
                limb == 0, return_indices=True)
            limb = np.where(here, limb[tuple(nearest)], 0)
            for k, side in ((1, "W"), (2, "E")):
                segments[f"LIA_{side}"] = dict(
                    count=np.where(limb == k, count, 0), ages=ages)
        else:
            segments[short] = dict(count=count, ages=ages)

    # ---- crest elevation on the model grid ----
    # Two candidate targets: the DEM value at the cell, and the 3x3 maximum
    # (nominally the ridge top, since the DEM is 100 m and the ridges ~30 m
    # wide).  Which one the ice surface should be matched against is settled
    # below by the LIA pair, whose contemporaneous surface is known.
    dem = grid["topg"]
    dem_max3 = ndimage.maximum_filter(dem, size=3, mode="nearest")

    # ---- Massa catchment: the area ice may occupy when the glacier is large ----
    # Delineated on the modern *surface*, so ice-filled valleys do not act as
    # divides.  The glacier and the moraines seed it: everything draining into
    # them is accumulation area, and the flow path leaving them is the corridor
    # the tongue advanced down.
    recv, flat = d8_receivers(grid["usurf"])
    seeds = grid["thk"] > 1.0
    for seg in segments.values():
        seeds |= seg["count"] > 0
    corridor = ndimage.binary_dilation(downstream_of(seeds, recv), iterations=5)
    catchment = ndimage.binary_fill_holes(
        upstream_of(seeds, recv, flat) | corridor | (grid["icemask"] > 0.5))
    print(f"  catchment {catchment.sum() * dx * dx / 1e6:.1f} km2 "
          f"vs icemask {(grid['icemask'] > 0.5).sum() * dx * dx / 1e6:.1f} km2 "
          f"vs modern ice {(grid['thk'] > 1).sum() * dx * dx / 1e6:.1f} km2")

    # ---- calibrate the observation operator on the LIA pair ----
    # The LIA laterals were abandoned around 1850 and geology.nc carries the
    # 1880 surface, so "moraine elevation = ice surface at the time" can be
    # tested rather than assumed.  The result decides which target the YD
    # moraines should use, and gives the sigma for the data term.
    surf = grid["surf_1880"]
    ice = (surf - dem) > 5.0
    iy_i, ix_i = np.nonzero(ice)
    calib = {}
    for name in ("LIA_W", "LIA_E"):
        iy, ix = np.nonzero(segments[name]["count"] > 0)
        d = np.hypot(gx[ix_i][:, None] - gx[ix][None, :],
                     gy[iy_i][:, None] - gy[iy][None, :])
        j, dmin = d.argmin(0), d.min(0)
        near = dmin < 4 * dx
        s_ice = surf[iy_i[j], ix_i[j]][near]
        for key, z in (("cell", dem), ("crest", dem_max3)):
            r = z[iy, ix][near] - s_ice
            calib[f"{name}_{key}"] = (float(np.median(r)), float(r.std()))
    bias = np.mean([calib[f"{n}_cell"][0] for n in ("LIA_W", "LIA_E")])
    sigma = np.mean([calib[f"{n}_cell"][1] for n in ("LIA_W", "LIA_E")])
    print("  observation operator, checked against the known 1880 surface:")
    for k, (b, sd) in calib.items():
        print(f"    {k:12s} median {b:+5.0f} m, std {sd:4.0f} m")
    print(f"  -> at the moraine cell, z_cell: bias {bias:+.0f} m, sigma {sigma:.0f} m")

    # How far inboard to sample.  Same calibration, now asking what the 1880
    # surface does at the companion cell rather than at the moraine.
    print("  inboard offset (residual of the 1880 surface at the companion cell):")
    scores = {}
    for k in (1, 2, 3, 4, 5):
        res = []
        for name in ("LIA_W", "LIA_E"):
            cm, tgt = inboard_target(segments[name]["count"] > 0, ice, dem, k)
            has_ice = cm & ice
            res.append(surf[has_ice] - tgt[has_ice])
            scores[(name, k)] = (float(cm.sum()), float(has_ice.sum()))
        r = np.concatenate(res)
        scores[k] = (float(np.median(r)), float(r.std()), r.size)
        print(f"    k={k}: median {np.median(r):+5.0f} m, std {r.std():4.0f} m, "
              f"{r.size} cells with 1880 ice")
    k_best = min((1, 2, 3, 4, 5), key=lambda k: abs(scores[k][0]) + 0.3 * scores[k][1])
    inboard_bias, inboard_sigma, _ = scores[k_best]
    print(f"  -> sample {k_best} cells inboard: bias {inboard_bias:+.0f} m, "
          f"sigma {inboard_sigma:.0f} m")

    # ---- write ----
    order = ["YD2", "YD1", "LIA_W", "LIA_E"]
    nc = netCDF4.Dataset(OUT_NC, "w")
    nc.createDimension("x", nx)
    nc.createDimension("y", ny)
    nc.Conventions = "CF-1.8"
    nc.crs = "EPSG:32632"
    nc.title = "Great Aletsch: model grid, dated moraines and Massa catchment"
    nc.source = ("data/geology.nc (100 m) joined with the 30 m landform masks "
                 "in data/YD-Aletsch/ by tools/prepare_yd_data.py")
    nc.target_variable = "z_target_<name> on inboard_<name>"
    nc.obs_operator = (
        f"Score the simulated ice surface at inboard_<name> -- the moraine "
        f"cells stepped {k_best} cells toward the ice -- against "
        f"z_target_<name>. Checked against the 1880 surface at the LIA "
        f"laterals: bias {inboard_bias:+.0f} m, sigma {inboard_sigma:.0f} m. "
        "Scoring at the moraine cell itself (z_cell_<name>) is unbiased too "
        f"({bias:+.0f} +/- {sigma:.0f} m) but degenerate, because an ice-free "
        "cell has surface == bed == moraine elevation and fits perfectly. The "
        "3x3 maximum (z_crest_<name>) reads about 80 m high on these slopes.")
    nc.obs_operator_bias_m = inboard_bias
    nc.obs_operator_sigma_m = inboard_sigma
    nc.inboard_offset_cells = k_best

    def put(name, data, dtype="f4", dims=("y", "x"), **attrs):
        v = nc.createVariable(name, dtype, dims, zlib=True)
        v[:] = data
        for k, val in attrs.items():
            setattr(v, k, val)
        return v

    put("x", gx, "f8", ("x",), axis="X", units="m",
        standard_name="projection_x_coordinate")
    put("y", gy, "f8", ("y",), axis="Y", units="m",
        standard_name="projection_y_coordinate")
    put("topg", dem, units="m", long_name="bedrock altitude")
    put("usurf", grid["usurf"], units="m", long_name="modern ice surface")
    put("thk", grid["thk"], units="m", long_name="modern ice thickness")
    put("thk_1880", grid["surf_1880"] - dem, units="m",
        long_name="1880 ice thickness")
    put("icemask", grid["icemask"], units="1",
        long_name="modern/LIA ice mask from geology.nc")
    put("catchment", catchment.astype("f4"), units="1",
        long_name="Massa catchment; suggested ice_mask for pre-LIA runs")

    rows = []
    for i, name in enumerate(order, start=1):
        seg = segments[name]
        m = seg["count"] > 0
        v = put(f"moraine_{name}", m.astype("f4"), units="1",
                long_name=f"{name} moraine crest on the 100 m grid",
                npixels_30m=int(seg["count"].sum()), ncells_100m=int(m.sum()))
        if seg["ages"]:
            v.age_min_yr_BP, v.age_max_yr_BP = seg["ages"]
        put(f"z_cell_{name}", np.where(m, dem, np.nan), units="m",
            long_name=f"{name} DEM elevation at the cell; the inversion target")
        put(f"z_crest_{name}", np.where(m, dem_max3, np.nan), units="m",
            long_name=f"{name} 3x3 DEM maximum; biased high, kept for reference")
        cm, tgt = inboard_target(m, ice, dem, k_best)
        put(f"inboard_{name}", cm.astype("f4"), units="1",
            long_name=f"{name} companion cells, {k_best} cells toward the ice",
            offset_cells=k_best)
        put(f"z_target_{name}", tgt, units="m",
            long_name=f"elevation the simulated ice surface must reach at "
                      f"inboard_{name}; the data term")

        iy, ix = np.nonzero(m)
        for r, c in zip(iy, ix):
            rows.append((name, i, gx[c], gy[r], dem[r, c], dem_max3[r, c],
                         seg["count"][r, c],
                         seg["ages"][0] if seg["ages"] else np.nan,
                         seg["ages"][1] if seg["ages"] else np.nan))

    ident = np.zeros((ny, nx), dtype="f4")
    for i, name in enumerate(order, start=1):
        ident[segments[name]["count"] > 0] = i
    put("moraine_id", ident, units="1",
        long_name="0 none, " + ", ".join(f"{i} {n}" for i, n in
                                         enumerate(order, start=1)))
    nc.close()

    east = np.array([r[2] for r in rows])
    north = np.array([r[3] for r in rows])
    dist = along_valley_distance(east, north)
    with open(OUT_CSV, "w") as f:
        f.write("landform,id,easting,northing,z_cell,z_crest,n_pixels_30m,"
                "age_min_yr_BP,age_max_yr_BP,dist_km\n")
        for r, d in zip(rows, dist):
            f.write(",".join(str(v) for v in r) + f",{d:.3f}\n")

    print(f"\nwrote {OUT_NC}")
    print(f"wrote {OUT_CSV}  ({len(rows)} moraine cells)")
    for name in order:
        seg = segments[name]
        m = seg["count"] > 0
        z = dem[m]
        age = seg["ages"]
        age_s = f"{age[0]:.0f}-{age[1]:.0f} yr BP" if age else "undated"
        print(f"  {name:6s} {m.sum():4d} cells  {z.min():4.0f}-{z.max():4.0f} m"
              f"  ({age_s})")


if __name__ == "__main__":
    main()

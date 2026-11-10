"""Gather the raw inputs for the Argentière experiment into data/Argentiere/sources/.

Every source is public (or already on this machine), and each is cropped to a box
around the upper Arve valley so the repository carries a few MB instead of GB:

* present-day geometry, 2012 and 2021 (DEM, thickness, bed, velocities, stake SMB)
  -- copied from ``../igm-smb-inference/data/argentiere``;
* AlpIce-1 (Kamleitner et al. 2026, Boreas, doi:10.1111/bor.70084; Zenodo
  doi:10.5281/zenodo.21841122, CC-BY 4.0) -- dated ice margins, bedrock and
  erratic exposure ages, individual TCN and 14C ages;
* LIA outlines and the 30 m LIA ice-surface DEM of Reinthaler & Paul (2025, TC,
  doi:10.5194/tc-19-753-2025; Zenodo doi:10.5281/zenodo.14336827, CC-BY 4.0);
* the ice-free Alps bed of Henz et al. (2025, TC, doi:10.5194/tc-19-5913-2025;
  Zenodo doi:10.5281/zenodo.17608701, CC-BY 4.0): AW3D30 minus Cook et al. (2023)
  thickness, 50 m. Used outside the Argentière radar domain.

Run it in an env with geopandas and rasterio (``igm`` here; ``IGEM`` has neither):

    python tools/fetch_argentiere_sources.py [--cache DIR]

``--cache`` is a directory where the Zenodo archives are (or will be) downloaded.
"""

import argparse
import os
import shutil
import subprocess
import zipfile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box

OUT = "data/Argentiere/sources"
PRESENT = "../igm-smb-inference/data/argentiere"
BOX = (314000.0, 5058000.0, 372000.0, 5116000.0)        # EPSG:32632, ~58 x 58 km

ALPICE = "https://zenodo.org/records/21841122/files/DataS6_AlpIce-1_GIS.zip?download=1"
ALPICE_TABLES = ("https://zenodo.org/api/records/21841122/files/"
                 "DataS1%E2%80%93S5_AlpIce-1.zip/content")
LIA_OUTLINES = ("https://zenodo.org/records/14336827/files/"
                "LIA_glacier_outlines_Alps.zip?download=1")
LIA_SURFACE = "/vsicurl/https://zenodo.org/records/14336827/files/LIA_glacier_surface_Alps.tif"
HENZ_BED = "https://zenodo.org/records/17608701/files/input-data-igm.zip?download=1"


def download(url, path):
    if not os.path.exists(path):
        print("  downloading", url)
        subprocess.run(["curl", "-sL", "-m", "1800", "-o", path, url], check=True)
    return path


def unzip(path, dest):
    if not os.path.isdir(dest):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    return dest


def find(root, name):
    for d, _, files in os.walk(root):
        if name in files:
            return os.path.join(d, name)
    raise FileNotFoundError(f"{name} not under {root}")


def crop_raster(src_path, dst_path, bounds=BOX):
    with rasterio.open(src_path) as src:
        w = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
        a = src.read(1, window=w)
        prof = src.profile.copy()
        prof.update(width=a.shape[1], height=a.shape[0], driver="GTiff",
                    transform=src.window_transform(w), compress="deflate",
                    tiled=False, blockysize=min(64, a.shape[0]))
        prof.pop("blockxsize", None)
    with rasterio.open(dst_path, "w", **prof) as dst:
        dst.write(a, 1)
    print(f"  {os.path.basename(dst_path)}: {a.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/argentiere_sources_cache")
    ap.add_argument("--present", default=PRESENT)
    opts = ap.parse_args()
    os.makedirs(opts.cache, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    clip = box(*BOX)

    # ---- present-day geometry -------------------------------------------
    pres = os.path.join(OUT, "present")
    os.makedirs(pres, exist_ok=True)
    for f in ("bedrock_100m_utm32n.tif", "dem_2012_100m_utm32n.tif",
              "dem_2021_100m_utm32n.tif", "thickness_2012_100m_utm32n.tif",
              "thickness_2021_100m_utm32n.tif", "vx_100m_utm32n.tif",
              "vy_100m_utm32n.tif", "SMB_argentiere_2012-2021_utm32N.csv",
              "zbed_arg_measured_UTM32N.gpkg"):
        shutil.copy2(os.path.join(opts.present, f), os.path.join(pres, f))
    print("present-day files copied from", opts.present)

    # ---- AlpIce-1 ---------------------------------------------------------
    gis = unzip(download(ALPICE, os.path.join(opts.cache, "alpice_gis.zip")),
                os.path.join(opts.cache, "alpice_gis"))
    tables = unzip(download(ALPICE_TABLES, os.path.join(opts.cache, "alpice_tables.zip")),
                   os.path.join(opts.cache, "alpice_tables"))
    gpkg = os.path.join(OUT, "alpice1_chamonix.gpkg")
    if os.path.exists(gpkg):
        os.remove(gpkg)
    layers = {
        "ice_margin": "AlpIce1_ice_margin.shp",
        "tcn_event_pt": "AlpIce1_TCN_event_age_geometry_pt.shp",
        "tcn_event_ln": "AlpIce1_TCN_event_age_geometry_ln.shp",
        "tcn_ages": "AlpIce1_TCN_ages.shp",
        "c14_ages": "AlpIce1_14C_ages.shp",
        "c14_event_pt": "AlpIce1_14C_event_age_geometry_pt.shp",
    }
    for layer, name in layers.items():
        g = gpd.read_file(find(gis, name)).to_crs(32632)
        g = g[g.intersects(clip)]
        g.to_file(gpkg, layer=layer, driver="GPKG")
        print(f"  AlpIce-1 {layer}: {len(g)} features")
    for d, _, files in os.walk(tables):
        for f in files:
            if f.startswith("DataS1") or f.startswith("DataS2"):
                shutil.copy2(os.path.join(d, f), os.path.join(OUT, f))
    for d, _, files in os.walk(gis):
        for f in files:
            if f.startswith("001_READ_ME"):
                shutil.copy2(os.path.join(d, f), os.path.join(OUT, "alpice1_" + f))

    # ---- LIA outlines and surface (Reinthaler & Paul 2025) -----------------
    lia = unzip(download(LIA_OUTLINES, os.path.join(opts.cache, "lia_outlines.zip")),
                os.path.join(opts.cache, "lia_outlines"))
    g = gpd.read_file(find(lia, "LIA_Alps_Reinthaler_Paul_2024.shp")).to_crs(32632)
    g = g[g.intersects(clip)]
    g.to_file(os.path.join(OUT, "lia_outlines_chamonix.gpkg"), driver="GPKG")
    print(f"  LIA outlines: {len(g)} glaciers")
    surf = os.path.join(OUT, "lia_surface_30m_chamonix.tif")
    if True:                                   # re-crop whenever BOX changes
        os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
        os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "4194304")
        os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "8")
        os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")
        crop_raster(LIA_SURFACE, surf)

    # ---- ice-free bed of Henz et al. (2025) --------------------------------
    henz = unzip(download(HENZ_BED, os.path.join(opts.cache, "henz_input.zip")),
                 os.path.join(opts.cache, "henz_input"))
    bed = os.path.join(OUT, "bed_henz2025_50m_chamonix.tif")
    if True:
        crop_raster('NETCDF:"' + find(henz, "aw3d30_50m_noice_cook.nc") + '":topg', bed)
    with rasterio.open(bed) as r:
        a = r.read(1)
        print(f"  bed range {np.nanmin(a):.0f}-{np.nanmax(a):.0f} m, crs {r.crs}")
    print("done:", OUT)


if __name__ == "__main__":
    main()

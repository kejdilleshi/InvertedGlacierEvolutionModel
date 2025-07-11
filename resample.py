#!/usr/bin/env python3
"""
downsample_bedrock.py – reduce 100 m bedrock grid to 200 m resolution
"""

import netCDF4
import torch
import torch.nn.functional as F
from pathlib import Path

# ----------------------------------------------------------------------
# 1) Load the 100 m grid
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
src_path = Path("bedrock.nc").expanduser()

with netCDF4.Dataset(src_path) as nc_in:
    topo_raw = nc_in.variables["topg"][:]               # shape (ny, nx)
    try:
        fill_val = nc_in.variables["topg"]._FillValue   # NaN substitute if present
    except AttributeError:
        fill_val = None

Z_topo = torch.as_tensor(topo_raw, dtype=torch.float32, device=device)

# Replace fill values (if any) with NaN so we can do nan-aware averaging
if fill_val is not None:
    Z_topo[Z_topo == fill_val] = float("nan")

# ----------------------------------------------------------------------
# 2) Down-sample from 100 m → 200 m (factor 2 in X and Y)
# ----------------------------------------------------------------------
def downsample_mean_2x(x: torch.Tensor) -> torch.Tensor:
    """
    Nan-aware 2 × 2 averaging.
    x – 2-D tensor [H, W] on any device.
    Returns tensor of shape [H//2, W//2] on the same device.
    """
    # If dimensions are odd, drop the last row/col
    h, w = x.shape
    if h % 2 == 1:
        x = x[:-1, :]
    if w % 2 == 1:
        x = x[:, :-1]

    # Build validity mask and zero-fill NaNs
    valid = (~torch.isnan(x)).float()
    x_filled = torch.nan_to_num(x, nan=0.0)

    # Reshape to blocks and sum
    # After reshape: [H/2, 2, W/2, 2]  -> permute to group 2×2 blocks together
    x_blocks = x_filled.reshape(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3)
    v_blocks = valid.reshape(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3)

    summed   = x_blocks.sum(dim=(-1, -2))         # sum of values in each block
    counts   = v_blocks.sum(dim=(-1, -2))         # how many non-NaN in block
    out      = summed / torch.where(counts == 0, torch.ones_like(counts), counts)
    out[counts == 0] = float("nan")               # keep NaN where block empty
    return out

Z_topo_200m = downsample_mean_2x(Z_topo).to("cpu")  # move back to CPU for I/O

# ----------------------------------------------------------------------
# 3) Write the 200 m grid to a new NetCDF file
# ----------------------------------------------------------------------
dst_path = Path("bedrock_200m.nc")
with netCDF4.Dataset(dst_path, "w", format="NETCDF4") as nc_out:
    ny, nx = Z_topo_200m.shape
    nc_out.createDimension("y", ny)
    nc_out.createDimension("x", nx)

    var = nc_out.createVariable("topg", "f4", ("y", "x"), zlib=True, complevel=4)
    var[:] = Z_topo_200m.numpy()
    var.units = "meters"        # change if your source uses something else

print(f"Coarser grid written to {dst_path.resolve()}")

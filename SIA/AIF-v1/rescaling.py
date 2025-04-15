import xarray as xr
import numpy as np
from scipy.ndimage import zoom

# Load the original NetCDF file
ds = xr.open_dataset("input-100.nc")

# Extract the topography variable
topg = ds['topg']

# Get scale factor (100m → 500m ⇒ factor 1/5)
scale_factor = 1 / 5

# Apply zoom with averaging (order=1 → bilinear interpolation; order=0 for nearest, etc.)
# If you want proper averaging, you can use block reduce instead (see alternative below)
topg_coarse = xr.DataArray(
    zoom(topg.values, zoom=(scale_factor, scale_factor), order=1),
    dims=("y", "x"),
    attrs=topg.attrs
)

# Create new coordinates assuming uniform spacing
new_y = np.linspace(topg['y'].values[0], topg['y'].values[-1], topg_coarse.shape[0])
new_x = np.linspace(topg['x'].values[0], topg['x'].values[-1], topg_coarse.shape[1])

topg_coarse = topg_coarse.assign_coords(y=new_y, x=new_x)
topg_coarse.name = "topg"

# Create new dataset
ds_coarse = xr.Dataset({"topg": topg_coarse})

# Save to new NetCDF file
ds_coarse.to_netcdf("output-500.nc")

print("Resampling complete. Saved as 'output-500.nc'.")



import xarray as xr

# Load the datasets
ds_topg = xr.open_dataset("AIF-v1/output-500.nc")              # Has 'topg' and desired grid
ds_obs = xr.open_dataset("AIF-v1/obs_Ehler_500m.nc")           # Has 'mx_thk_obs'

# Extract only mx_thk_obs
mx_thk_obs = ds_obs["mx_thk_obs"]

# Reindex obs to match topg dimensions and coordinates
mx_thk_obs_aligned = mx_thk_obs.interp_like(ds_topg, method="nearest")

# Add the aligned mx_thk_obs to a copy of ds_topg
combined = ds_topg.copy()
combined["mx_thk_obs"] = mx_thk_obs_aligned

# Save to NetCDF
combined.to_netcdf("combined_500m.nc")

print("✅ Successfully combined into 'combined_500m.nc' with variables:", list(combined.data_vars))


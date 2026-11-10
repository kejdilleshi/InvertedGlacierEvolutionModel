import matplotlib
matplotlib.use('Agg')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource

ds = xr.open_dataset('data/geology.nc')
dem_1980 = ds['surf_1980'].values
dem_2017 = ds['surf_2017'].values
topg     = ds['topg'].values

x = ds['x'].values
y = ds['y'].values
dx = float(x[1] - x[0])  # 100 m

ls   = LightSource(azdeg=315, altdeg=45)
cmap = plt.cm.terrain
vmin, vmax = 800, 4200
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# km extent for axes
ext = [x.min()/1000, x.max()/1000, y.min()/1000, y.max()/1000]

fig, axes = plt.subplots(1, 2, figsize=(12, 8), facecolor='#1a1a2e',
                          constrained_layout=True)

titles = ['1980', '2017']
dems   = [dem_1980, dem_2017]

for ax, dem, title in zip(axes, dems, titles):
    ax.set_facecolor('#0d0d1a')

    # Hillshaded RGB image
    shade = ls.hillshade(dem, vert_exag=3, dx=dx, dy=dx)
    rgba  = cmap(norm(dem))
    rgb   = rgba[..., :3] * (0.4 + 0.6 * shade[..., np.newaxis])
    rgb   = np.clip(rgb, 0, 1)

    ax.imshow(rgb, origin='lower', extent=ext, aspect='equal')

    # Ice thickness → glacier mask
    ice_thk = dem - topg
    glacier  = ice_thk > 1.0  # 1 m threshold

    # Transparent blue overlay on glacier area
    blue_overlay = np.zeros((*dem.shape, 4))  # RGBA
    blue_overlay[glacier] = [0.3, 0.6, 1.0, 0.35]
    ax.imshow(blue_overlay, origin='lower', extent=ext, aspect='equal')

    # Glacier outline from ice thickness mask
    ax.contour(x/1000, y/1000, ice_thk, levels=[1.0],
               colors='white', linewidths=0.9, alpha=0.85)

    ax.set_title(title, color='white', fontsize=20, fontweight='bold', pad=8)
    ax.tick_params(colors='#bbbbcc', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#33334d')
    ax.set_xlabel('Easting (km)', color='#aaaacc', fontsize=10)

axes[0].set_ylabel('Northing (km)', color='#aaaacc', fontsize=10)
axes[1].set_yticklabels([])

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                    fraction=0.025, pad=0.02, shrink=0.85)
cbar.set_label('Elevation (m a.s.l.)', color='white', fontsize=11, labelpad=8)
cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=9)

fig.suptitle('Great Aletsch Glacier — Surface DEM', color='white',
             fontsize=18, fontweight='bold')

plt.savefig('aletsch_dem_hillshade.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved: aletsch_dem_hillshade.png")

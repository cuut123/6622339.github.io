import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import warnings
from cartopy.io import DownloadWarning
warnings.filterwarnings("ignore", category=DownloadWarning)

sns.set_style("white")

# Figure output config
FIG_DIR = "../results"
FIG_NAME = "fig1c.pdf"

# Load and preprocess precipitation data
# Load distance to land and precipitation frequency datasets
dist2land  = xr.open_dataset("../data/dist2land_files/mswep_790000_dist2land.nc").dist2land
precip_ds = xr.open_dataset("../data/coastal_rainfall/mswep_precip_count_100km.nc")
precip_freq = precip_ds.precip_frequency

# Compute total precipitation count over time
precip_sum = precip_freq.sum(dim='time')
precip_sum_masked = np.ma.masked_equal(precip_sum, 0)  # Mask out zeros

# Setup map for plotting
fig, ax = plt.subplots(
    figsize=(12, 6),
    subplot_kw={"projection": ccrs.Robinson()},
    dpi=400
)

# Set global map extent
ax.set_extent([-180, 180, -60, 60])

# Plot precipitation frequency
contour = ax.contourf(
    precip_sum.lon.values,
    precip_sum.lat.values,
    precip_sum_masked,
    levels=np.arange(0, 201, 5),
    cmap='Blues',
    extend='max',
    transform=ccrs.PlateCarree(),
    zorder=3
)

# Add map features
# ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='gray', linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='#D4D9DD')
ax.add_feature(cfeature.OCEAN, facecolor='white')
ax.coastlines(resolution='110m', linewidth=0.5, zorder=1)

# Add colorbar
# Create axes for colorbar (left, bottom, width, height)
cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.04])
cbar = plt.colorbar(contour, cax=cbar_ax, orientation='horizontal')

# Customize colorbar
cbar.set_label('Rainfall frequency', fontsize=14)
cbar.ax.tick_params(labelsize=14)

# Load and plot TC tracks from MSWEP
# Load and filter TC data
df = pd.read_csv("../data/mswep/area_790000/result_r500.csv", na_values=[''], keep_default_na=False)
df = df[df['NEAR_FID'] != -1]
df = df[df['SEASON'] <= 2023]
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
df['MONTH'] = df['ISO_TIME'].dt.month
df = df[df['NEAR_DIST'] >= -100]  # Within 100 km inland

# Group by TC ID and draw track lines
for sid, group in df.groupby('SID'):
    group = group.sort_values('ISO_TIME')
    group = group.dropna(subset=['USA_LAT', 'USA_LON', 'ISO_TIME'])

    lons = group['USA_LON'].values
    lats = group['USA_LAT'].values
    times = group['ISO_TIME'].values

    if len(lons) < 2:
        continue

    # Step 1
    time_deltas = np.diff(times).astype('timedelta64[h]').astype(int)
    time_split_indices = np.where(time_deltas > 3)[0] + 1

    # Step 2
    signs = np.sign(lons)
    sign_split_indices = np.where(signs[:-1] != signs[1:])[0] + 1

    # Step 3
    all_splits = np.unique(np.concatenate([time_split_indices, sign_split_indices]))

    # Step 4
    lon_segments = np.split(lons, all_splits)
    lat_segments = np.split(lats, all_splits)

    # Step 5
    for slon, slat in zip(lon_segments, lat_segments):
        if len(slon) >= 2:
            ax.plot(
                slon, slat,
                color='lightgray',
                linewidth=0.4,
                alpha=0.3,
                transform=ccrs.PlateCarree(),
                zorder=2
            )

    # Step 6
    na_points = [(-100, 40), (-100, 13), (-70, 13), (-70, 40), (-100, 40)]
    xs, ys = zip(*na_points)
    ax.plot(xs, ys, color='#BF453D', linewidth=1, transform=ccrs.PlateCarree(), zorder=4)

    na_points = [(77, 8), (77, 28), (95, 28), (95, 8), (77, 8)]
    xs, ys = zip(*na_points)
    ax.plot(xs, ys, color='#BF453D', linewidth=1, transform=ccrs.PlateCarree(), zorder=4)

    na_points = [(103.5, 8), (103.5, 33), (125, 33), (125, 8), (103.5, 8)]
    xs, ys = zip(*na_points)
    ax.plot(xs, ys, color='#BF453D', linewidth=1, transform=ccrs.PlateCarree(), zorder=4)

# Save figure
import os
os.makedirs(FIG_DIR, exist_ok=True)

fig_path = os.path.join(FIG_DIR, FIG_NAME)
plt.savefig(fig_path, dpi=400, bbox_inches="tight")
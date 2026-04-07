import pandas as pd
import numpy as np
import xarray as xr
import os
from scipy import stats
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings
from cartopy.io import DownloadWarning
warnings.filterwarnings("ignore", category=DownloadWarning)

# Common constants
DISTANCES = [500]
PLACEHOLDER_BAND_KM = 30
SPLIT_INDEX = 22
GAUSSIAN_SIGMA = 1

# Path to distance-to-land dataset
DIST2LAND_PATH = "../data/dist2land_files/mswep_790000_dist2land.nc"
DIST2LAND_VAR = 'dist2land'
OUTDIR = "../results"
os.makedirs(OUTDIR, exist_ok=True)
DIST2LAND_CONTOUR_LEVELS = [-100, 500]

# Font / figure settings
TICK_LABELSIZE = 26
FIGSIZE = (9, 6)
FIG_DPI = 400

# Region configurations
REGIONS = {
    "WNA": {
        "csv_pattern": "../data/coastal_rainfall/WNA_0.5_r500/filtered_df_WNA_r500.csv",
        "freq_nc_pattern": "../data/coastal_rainfall/WNA_0.5_r500/WNA_pre_frequncy_r500.nc",
        "extent": [-103, -69, 10, 40],
        "xticks": list(range(-100, -69, 10)),
        "yticks": list(range(15, 36, 5)),
        "outfile": "fig1d.pdf",
    },
    "BOB": {
        "csv_pattern": "../data/coastal_rainfall/BOB_0.5_r500/filtered_df_BOB_r500.csv",
        "freq_nc_pattern": "../data/coastal_rainfall/BOB_0.5_r500/BOB_pre_frequncy_r500.nc",
        "extent": [74, 96, 3, 27],
        "xticks": list(range(75, 96, 10)),
        "yticks": list(range(5, 26, 5)),
        "outfile": "fig1e.pdf",
    },
    "WNP": {
        "csv_pattern": "../data/coastal_rainfall/WNP_0.5_r500/filtered_df_WNP_r500.csv",
        "freq_nc_pattern": "../data/coastal_rainfall/WNP_0.5_r500/WNP_pre_frequncy_r500.nc",
        "extent": [100, 133, 7, 33],
        "xticks": list(range(105, 133, 10)),
        "yticks": list(range(10, 33, 5)),
        "outfile": "fig1f.pdf",
    },
}

# Helper functions
def trendline(series: pd.Series):
    """Compute linear trend slope, p-value, and R²."""
    y = pd.Series(series).dropna()
    x = np.arange(len(y))
    res = stats.linregress(x, y)
    return res.slope, res.pvalue, res.rvalue ** 2

def plot_clipped_polyline(
    ax,
    lon0, lat0,
    lon1, lat1,
    dist2land: xr.DataArray,
    dist_min=-100,
    dist_max=500,
    npts=600,
    **plot_kwargs
):
    """
    Plot a line segment clipped by distance-to-land thresholds.
    The line is interpolated at `npts` points, and only the portion
    where dist_min ≤ distance ≤ dist_max is drawn.
    """
    t = np.linspace(0, 1, npts)
    lons = lon0 + t * (lon1 - lon0)
    lats = lat0 + t * (lat1 - lat0)

    dist_vals = dist2land.interp(
        lon=xr.DataArray(lons, dims="points"),
        lat=xr.DataArray(lats, dims="points"),
        method="linear"
    ).values

    mask = (dist_vals >= dist_min) & (dist_vals <= dist_max)

    if np.sum(mask) < 2:
        return

    ax.plot(
        lons[mask],
        lats[mask],
        transform=ccrs.PlateCarree(),
        **plot_kwargs
    )

def plot_clipped_latline(
    ax,
    lat,
    lon_min,
    lon_max,
    dist2land: xr.DataArray,
    dist_min=-100,
    dist_max=500,
    npts=800,
    **plot_kwargs
):
    """
    Plot a latitude line clipped by distance-to-land thresholds.
    """
    lons = np.linspace(lon_min, lon_max, npts)
    lats = np.full_like(lons, lat)

    dist_vals = dist2land.interp(
        lon=xr.DataArray(lons, dims="points"),
        lat=xr.DataArray(lats, dims="points"),
        method="linear"
    ).values

    mask = (dist_vals >= dist_min) & (dist_vals <= dist_max)

    if np.sum(mask) < 2:
        return

    ax.plot(
        lons[mask],
        lats[mask],
        transform=ccrs.PlateCarree(),
        **plot_kwargs
    )

def plot_clipped_lonline(
    ax,
    lon,
    lat_min,
    lat_max,
    dist2land: xr.DataArray,
    dist_min=-100,
    dist_max=500,
    npts=800,
    **plot_kwargs
):
    """
    Plot a longitude line clipped by distance-to-land thresholds.
    """
    lats = np.linspace(lat_min, lat_max, npts)
    lons = np.full_like(lats, lon)

    dist_vals = dist2land.interp(
        lon=xr.DataArray(lons, dims="points"),
        lat=xr.DataArray(lats, dims="points"),
        method="linear"
    ).values

    mask = (dist_vals >= dist_min) & (dist_vals <= dist_max)

    if np.sum(mask) < 2:
        return

    ax.plot(
        lons[mask],
        lats[mask],
        transform=ccrs.PlateCarree(),
        **plot_kwargs
    )

def plot_region_map(region_name: str,
                    distance: int,
                    dist2land: xr.DataArray,
                    csv_path: str,
                    freq_nc_path: str,
                    extent: list,
                    xticks: list,
                    yticks: list):
    """
    Map plot for a single region and distance threshold.
    - Filled contours: early (Blues), late (Reds)
    - Lines: distance-to-land contours (land mask and 500 km)
    - Points: TC locations from CSV
    """
    # Load TC CSV for scatter layer
    df = pd.read_csv(csv_path)

    # Load frequency field (already processed & coarsened)
    ds = xr.open_dataset(freq_nc_path)
    precip = ds['precipitation']

    # --- early / late split ---
    early_raw = precip.isel(time=slice(None, SPLIT_INDEX)).sum(dim='time').values
    late_raw  = precip.isel(time=slice(SPLIT_INDEX, None)).sum(dim='time').values

    # --- Gaussian smoothing ---
    early = gaussian_filter(early_raw, sigma=GAUSSIAN_SIGMA)
    late  = gaussian_filter(late_raw,  sigma=GAUSSIAN_SIGMA)

    # Figure & axes
    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={"projection": ccrs.PlateCarree()}, dpi=FIG_DPI)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Base features
    ax.add_feature(cfeature.LAND, facecolor='#D4D9DD', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)

    # Grids / ticks
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.xaxis.set_tick_params(
        which='major',
        direction='out',
        length=8,
        width=1.2,
        labelsize=TICK_LABELSIZE,
        bottom=True,
        top=False
    )

    ax.yaxis.set_tick_params(
        which='major',
        direction='out',
        length=8,
        width=1.2,
        labelsize=TICK_LABELSIZE,
        left=True,
        right=False
    )

    ax.minorticks_off()
    ax.spines['geo'].set_linewidth(1.2)

    # Filled contours: early (Blues), late (Reds)
    c1 = ax.contourf(precip['lon'], precip['lat'], early,
                     levels=range(1, 21, 4), cmap='Blues',
                     extend='max', alpha=0.8, zorder=1)

    c2 = ax.contourf(precip['lon'], precip['lat'], late,
                     levels=range(1, 21, 4), cmap='Reds',
                     extend='max', alpha=0.4, zorder=1)

    # Land mask & 500-km lines
    ax.contour(dist2land['lon'], dist2land['lat'], dist2land,
               levels=DIST2LAND_CONTOUR_LEVELS,
               colors='black', linewidths=1.5, zorder=2)

    # Additional clipped lines (custom geographic features)
    plot_clipped_polyline(
        ax,
        lon0=-86, lat0=14.2,
        lon1=-75, lat1=22,
        dist2land=dist2land,
        dist_min=-100,
        dist_max=500,
        color='black',
        linewidth=1.5,
        linestyle='-',
        zorder=4
    )

    plot_clipped_polyline(
        ax,
        lon0=102, lat0=15,
        lon1=118, lat1=1,
        dist2land=dist2land,
        dist_min=-100,
        dist_max=500,
        color='black',
        linewidth=1.5,
        linestyle='-',
        zorder=4
    )

    plot_clipped_latline(
        ax,
        lat=37.0,
        lon_min=-105,
        lon_max=-60,
        dist2land=dist2land,
        dist_min=-100,
        dist_max=500,
        color='black',
        linewidth=1.5,
        linestyle='-',
        zorder=4
    )

    plot_clipped_latline(
        ax,
        lat=32.0,
        lon_min=101,
        lon_max=135,
        dist2land=dist2land,
        dist_min=-100,
        dist_max=500,
        color='black',
        linewidth=1.5,
        linestyle='-',
        zorder=4
    )

    plot_clipped_lonline(
        ax,
        lon=93.0,
        lat_min=extent[2],
        lat_max=extent[3],
        dist2land=dist2land,
        dist_min=-100,
        dist_max=500,
        color='black',
        linewidth=1.5,
        linestyle='-',
        zorder=4
    )

    plot_clipped_lonline(
        ax,
        lon=77.5,
        lat_min=extent[2],
        lat_max=extent[3],
        dist2land=dist2land,
        dist_min=-100,
        dist_max=500,
        color='black',
        linewidth=1.5,
        linestyle='-',
        zorder=4
    )

    # TC points
    if {'USA_LON', 'USA_LAT'}.issubset(df.columns):
        ax.scatter(df['USA_LON'], df['USA_LAT'],
                   color='grey', s=5, alpha=0.1, zorder=3)

    plt.tight_layout()

    out_path = os.path.join(OUTDIR, REGIONS[region_name]["outfile"])
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

# Colorbar function
def plot_colorbar():
    """
    Create a standalone colorbar figure for the early (blue) and late (red)
    rainfall frequency maps. Saves as 'fig1def_Colorbar.pdf'.
    """
    levels = np.arange(1, 25, 4)

    def create_alpha_colormap(cmap_name, alpha_val):
        """Create a colormap with a fixed alpha transparency."""
        base = plt.get_cmap(cmap_name)
        colors = base(np.linspace(0, 1, base.N))
        colors[:, -1] = alpha_val
        return ListedColormap(colors)

    cmap_blues_alpha = create_alpha_colormap('Blues', 0.8)
    cmap_reds_alpha  = create_alpha_colormap('Reds',  0.4)

    norm = mpl.colors.BoundaryNorm(
        boundaries=levels,
        ncolors=256,
        extend='max'
    )

    fig = plt.figure(figsize=(6, 3), dpi=300)

    # Colorbar 1: pre-2001 (blue)
    cax1 = fig.add_axes([0.1, 0.60, 0.8, 0.08])
    cb1 = mpl.colorbar.ColorbarBase(
        cax1,
        cmap=cmap_blues_alpha,
        norm=norm,
        orientation='horizontal',
        ticks=levels,
        extend='max'
    )
    cb1.set_label("Rainfall frequency (pre-2001)", fontsize=16)
    cb1.ax.tick_params(labelsize=16)

    # Colorbar 2: post-2001 (red)
    cax2 = fig.add_axes([0.1, 0.20, 0.8, 0.08])
    cb2 = mpl.colorbar.ColorbarBase(
        cax2,
        cmap=cmap_reds_alpha,
        norm=norm,
        orientation='horizontal',
        ticks=levels,
        extend='max'
    )
    cb2.set_label("Rainfall frequency (post-2001)", fontsize=16)
    cb2.ax.tick_params(labelsize=16)

    out_path = os.path.join(OUTDIR, "fig1def_Colorbar.pdf")
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

# Main program
def main():
    # Load distance-to-land dataset
    ds = xr.open_dataset(DIST2LAND_PATH)
    dist2land = ds[DIST2LAND_VAR]

    # Generate maps for each region
    for region_name, cfg in REGIONS.items():
        # Patterns already contain the full path (no need to format with distance)
        csv_path = cfg['csv_pattern']
        freq_nc_path = cfg['freq_nc_pattern']

        plot_region_map(
            region_name=region_name,
            distance=500,                     # fixed distance, consistent with file names
            dist2land=dist2land,
            csv_path=csv_path,
            freq_nc_path=freq_nc_path,
            extent=cfg['extent'],
            xticks=cfg['xticks'],
            yticks=cfg['yticks'],
        )

    # Generate the separate colorbar figure
    plot_colorbar()

if __name__ == "__main__":
    main()
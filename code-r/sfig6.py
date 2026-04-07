import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import linregress, norm
# ---- map dependencies ----
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import geopandas as gpd
from shapely.geometry import Polygon
import regionmask
from matplotlib import cm
from matplotlib.colors import ListedColormap
import warnings
from cartopy.io import DownloadWarning
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=DownloadWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# =============================================================================
# 0) Paths: codes/ -> project root -> data/ & results/
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../codes
PROJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))       # project root
DATA_DIR = os.path.join(PROJ_DIR, "data")
FIG_DIR  = os.path.join(PROJ_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# 0.1) Fonts: Arial from ../data/fonts/Arial (fallback if missing)
# =============================================================================
FONT_DIR = os.path.join(DATA_DIR, "fonts", "Arial")
for fn in ["ARIAL.TTF", "ARIALBD.TTF", "ARIALI.TTF", "ARIALBI.TTF",
           "Arial.ttf", "Arialbd.ttf", "Ariali.ttf", "Arialbi.ttf"]:
    p = os.path.join(FONT_DIR, fn)
    if os.path.isfile(p):
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["axes.formatter.use_mathtext"] = True


# =============================================================================
# 1) Output file names (EDIT ONLY HERE if you want to rename)
# =============================================================================
OUT_NAMES = {
    # --- Bar plots ---
    "bar_fcs":    "sfig6a.png",
    "bar_cci":    "sfig6b.png",
    "bar_legend": "sfig6_legend.png",

    # --- Maps (impervious ratio) ---
    # five regions for maps (you can rename to sfig5c... etc)
    "map_WNA":    "sfig6c.png",
    "map_WNP":    "sfig6d.png",
    "map_BOB":    "sfig6e.png",
    "map_NAUS":   "sfig6f.png",
    "map_EMOZ":   "sfig6g.png",
    # --- Map colorbar ---
    "map_cbar":   "sfig6_colorbar.png",
}


# =============================================================================
# Part A: Bar plots (FCS / CCI)
# =============================================================================

# -------------------------
# Shared parameters
# -------------------------
FONTSIZE = 18
regions = ["USA_East", "South_Asia", "East_Asia"]
thresholds = [0.20, 0.35]
distance = "200km"
end_year = 2022

region_map = {
    "USA_East": "WNA",
    "South_Asia": "BOB",
    "East_Asia": "WNP"
}

# -------------------------
# Dataset configurations (paths are relative to ../data)
# -------------------------
DATASETS = {
    "FCS": {
        "root_dir": os.path.join(DATA_DIR, "sfig6_data", "FCS30V2_offshore_distance_results_K_tree"),
        "start_year": 1985,
        "yticks": [0, 2, 4, 6, 8, 10],
        "ylim": (0, 10),
        "sig_offset": 0.1,
        "out_png": os.path.join(FIG_DIR, OUT_NAMES["bar_fcs"]),
    },
    "CCI": {
        "root_dir": os.path.join(DATA_DIR, "sfig6_data", "CCI_offshore_distance_results_K_tree"),
        "start_year": 1992,
        "yticks": [0, 4, 8, 12, 16],
        "ylim": (0, 16),
        "sig_offset": 0.2,
        "out_png": os.path.join(FIG_DIR, OUT_NAMES["bar_cci"]),
    }
}

def mk_pvalue(y):
    """Standard Mann–Kendall trend test (two-sided p-value). NaNs are dropped."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = y.size
    if n < 3:
        return 1.0

    s = 0
    for k in range(n - 1):
        s += np.sign(y[k + 1:] - y[k]).sum()

    unique, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return 1.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    return float(2 * (1 - norm.cdf(abs(z))))

def calc_trend(file_path, start_year):
    """
    Read Excel and compute:
    - slope and stderr: OLS (linregress)
    - p-value: MK
    Assumption: column 'Year' exists; target variable is 2nd column.
    """
    df = pd.read_excel(file_path)
    df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]

    y = df.iloc[:, 1].astype(float).values
    x = df["Year"].astype(int).values

    m = ~np.isnan(y)
    x = x[m]
    y = y[m]

    if len(y) < 3:
        return 0.0, 0.0, 1.0

    slope, _, _, _, stderr = linregress(x, y)
    p_mk = mk_pvalue(y)
    return slope, stderr, p_mk

def run_one_dataset(cfg, dataset_name):
    root_dir = cfg["root_dir"]
    start_year = cfg["start_year"]

    all_slopes, all_errs, all_pvals = [], [], []

    for region in regions:
        slopes, errs, pvals = [], [], []
        for thr in thresholds:
            fname = f"{region}_buffer_{distance}_{thr}.xlsx"
            fpath = os.path.join(root_dir, fname)

            if not os.path.exists(fpath):
                print(f"[{dataset_name}] Missing file: {fpath}")
                slope, err, p = 0.0, 0.0, 1.0
            else:
                slope, err, p = calc_trend(fpath, start_year)

            # keep your original rule: per-decade + sign
            slopes.append(-slope * 10.0)
            errs.append(err * 19.6)  # 95% CI for decade slope
            pvals.append(p)

        all_slopes.append(slopes)
        all_errs.append(errs)
        all_pvals.append(pvals)

    # Plot
    x = np.array([0, 0.4, 0.8])
    bar_width = 0.15
    offsets = [-bar_width / 2, bar_width / 2]
    colors = ['#3B83B4', '#BF453D']
    labels = [
        "Threshold of impervious area ratio = 0.20",
        "Threshold of impervious area ratio = 0.35",
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=400)

    for i_thr, _ in enumerate(thresholds):
        slopes_i = [all_slopes[r][i_thr] for r in range(len(regions))]
        errs_i   = [all_errs[r][i_thr] for r in range(len(regions))]
        pvals_i  = [all_pvals[r][i_thr] for r in range(len(regions))]

        bars = ax.bar(
            x + offsets[i_thr], slopes_i, width=bar_width, yerr=errs_i,
            capsize=5, label=labels[i_thr], color=colors[i_thr],
            error_kw={"elinewidth": 2}
        )

        for j, bar in enumerate(bars):
            height = bar.get_height()
            err = errs_i[j]
            p = pvals_i[j]
            y_pos = height + err - cfg["sig_offset"]

            marker = "**" if p <= 0.05 else ("*" if p <= 0.1 else "")
            if marker:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, y_pos, marker,
                    ha="center", va="bottom", color="black",
                    fontsize=FONTSIZE, fontweight="bold"
                )

    ax.set_xticks(x)
    ax.set_xticklabels([region_map[r] for r in regions], fontsize=FONTSIZE)
    ax.set_yticks(cfg["yticks"])
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.axhline(0, color="black", linewidth=1)

    ax.set_ylabel("Trend of landward distance\nof coastal city (km decade$^{-1}$)",
                  fontsize=FONTSIZE)

    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax.set_ylim(*cfg["ylim"])
    plt.tight_layout()

    out_png = cfg["out_png"]
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Part B: Standalone legend (transparent background)
# =============================================================================
def make_legend_png(out_path):
    from matplotlib.patches import Patch

    labels = [
        "Threshold of impervious area ratio = 0.20",
        "Threshold of impervious area ratio = 0.35",
    ]
    colors = ['#3B83B4', '#BF453D']
    handles = [Patch(facecolor=colors[i]) for i in range(len(labels))]

    fig_legend = plt.figure(figsize=(12, 1), dpi=400)
    fig_legend.legend(handles, labels, loc="center", ncol=2, fontsize=18, frameon=False)
    plt.axis("off")

    plt.savefig(out_path, bbox_inches="tight", dpi=400, transparent=True)
    plt.close(fig_legend)

# =============================================================================
# Part C: Impervious area ratio maps (5 regions)
# =============================================================================

# --- Inputs (relative to ../data) ---
CITY_RATIO_DIR = os.path.join(DATA_DIR, "sfig6_data", "FCS30D_city_ratio_V2")

LAND_SHP   = os.path.join(DATA_DIR, "shp", "land_790000.shp")
SHP_C200   = os.path.join(DATA_DIR, "shp", "c200.shp")
SHP_C100   = os.path.join(DATA_DIR, "shp", "c100.shp")

ADMIN1_SHP = os.path.join(DATA_DIR, "shp", "ne_10m_admin_1_states_provinces",
                          "ne_10m_admin_1_states_provinces.shp")
COAST_SHP  = os.path.join(DATA_DIR, "shp", "ne_10m_coastline", "ne_10m_coastline.shp")

# --- Region definitions ---
country_bounds = {
    "WNA": [(-105, 30), (-98, 18), (-92, 16.6), (-90, 15), (-86, 14.2), (-75, 22), (-68, 40), (-87, 40)],
    "WNP": [(103.5, 8), (103.5, 33), (125, 33), (125, 8)],
    "BOB": [(77, 28), (77, 8), (93, 8), (93, 28)],
    "Northern_Australia": [(111, -25), (155, -25), (155, -10), (110, -10)],
    "Eastern_mozambique": [(43, -10), (25, -10), (22, -25), (40, -25)],
}

plot_extents = {
    "WNA": [-105, -68, 14.2, 40],
    "WNP": [103.5, 125, 8, 33],
    "BOB": [77, 100, 8, 28],
    "Northern_Australia": [110, 155, -25, -10],
    "Eastern_mozambique": [22, 43, -25, -10],
}

# Tick / annotation sizes (region-specific)
TICK_FONT_BY_REGION = {"WNA": 18, "WNP": 18, "BOB": 18, "Northern_Australia": 21, "Eastern_mozambique": 21}
ANNO_FONT_BY_REGION = {"WNA": 14, "WNP": 14, "BOB": 14, "Northern_Australia": 17, "Eastern_mozambique": 17}

DEFAULT_TICK_FONTSIZE = 18
DEFAULT_ANNO_FONTSIZE = 14

def add_topleft_text(ax, text, fontsize):
    ax.annotate(
        text,
        xy=(0, 1), xycoords="axes fraction",
        xytext=(6, -6), textcoords="offset points",
        ha="left", va="top",
        fontsize=fontsize, color="black", zorder=200
    )

def plot_city_ratio(ds, region_name, polygon_coords, plot_extent,
                    year, land_gdf, gdf_huan, coastline_gdf, out_png):

    varname = "city_ratio"
    if varname not in ds.data_vars:
        print(f"[WARN] Variable '{varname}' not found. Skip {region_name} {year}")
        return

    fs_tick = TICK_FONT_BY_REGION.get(region_name, DEFAULT_TICK_FONTSIZE)
    fs_anno = ANNO_FONT_BY_REGION.get(region_name, DEFAULT_ANNO_FONTSIZE)

    city_ratio = ds[varname]

    # Land mask
    land_polygons = list(land_gdf.geometry)
    land_region = regionmask.Regions(land_polygons, names=["land"] * len(land_polygons))
    land_mask = land_region.mask(city_ratio)
    city_ratio_land = city_ratio.where(land_mask.notnull())

    # Research polygon
    poly = Polygon(polygon_coords)
    poly_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    city_in_poly = gpd.overlay(gdf_huan, poly_gdf, how="intersection")
    if city_in_poly.empty:
        print(f"[WARN] No intersection polygons for {region_name} {year}. Skip.")
        return

    try:
        # newer geopandas + shapely may support this on GeoSeries
        research_union = city_in_poly.geometry.union_all()
    except Exception:
        # geopandas 0.14 compatible
        research_union = city_in_poly.geometry.unary_union

    # Research mask
    city_polygons = list(city_in_poly.geometry)
    city_region = regionmask.Regions(city_polygons, names=["city"] * len(city_polygons))
    city_mask = city_region.mask(city_ratio)
    masked_ratio = city_ratio_land.where(city_mask.notnull())

    # Plot extent
    lon_min, lon_max, lat_min, lat_max = plot_extent
    height = 4.0
    aspect = (lon_max - lon_min) / max((lat_max - lat_min), 1e-6)
    width = height * aspect

    fig, ax = plt.subplots(figsize=(width, height), subplot_kw={"projection": ccrs.PlateCarree()}, dpi=400)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", zorder=0)

    # Coastline
    coastline_gdf.plot(ax=ax, edgecolor="black", linewidth=0.5,
                       transform=ccrs.PlateCarree(), zorder=5)

    # Ticks
    ax.set_xticks(np.arange(np.floor(lon_min) + 5, np.ceil(lon_max) + 1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.floor(lat_min), np.ceil(lat_max) + 1, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(axis="both", labelsize=fs_tick)

    # Colormap
    orig_cmap = cm.get_cmap("OrRd")
    new_cmap = ListedColormap(orig_cmap(np.linspace(0.3, 1.0, 256)))

    # Raster
    plot_vals = city_ratio_land.values
    plot_vals = np.where(plot_vals >= 0, plot_vals, np.nan)

    ax.pcolormesh(
        city_ratio_land.lon, city_ratio_land.lat, plot_vals,
        cmap=new_cmap, shading="auto", vmin=0, vmax=0.2,
        transform=ccrs.PlateCarree(), zorder=1
    )

    # Fade non-research area
    buffer = 1.0
    full_bounds = Polygon([
        (lon_min - buffer, lat_min - buffer),
        (lon_max + buffer, lat_min - buffer),
        (lon_max + buffer, lat_max + buffer),
        (lon_min - buffer, lat_max + buffer)
    ])
    outside_poly = full_bounds.difference(research_union)
    outside_gdf = gpd.GeoDataFrame(geometry=[outside_poly], crs="EPSG:4326")
    outside_gdf.plot(ax=ax, facecolor="#FEF5F2", edgecolor="none", alpha=0.7,
                     transform=ccrs.PlateCarree(), zorder=3)

    # Border
    ax.spines["geo"].set_edgecolor("black")
    ax.spines["geo"].set_linewidth(1.2)

    # Statistics
    stat_vals = masked_ratio.values
    stat_vals = np.where(stat_vals >= 0, stat_vals, np.nan)

    count_ge_02 = int(np.count_nonzero(np.isfinite(stat_vals) & (stat_vals >= 0.2)))
    mean_ratio_in_shadow = float(np.nanmean(stat_vals)) if np.isfinite(stat_vals).any() else np.nan

    anno_text = (
        "Grids (≥0.2): {:,}\n"
        "Average impervious \narea ratio: {:.3f}"
    ).format(
        count_ge_02,
        mean_ratio_in_shadow if np.isfinite(mean_ratio_in_shadow) else np.nan
    )
    add_topleft_text(ax, anno_text, fs_anno)

    plt.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def make_city_ratio_maps(year=2022):
    # Load shapefiles
    admin1_gdf = gpd.read_file(ADMIN1_SHP).to_crs("EPSG:4326")
    coastline_gdf = gpd.read_file(COAST_SHP).to_crs("EPSG:4326")
    land_gdf = gpd.read_file(LAND_SHP).to_crs("EPSG:4326")

    gdf_c200 = gpd.read_file(SHP_C200)
    gdf_c100 = gpd.read_file(SHP_C100)
    gdf_huan = gpd.GeoDataFrame(pd.concat([gdf_c200, gdf_c100], ignore_index=True), crs=gdf_c200.crs).to_crs("EPSG:4326")

    nc_filename = f"{year}-30m.nc"
    nc_path = os.path.join(CITY_RATIO_DIR, nc_filename)
    if not os.path.exists(nc_path):
        print(f"[WARN] NetCDF not found: {nc_path}. Skip maps.")
        return

    with xr.open_dataset(nc_path) as ds:
        # map outputs
        out_map_paths = {
            "WNA": os.path.join(FIG_DIR, OUT_NAMES["map_WNA"]),
            "WNP": os.path.join(FIG_DIR, OUT_NAMES["map_WNP"]),
            "BOB": os.path.join(FIG_DIR, OUT_NAMES["map_BOB"]),
            "Northern_Australia": os.path.join(FIG_DIR, OUT_NAMES["map_NAUS"]),
            "Eastern_mozambique": os.path.join(FIG_DIR, OUT_NAMES["map_EMOZ"]),
        }

        for region_name, poly_coords in country_bounds.items():
            if region_name not in plot_extents:
                continue
            out_png = out_map_paths.get(region_name, os.path.join(FIG_DIR, f"{region_name}_{year}_city_ratio.png"))

            plot_city_ratio(
                ds=ds,
                region_name=region_name,
                polygon_coords=poly_coords,
                plot_extent=plot_extents[region_name],
                year=year,
                land_gdf=land_gdf,
                gdf_huan=gdf_huan,
                coastline_gdf=coastline_gdf,
                out_png=out_png
            )


# =============================================================================
# Part D: Standalone colorbar for impervious area ratio
# =============================================================================
def make_impervious_colorbar(out_path):
    fig, ax = plt.subplots(figsize=(8, 0.5), dpi=400)
    fig.subplots_adjust(bottom=0.5)

    city_levels = np.arange(0, 0.21, 0.05)

    orig_cmap = plt.get_cmap("OrRd")
    new_cmap = mpl.colors.ListedColormap(orig_cmap(np.linspace(0.3, 1.0, 256)))
    norm_ = mpl.colors.BoundaryNorm(city_levels, new_cmap.N)

    cb = mpl.colorbar.ColorbarBase(
        ax, cmap=new_cmap, norm=norm_, orientation="horizontal", extend="max"
    )
    cb.ax.tick_params(labelsize=16)
    cb.set_label("Impervious area ratio", fontsize=16)

    plt.savefig(out_path, dpi=400, bbox_inches="tight", transparent=False)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # --- A) Bar plots ---
    for name, cfg in DATASETS.items():
        run_one_dataset(cfg, name)

    # --- B) Legend ---
    make_legend_png(os.path.join(FIG_DIR, OUT_NAMES["bar_legend"]))

    # --- C) Maps (year=2022) ---
    make_city_ratio_maps(year=2022)

    # --- D) Colorbar ---
    make_impervious_colorbar(os.path.join(FIG_DIR, OUT_NAMES["map_cbar"]))
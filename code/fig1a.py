import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle 
from scipy import stats

sns.set_style("white")

# Figure output config
FIG_DIR = "../results"
FIG_NAME = "fig1a.pdf"

# Define linear regression function
def trendline(data):
    index = np.arange(len(data))
    result = stats.linregress(index, data)
    return result.slope * 10, result.pvalue, result.rvalue ** 2  # slope scaled to per decade

# Define plotting function with regression line
def plot_line_with_reg(x, y, color, label, ax, annotate=False, linewidth=2.2):
    sns.lineplot(x=x, y=y, color=color, linewidth=linewidth, label=label, ax=ax)
    sns.regplot(x=x, y=y, scatter=False, ci=95,
                line_kws={'linestyle': '--', 'color': color}, ax=ax)
    if annotate:
        slope, p_value, R2 = trendline(y.dropna())
        slope_text = f"{slope:.2f}"
        p_value_text = f"{p_value:.2f}"
        rvalue2_text = f"{R2:.2f}"
        ax.text(0.5, 0.93, f'slope={slope_text}, p={p_value_text}, $R^2$={rvalue2_text}',
                transform=ax.transAxes, fontsize=15, ha='center')

# === Read MSWEP data ===
df = pd.read_csv("../data//mswep/area_790000/result_r500.csv",
                 na_values=[''], keep_default_na=False)
df_mswep = df[df['NEAR_FID'] != -1]  # Filter out invalid points
df_mswep = df_mswep[df_mswep['SEASON'] <= 2023]
df_mswep['ISO_TIME'] = pd.to_datetime(df_mswep['ISO_TIME'], errors='coerce')
df_mswep['MONTH'] = df_mswep['ISO_TIME'].dt.month

# === Set placeholder threshold (distance band) ===
placeholder = 30  # Example: distance threshold in km, used in column names

# Function to calculate annual mean landward distance for filtered data
def calc_land_year(df_base, dist_min=-100, dist_max=500):
    # Filter by distance range from coastline
    df_filtered = df_base[(df_base['NEAR_DIST'] <= dist_max) &
                          (df_base['NEAR_DIST'] >= dist_min)]

    # Sum weighted values and weights by storm ID (SID)
    sum_weighted = df_filtered.groupby('SID')[f'weighted_sum_land_{placeholder}'].sum()
    sum_weights  = df_filtered.groupby('SID')[f'weight_land_{placeholder}'].sum()

    # Calculate weighted average; replace inf with NaN
    land_avg = (sum_weighted / sum_weights).replace([np.inf, -np.inf], np.nan)

    # Merge with season info
    season_map = df_filtered[['SID', 'SEASON']].drop_duplicates().set_index('SID')
    land = pd.concat([season_map, land_avg.rename('land_avg')], axis=1).dropna()

    # Compute seasonal mean
    land_year = land.groupby('SEASON')['land_avg'].mean()

    return land_year

def print_results_table_lineplot(datasets_info, dataset_name="Landward distance time series"):
    """
    datasets_info: list of dicts with keys: name, slope, p, r2, stderr
      - slope: raw slope (before negation for plotting convention)
      - stderr: standard error of slope (per year)
    """
    print(f"\n=== Results for {dataset_name} ===")
    header = "Dataset | Slope_plot (km/dec) | CI± (km/dec) | p-value | R²"
    print(header)
    print("-" * len(header))
    for ds in datasets_info:
        slope_plot = -ds["slope"]          # match plotting sign
        ci_hw = ds["stderr"] * 10 * 1.96   # convert stderr/year to CI half-width per decade
        print(f"{ds['name']:7} | {slope_plot:21.3f} | {ci_hw:12.3f} | {ds['p']:7.3g} | {ds['r2']:4.3f}")

# Collect stderr for each dataset (need to recompute linregress to get stderr)
def get_slope_stats(series):
    series = series.dropna()
    x = np.arange(len(series))
    res = stats.linregress(x, series.values)
    return res.slope, res.pvalue, res.rvalue**2, res.stderr

# Compute annual series for each dataset
land_year_mswep = calc_land_year(df_mswep)

# Get stats
s_m, p_m, r2_m, stderr_m = get_slope_stats(land_year_mswep)

datasets_info = [
    {"name": "MSWEP",  "slope": s_m, "p": p_m, "r2": r2_m, "stderr": stderr_m},
]

print_results_table_lineplot(datasets_info)
# Compute trend metrics for each dataset
slope_m, p_m, r2_m = trendline(land_year_mswep.dropna())

# Construct legend labels including slope, p-value
def make_label(name, slope, p, r2):
    slope_str = f"{-slope:.1f}"  # Note: negative sign for plotting convention
    p_str = " < 0.05" if p < 0.05 else f" = {p:.2f}"
    return f"Trend: {slope_str} km decade$^{{-1}}$, $P${p_str}"

label_mswep = make_label("MSWEP", slope_m, p_m, r2_m)

# === Plotting ===
fig, ax = plt.subplots(figsize=(7, 2.4), dpi=150)

# Plot each dataset with regression line
plot_line_with_reg(land_year_mswep.index, -land_year_mswep, color='#3B83B4',
                   label=label_mswep, ax=ax, annotate=False, linewidth=3.2)

# Axis labels and ticks
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Landward distance \n from coastline (km)', fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

handles, labels = ax.get_legend_handles_labels()
transparent_handle = Rectangle((0, 0), 0, 0, alpha=0, linewidth=0)
new_handles = [transparent_handle] * len(handles)
ax.legend(new_handles, labels, loc='upper left', bbox_to_anchor=(0, 1.04),
          fontsize=14, labelspacing=0.2, frameon=False, handlelength=0)

ax.set_xlim(1978, 2025)
ax.set_ylim(35, 80)

# 主刻度
ax.set_xticks(np.arange(1980, 2024, 10))
ax.set_yticks(np.arange(40, 81, 10))
ax.tick_params(
    axis='both', which='major',
    bottom=True, left=True, top=False, right=False,
    length=6, width=0.8,
    labelsize=14,
    direction='out',
    colors='k'
)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_linewidth(0.8)
ax.spines['left'].set_linewidth(0.8)
ax.spines['top'].set_linewidth(0.8)
ax.spines['right'].set_linewidth(0.8)

# Save figure
import os
os.makedirs(FIG_DIR, exist_ok=True)

fig_path = os.path.join(FIG_DIR, FIG_NAME)
plt.savefig(fig_path, dpi=400, bbox_inches="tight")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pymannkendall as mk

# Figure output config
FIG_DIR = "../results"
FIG_NAME = "fig1b.pdf"

# Helper functions
def trendline(data: pd.Series):
    """
    OLS slope + 95% CI (per decade)
    MK test for significance
    """
    data = pd.Series(data).dropna()
    if len(data) < 3:
        return np.nan, np.nan, np.nan

    x = np.arange(len(data))
    ols = stats.linregress(x, data.values)
    slope_per_decade = ols.slope * 10
    ci95_halfwidth_per_decade = ols.stderr * 10 * 1.96

    mk_res = mk.original_test(data.values)
    p_mk = mk_res.p
    return slope_per_decade, p_mk, ci95_halfwidth_per_decade

def calc_land_year(df_base: pd.DataFrame, placeholder: int, dist_min=-100, dist_max=500):
    df_filtered = df_base[
        (df_base['NEAR_DIST'] <= dist_max) &
        (df_base['NEAR_DIST'] >= dist_min)
    ]

    sum_weighted = df_filtered.groupby('SID')[f'weighted_sum_land_{placeholder}'].sum()
    sum_weights  = df_filtered.groupby('SID')[f'weight_land_{placeholder}'].sum()

    land_avg = (sum_weighted / sum_weights).replace([np.inf, -np.inf], np.nan)

    season_map = df_filtered[['SID', 'SEASON']].drop_duplicates().set_index('SID')
    land = pd.concat([season_map, land_avg.rename('land_avg')], axis=1).dropna()

    land_year = land.groupby('SEASON')['land_avg'].mean()
    return land_year


def print_results_table_hemisphere(df: pd.DataFrame, hemispheres_order, datasets_order,
                                   dataset_name="Hemispheric landward trends"):
    print(f"\n=== Results for {dataset_name} ===")
    header = "Hemisphere | Dataset | Slope_plot (km/dec) | CI± (km/dec) | p-value"
    print(header)
    print("-" * len(header))

    for hemi in hemispheres_order:
        for ds in datasets_order:
            row = df[(df["hemisphere"] == hemi) & (df["dataset"] == ds)]
            if row.empty:
                print(f"{hemi:10} | {ds:7} | {'NA':>21} | {'NA':>12} | {'NA':>7}")
                continue
            slope = row["slope"].values[0]
            pval  = row["pvalue"].values[0]
            ci_hw = row["stderr"].values[0]
            slope_plot = -slope
            if np.isnan(slope_plot) or np.isnan(ci_hw) or np.isnan(pval):
                print(f"{hemi:10} | {ds:7} | {'NA':>21} | {'NA':>12} | {'NA':>7}")
            else:
                print(f"{hemi:10} | {ds:7} | {slope_plot:21.3f} | {ci_hw:12.3f} | {pval:7.3g}")


# Data loading
df = pd.read_csv(
    "../data/mswep/area_790000/result_r500.csv",
    na_values=[''], keep_default_na=False
)
df_mswep = df[(df['NEAR_FID'] != -1) & (df['SEASON'] <= 2023)].copy()
df_mswep['ISO_TIME'] = pd.to_datetime(df_mswep['ISO_TIME'], errors='coerce')

df_imerg = pd.read_csv(
    "../data/aftertreatment/imerg/result_r500_imerg.csv",
    na_values=[''], keep_default_na=False
)
for col in ['NEAR_FID', 'USA_LAT']:
    if col not in df_imerg.columns:
        df_imerg[col] = df[col].values
df_imerg = df_imerg[(df_imerg['NEAR_FID'] != -1) & (df_imerg['SEASON'] <= 2023)].copy()
df_imerg['ISO_TIME'] = pd.to_datetime(df_imerg['ISO_TIME'], errors='coerce')

placeholder = 30

# Compute trends
hemispheres = ['NH', 'SH']
datasets_all = ['MSWEP', 'IMERG']
results = []

for hemi in hemispheres:
    if hemi == 'NH':
        f_mswep  = df_mswep[df_mswep['USA_LAT'] > 0]
        f_imerg  = df_imerg[df_imerg['USA_LAT'] > 0]
    else:
        f_mswep  = df_mswep[df_mswep['USA_LAT'] < 0]
        f_imerg  = df_imerg[df_imerg['USA_LAT'] < 0]

    land_year_mswep  = calc_land_year(f_mswep,  placeholder)
    land_year_imerg  = calc_land_year(f_imerg,  placeholder)

    slope_m, p_m, err_m = trendline(land_year_mswep)
    slope_i, p_i, err_i = trendline(land_year_imerg)

    results.extend([
        {'hemisphere': hemi, 'dataset': 'MSWEP',  'slope': slope_m, 'pvalue': p_m, 'stderr': err_m},
        {'hemisphere': hemi, 'dataset': 'IMERG',  'slope': slope_i, 'pvalue': p_i, 'stderr': err_i},
    ])

results_df = pd.DataFrame(results)
results_df['hemisphere'] = pd.Categorical(results_df['hemisphere'],
                                          categories=hemispheres, ordered=True)
results_df.sort_values(['hemisphere', 'dataset'], inplace=True)

print_results_table_hemisphere(
    results_df, hemispheres_order=hemispheres, datasets_order=datasets_all,
    dataset_name="Landward distance trend by hemisphere & dataset"
)

# Plotting
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=400)

gap = 0.7
x = np.arange(len(hemispheres)) * gap
width = 0.28

plot_datasets = ['MSWEP', 'IMERG']
plot_colors = {
    'MSWEP': '#3B83B4',
    'IMERG': '#BF453D'
}
offsets = [-width/2, width/2]

for i, dataset in enumerate(plot_datasets):
    color = plot_colors[dataset]
    data = results_df[results_df['dataset'] == dataset]

    y_vals = (-data['slope']).to_numpy()
    y_errs = data['stderr'].to_numpy()

    bars = ax.bar(
        x + offsets[i], y_vals, width,
        label=dataset, color=color, edgecolor='none',
        yerr=y_errs, capsize=5
    )

    # significance markers
    for bar, pvalue, err in zip(bars, data['pvalue'].to_numpy(), y_errs):
        if pvalue <= 0.05:
            marker = '**'
        elif pvalue <= 0.1:
            marker = '*'
        else:
            continue
        height = bar.get_height()
        y_pos = height + np.sign(height) * (err - 0.82)
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                marker, ha='center', va=va,
                fontsize=24, fontweight='bold', fontfamily='Arial')

# Axes styling
ax.set_ylabel('Trend of landward \n distance (km decade$^{-1}$)', fontsize=19)
ax.set_ylim(-18, 18)
ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
ax.set_xticks(x)
ax.set_xticklabels(hemispheres, fontsize=22)
ax.tick_params(axis='y', labelsize=22)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.legend(loc='lower left', fontsize=22)

ax.tick_params(
    axis='both', which='major',
    bottom=True, left=True, top=False, right=False,
    length=6, width=1.2,
    labelsize=19,
    direction='out',
    colors='k'
)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)
ax.spines['top'].set_linewidth(1.2)
ax.spines['right'].set_linewidth(1.2)

# Save figure
import os
os.makedirs(FIG_DIR, exist_ok=True)

fig_path = os.path.join(FIG_DIR, FIG_NAME)
plt.savefig(fig_path, dpi=400, bbox_inches="tight")
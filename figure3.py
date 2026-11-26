from pathlib import Path
from itertools import permutations
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import rankdata, spearmanr


BRAAK_GROUPS = {
    1: ["OEF_Entorhinal"],
    2: ["OEF_Hippocampus"],
    3: ["OEF_Amygdala", "OEF_Fusiform", "OEF_Parahippocampal"],
    4: ["OEF_Inferior temporal", "OEF_Middle temporal", "OEF_Isthmus cingulate", "OEF_Precuneus", "OEF_Insula", 
        "OEF_Posterior cingulate cortex"],
    5: ["OEF_Lateral occipital", "OEF_Lingual", "OEF_Parsopercularis", "OEF_Rostral middle frontal", "OEF_Parstriangularis", 
        "OEF_Suptemporal", "OEF_Pars orbitalis", "OEF_Caudale anterior cingulate", "OEF_Rostral anterior cingulate", "OEF_Cuneus", 
        "OEF_Supfrontal", "OEF_Caudal middle frontal", "OEF_Supra marginal", "OEF_Inferior parietal", "OEF_Supparietal", 
        "OEF_Lateral orbitofrontal", "OEF_Medial Orbital"],
    6: ["OEF_Para central", "OEF_Post central", "OEF_Precentral", "OEF_Pericalcarine", "OEF_Transtemporal"],
    "Others": ["OEF_Pallidum", "OEF_Putamen", "OEF_Caudate", "OEF_Thalamus"],
}


def spearman_pair(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, np.nan
    x = x[valid]
    y = y[valid]
    result = spearmanr(x, y)
    if not np.isfinite(result.statistic):
        return np.nan, np.nan
    if valid.sum() > 9:
        return result.statistic, result.pvalue
    rank_x = rankdata(x)
    rank_y = rankdata(y)
    centered_x = rank_x - rank_x.mean()
    centered_y = rank_y - rank_y.mean()
    denominator = np.sqrt(np.sum(centered_x ** 2) * np.sum(centered_y ** 2))
    observed = abs(np.sum(centered_x * centered_y) / denominator)
    extreme = sum(abs(np.sum(centered_x * (np.asarray(order) - rank_y.mean())) / denominator) >= observed - 1e-12 for order in permutations(rank_y))
    return result.statistic, extreme / factorial(len(rank_y))


def bh_qvalues(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    qvalues = np.full(pvalues.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    order = np.argsort(pvalues[valid])
    ranked = pvalues[valid][order] * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    qvalues[valid[order]] = np.minimum(adjusted, 1)
    return qvalues


folder = Path(__file__).resolve().parent
data = pd.read_excel(folder / "TRIAD_OEF.xlsx")
region_order = [region for group in BRAAK_GROUPS.values() for region in group]
biomarker_start = data.columns.get_loc("plasma_ab40_pg_ml__simoa1")      
biomarker_end = data.columns.get_loc("alps") + 1
biomarker_columns = [column for column in data.columns[biomarker_start:biomarker_end] if "batch" not in str(column).lower() and column != "massspect_sample_id"]
oef = data[region_order].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
biomarkers = data[biomarker_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

results = []
for region in region_order:
    for biomarker in biomarker_columns:
        x_values = oef[region].to_numpy(dtype=float)
        y_values = biomarkers[biomarker].to_numpy(dtype=float)
        rho, pvalue = spearman_pair(x_values, y_values)
        n = np.sum(np.isfinite(x_values) & np.isfinite(y_values))
        results.append((region, biomarker, rho, pvalue, n))

correlations = pd.DataFrame(results, columns=["OEF_Region", "Biomarker", "Spearman_Correlation", "P_Value", "N"])
correlations["Q_Value"] = bh_qvalues(correlations["P_Value"].to_numpy())

fig = plt.figure(figsize=(20, 15))
grid = fig.add_gridspec(2, 2, height_ratios=(1.12, 1), hspace=0.56, wspace=0.38)
ax_a = fig.add_subplot(grid[0, :])
ax_b = fig.add_subplot(grid[1, 0])
ax_c = fig.add_subplot(grid[1, 1])

heatmap = correlations.pivot(index="OEF_Region", columns="Biomarker", values="Spearman_Correlation")
heatmap = heatmap.reindex(region_order).dropna(axis=1, how="all")
column_order = leaves_list(linkage(heatmap.fillna(0).to_numpy().T, method="ward"))
heatmap = heatmap.iloc[:, column_order]
image_a = ax_a.imshow(heatmap.to_numpy(), aspect="auto", interpolation="nearest", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6))
ax_a.set_xticks(np.arange(heatmap.shape[1]))
ax_a.set_xticklabels(heatmap.columns, rotation=90, ha="center", fontsize=5.5)
ax_a.set_yticks(np.arange(heatmap.shape[0]))
ax_a.set_yticklabels(heatmap.index, fontsize=5.5)
ax_a.tick_params(length=2, pad=1)

group_centers = []
row_start = 0
for stage, regions in BRAAK_GROUPS.items():
    included = [region for region in regions if region in heatmap.index]
    if included:
        group_centers.append(((row_start + row_start + len(included) - 1) / 2, str(stage)))
        row_start += len(included)
        ax_a.axhline(row_start - 0.5, color="#072b3a", linewidth=0.7)

colorbar_a = fig.colorbar(image_a, ax=ax_a, fraction=0.012, pad=0.01)
colorbar_a.set_label("Spearman correlation", fontsize=10)
colorbar_a.set_ticks([-0.6, -0.3, 0, 0.3, 0.6])
fig.text(0.012, 0.965, "A", fontsize=24, fontweight="bold")

volcano = correlations.loc[correlations["N"] >= 10].dropna(subset=["Spearman_Correlation", "Q_Value"]).copy()
volcano["Minus_Log10_Q"] = -np.log10(volcano["Q_Value"].clip(lower=np.finfo(float).tiny))
x = volcano["Spearman_Correlation"].to_numpy(dtype=float)
y = volcano["Minus_Log10_Q"].to_numpy(dtype=float)
ax_b.scatter(x, y, c=x, cmap="coolwarm", edgecolors="black", linewidths=0.35, alpha=0.75, s=28)
ax_b.axhline(-np.log10(0.005), color="black", linestyle="--", linewidth=0.8)
ax_b.set_xlabel("Spearman correlation")
ax_b.set_ylabel("-log10(q-value)")
highlight = volcano.loc[(volcano["Spearman_Correlation"].abs() > 0.45) & (volcano["Q_Value"] < 0.005)].copy()
if highlight.empty:
    highlight = volcano.nsmallest(10, "Q_Value").copy()
else:
    highlight = highlight.sort_values("Q_Value").drop_duplicates("Biomarker")
rng = np.random.default_rng(0)
texts = []
for row in highlight.itertuples(index=False):
    texts.append(ax_b.text(row.Spearman_Correlation + rng.uniform(-0.015, 0.015), row.Minus_Log10_Q + rng.uniform(-0.3, 0.3), str(row.Biomarker), fontsize=7))
adjust_text(texts, ax=ax_b, x=x, y=y, arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.5}, expand_text=(1.2, 1.2), force_text=(0.5, 0.5))
fig.text(0.012, 0.46, "B", fontsize=24, fontweight="bold")

braak_means = pd.DataFrame({f"OEF_Braak{stage}": oef[regions].mean(axis=1) for stage, regions in BRAAK_GROUPS.items() if isinstance(stage, int)})
stage_labels = list(braak_means.columns)
number_of_stages = len(stage_labels)
correlation_matrix = np.eye(number_of_stages)
pairwise = []
for i in range(number_of_stages):
    for j in range(i + 1, number_of_stages):
        rho, pvalue = spearman_pair(braak_means.iloc[:, i].to_numpy(dtype=float), braak_means.iloc[:, j].to_numpy(dtype=float))
        pairwise.append((i, j, rho, pvalue))
qvalues = bh_qvalues([row[3] for row in pairwise])
log_q_matrix = np.full((number_of_stages, number_of_stages), np.nan)
for qvalue, (i, j, rho, pvalue) in zip(qvalues, pairwise):
    correlation_matrix[i, j] = rho
    correlation_matrix[j, i] = rho
    log_q = -np.log10(max(qvalue, np.finfo(float).tiny))
    log_q_matrix[i, j] = log_q
    log_q_matrix[j, i] = log_q

image_c = ax_c.imshow(np.ma.masked_invalid(log_q_matrix), cmap="RdYlGn", vmin=0, vmax=160, aspect="equal")
ax_c.set_xticks(np.arange(number_of_stages))
ax_c.set_yticks(np.arange(number_of_stages))
ax_c.set_xticklabels(stage_labels, rotation=15, ha="right", fontsize=8)
ax_c.set_yticklabels(stage_labels, fontsize=8)
for i in range(number_of_stages):
    for j in range(number_of_stages):
        ax_c.text(j, i, f"{correlation_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
colorbar_c = fig.colorbar(image_c, ax=ax_c, fraction=0.046, pad=0.04, ticks=np.arange(0, 161, 20))
colorbar_c.set_label("-log10 q-value")
fig.text(0.515, 0.46, "C", fontsize=24, fontweight="bold")

fig.subplots_adjust(left=0.17, right=0.96, top=0.98, bottom=0.06)
for center, label in group_centers:
    y_position = fig.transFigure.inverted().transform(ax_a.transData.transform((0, center)))[1]
    fig.text(ax_a.get_position().x0 - 0.075, y_position, label, ha="right", va="center", fontsize=8, fontweight="bold")
fig.text(ax_a.get_position().x0 - 0.145, ax_a.get_position().y0 + ax_a.get_position().height / 2, "Braak stages", rotation=90, ha="center", va="center", fontsize=10, fontweight="bold")
fig.savefig(folder / "Figure3.png", dpi=600, bbox_inches="tight")
plt.close(fig)

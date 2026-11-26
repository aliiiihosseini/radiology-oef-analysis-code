from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f, spearmanr, studentized_range, t

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / 'TRIAD_OEF.xlsx'
OUTPUT_DIR = ROOT / 'OEF_paper_figures'
GROUPS = ['CU (Y)', 'CU', 'MCI', 'AD']
OLDER_GROUPS = ['CU', 'MCI', 'AD']
REGIONS = [
    'OEF_Precuneus',
    'OEF_Posterior cingulate cortex',
    'OEF_Entorhinal',
    'OEF_Putamen',
    'OEF_Hippocampus',
    'OEF_Parahippocampal'
]
S4_MARKERS = [                                                                  
    'csf_ykl40_pg_ml__elisa',
    'cxcl1',
    'cxcl5',
    'cd40',
    'cxcl10',
    'hgf',
    'tnfb',
    'plasma_strem2',
    'plasma_egfr',
    'mcp_1',
    'lif',
    'il8'
]
S4_TITLES = {
    'csf_ykl40_pg_ml__elisa': 'csf ykl40',
    'plasma_egfr': 'plasma egfr'                                            
}
PALETTE = {
    'CU (Y)': '#1b998b',
    'CU': '#2e294e',
    'MCI': '#f46036',
    'AD': '#d7263d'
}

sns.set_style('white')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3
})


def format_p(value, threshold=0.001, digits=3):
    if not np.isfinite(value):
        return 'NA'
    if value < threshold:
        return f'<{threshold:.{digits}f}'
    return f'{value:.{digits}f}'


def clean_data(frame, outcome, groups, age_min=None):
    data = frame[['Category', 'age_at_mri', outcome]].copy()
    data.columns = ['group', 'age', 'value']
    data['group'] = data['group'].astype(str).str.strip()
    data['age'] = pd.to_numeric(data['age'], errors='coerce')
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=['group', 'age', 'value'])
    data = data[data['group'].isin(groups)].copy()
    if age_min is not None:
        data = data[data['age'] > age_min].copy()
    data['group'] = pd.Categorical(data['group'], categories=groups, ordered=True)
    return data


def region_entries(start_panel):
    return [
        (chr(ord(start_panel) + index), f'OEF_{region}', region)             
        for index, region in enumerate(REGIONS)
    ]


def marker_entries():
    return [
        (
            chr(ord('A') + index),
            marker,
            S4_TITLES.get(marker, marker.replace('_', ' '))
        )
        for index, marker in enumerate(S4_MARKERS)
    ]


def design_matrix(data, groups, with_age=False):
    group_values = data['group'].astype(str).to_numpy()
    columns = [np.ones(len(data))]
    for group in groups[1:]:
        columns.append((group_values == group).astype(float))
    if with_age:
        columns.append(data['age'].to_numpy(dtype=float) - data['age'].mean())
    return np.column_stack(columns)


def fit_model(matrix, values):
    beta, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
    residuals = values - matrix @ beta
    residual_df = len(values) - rank
    residual_ss = float(residuals @ residuals)
    mse = residual_ss / residual_df
    covariance = mse * np.linalg.pinv(matrix.T @ matrix)
    return beta, covariance, residual_ss, residual_df, rank, mse


def nested_f_test(values, reduced, full):
    _, _, reduced_ss, _, reduced_rank, _ = fit_model(reduced, values)
    _, _, full_ss, residual_df, full_rank, _ = fit_model(full, values)
    numerator_df = full_rank - reduced_rank
    statistic = ((reduced_ss - full_ss) / numerator_df) / (full_ss / residual_df)
    return float(statistic), float(f.sf(statistic, numerator_df, residual_df)), numerator_df, residual_df


def pairwise_tests(data, groups, age_adjusted=False):
    values = data['value'].to_numpy(dtype=float)
    group_values = data['group'].astype(str).to_numpy()
    counts = {group: int(np.sum(group_values == group)) for group in groups}
    group_means = {group: float(np.mean(values[group_values == group])) for group in groups}
    matrix = design_matrix(data, groups, with_age=age_adjusted)
    beta, covariance, _, residual_df, _, mse = fit_model(matrix, values)
    age_value = float(data['age'].mean())
    q_critical = float(studentized_range.ppf(0.95, len(groups), residual_df))  
    rows = []
    if age_adjusted:
        new_data = pd.DataFrame({
            'group': pd.Categorical(groups, categories=groups, ordered=True),
            'age': [age_value] * len(groups)
        })
        new_matrix = design_matrix(new_data, groups, with_age=True)
    for i, j in combinations(range(len(groups)), 2):
        if age_adjusted:
            contrast = new_matrix[i] - new_matrix[j]
            difference = float(contrast @ beta)
            variance = float(contrast @ covariance @ contrast)
            standard_error = np.sqrt(max(variance, 0.0))
        else:
            difference = group_means[groups[j]] - group_means[groups[i]]
            standard_error = np.sqrt(mse * (1 / counts[groups[i]] + 1 / counts[groups[j]]))
        q_statistic = np.sqrt(2) * abs(difference / standard_error)
        rows.append({
            'group_1': groups[i],
            'group_2': groups[j],
            'x1': i,
            'x2': j,
            'p': float(studentized_range.sf(q_statistic, len(groups), residual_df))
        })
    return rows


def region_statistics(frame, outcome, groups, age_min=None):
    data = clean_data(frame, outcome, groups, age_min)
    values = data['value'].to_numpy(dtype=float)
    group_matrix = design_matrix(data, groups)                                  
    age_matrix = np.column_stack([
        np.ones(len(data)),
        data['age'].to_numpy(dtype=float) - data['age'].mean()
    ])
    group_age_matrix = design_matrix(data, groups, with_age=True)
    anova_f, anova_p, anova_df, anova_residual_df = nested_f_test(
        values, np.ones((len(data), 1)), group_matrix
    )
    ancova_f, ancova_p, ancova_df, ancova_residual_df = nested_f_test(
        values, age_matrix, group_age_matrix
    )
    return {
        'data': data,
        'anova_f': anova_f,
        'anova_p': anova_p,
        'anova_df': anova_df,
        'anova_residual_df': anova_residual_df,
        'ancova_f': ancova_f,
        'ancova_p': ancova_p,
        'ancova_df': ancova_df,
        'ancova_residual_df': ancova_residual_df,
        'anova_pairs': pairwise_tests(data, groups),                         
        'ancova_pairs': pairwise_tests(data, groups, age_adjusted=True)
    }


def add_panel_letter(ax, panel, size=32):
    ax.text(-0.075, 1.06, panel, transform=ax.transAxes, ha='left', va='top', fontsize=size, fontweight='bold', clip_on=False)


def add_pairwise_brackets(ax, data, groups, stats, s4=False):
    minimum = float(data['value'].min())
    maximum = float(data['value'].max())
    span = maximum - minimum
    if not np.isfinite(span) or span <= 0:
        span = max(abs(maximum), 1.0) * 0.1
    bracket_height = 0.025 * span
    bracket_step = 0.175 * span
    bracket_base = maximum + 0.08 * span
    comparisons = sorted(zip(stats['anova_pairs'], stats['ancova_pairs']), key=lambda pair: (pair[0]['x2'], pair[0]['x1']))
    for level, (anova_row, ancova_row) in enumerate(comparisons):
        x1 = anova_row['x1']
        x2 = anova_row['x2']
        y = bracket_base + level * bracket_step
        ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y], color='black', linewidth=0.8, clip_on=False)
        threshold = 0.0001 if s4 else 0.001
        label = (
            f"ANCOVA Tukey p={format_p(ancova_row['p'], threshold, 4 if s4 else 3)}\n"
            f"ANOVA Tukey p={format_p(anova_row['p'], threshold, 4 if s4 else 3)}"
        )
        ax.text(
            (x1 + x2) / 2,
            y + bracket_height + 0.012 * span,
            label,
            ha='center',
            va='bottom',
            fontsize=6.5 if s4 else 7.5,
            fontweight='bold',
            linespacing=1.0,
            clip_on=False
        )
    return minimum - 0.06 * span, bracket_base + max(len(comparisons) - 1, 0) * bracket_step + 0.35 * span


def plot_violin_axis(ax, frame, outcome, region_name, panel, groups, age_min=None, s4=False):
    data = clean_data(frame, outcome, groups, age_min)
    counts = data['group'].value_counts().reindex(groups, fill_value=0)
    stats = region_statistics(frame, outcome, groups, age_min)
    sns.violinplot(
        data=data,
        x='group',
        y='value',
        hue='group',
        order=groups,
        hue_order=groups,
        palette=PALETTE,
        dodge=False,
        legend=False,
        inner=None,
        cut=0,
        linewidth=0.9,
        saturation=0.9,
        ax=ax
    )
    sns.boxplot(
        data=data,
        x='group',
        y='value',
        order=groups,
        width=0.18,
        showfliers=False,
        boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 0.8},
        whiskerprops={'color': 'black', 'linewidth': 0.8},
        capprops={'color': 'black', 'linewidth': 0.8},
        medianprops={'color': 'black', 'linewidth': 0.9},
        ax=ax
    )
    if not s4:
        sns.stripplot(
            data=data,
            x='group',
            y='value',
            order=groups,
            color='black',
            alpha=0.4,
            size=1.8,
            jitter=0.18,
            ax=ax
        )
    lower, upper = add_pairwise_brackets(ax, data, groups, stats, s4=s4)
    ax.set_ylim(lower, upper)
    labels = list(groups) if s4 else [f'{group}\n(n={int(counts[group])})' for group in groups]
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=12 if not s4 else 8, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Biomarker value' if s4 else ('OEF (fraction)' if panel in 'EFGHIJ' else 'OEF value'), fontweight='bold', fontsize=9 if s4 else 10, labelpad=5)
    ax.set_title(region_name, pad=6, fontweight='bold', fontsize=9 if s4 else 10)
    add_panel_letter(ax, panel, 30 if s4 else 34)
    if s4:
        omnibus_text = (
            f"Age-adjusted ANCOVA: p={format_p(stats['ancova_p'], 0.0001, 4)}\n"
            f"Unadjusted one-way ANOVA: p={format_p(stats['anova_p'], 0.0001, 4)}"
        )
    else:
        omnibus_text = (
            f"Age-adjusted ANCOVA: F({stats['ancova_df']}, {stats['ancova_residual_df']})={stats['ancova_f']:.2f}, p={format_p(stats['ancova_p'])}\n"
            f"Unadjusted one-way ANOVA: F({stats['anova_df']}, {stats['anova_residual_df']})={stats['anova_f']:.2f}, p={format_p(stats['anova_p'])}"
        )
    ax.text(
        0.015,
        0.985,
        omnibus_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=6.3 if s4 else 7.2,
        fontweight='bold',
        linespacing=1.15,
        bbox={'boxstyle': 'round,pad=0.28', 'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.95, 'linewidth': 0.6},
        zorder=20
    )
    sns.despine(ax=ax)


def add_regression(ax, data, color, linestyle, label):
    x = data['age'].to_numpy(dtype=float)
    y = data['value'].to_numpy(dtype=float)
    rho, p_value = spearmanr(x, y)
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    grid = np.linspace(x.min(), x.max(), 200)
    prediction = intercept + slope * grid
    residuals = y - (intercept + slope * x)
    mse = np.sum(residuals ** 2) / (len(x) - 2)
    se_mean = np.sqrt(mse * (1 / len(x) + (grid - x_mean) ** 2 / sxx))
    critical = t.ppf(0.975, len(x) - 2)
    ax.plot(grid, prediction, color=color, linestyle=linestyle, linewidth=1.2, label=label, zorder=4)
    ax.fill_between(grid, prediction - critical * se_mean, prediction + critical * se_mean, color=color, alpha=0.12, zorder=1)
    return float(rho), float(p_value)


def plot_age_axis(ax, frame, outcome, region_name, panel):
    data = clean_data(frame, outcome, GROUPS)
    for group in GROUPS:
        subset = data[data['group'] == group]
        ax.scatter(
            subset['age'],
            subset['value'],
            s=14,
            alpha=0.75,
            label=f'{group} (n={len(subset)})',
            color=PALETTE[group],
            edgecolor='white',
            linewidth=0.25,
            zorder=3
        )
    subsets = [
        ('All participants', data, '#000000', '-'),
        ('Age ≤60 years', data[data['age'] <= 60], '#0072B2', '--'),
        ('Age >60 years', data[data['age'] > 60], '#CC79A7', ':')
    ]
    annotation = []
    for label, subset, color, linestyle in subsets:
        rho, p_value = add_regression(ax, subset, color, linestyle, label)
        annotation.append(f'{label} (n={len(subset)}): ρ={rho:.2f}, p={format_p(p_value)}')
    ax.text(
        0.02,
        0.98,
        '\n'.join(annotation),
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=6.8,
        fontweight='bold',
        linespacing=1.1,
        bbox={'boxstyle': 'round,pad=0.28', 'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.92, 'linewidth': 0.6},
        zorder=10
    )
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label not in seen:
            unique_handles.append(handle)
            unique_labels.append(label)
            seen.add(label)
    ax.legend(unique_handles, unique_labels, frameon=False, loc='upper right', fontsize=6.3, handlelength=1.5, labelspacing=0.25, borderaxespad=0.3) # mayeb you could move the legend outside the plot (or one legend for all subplots) so that it doesn't overlap the dots
    ax.set_xlabel('Age at MRI (years)', fontsize=8.5, fontweight='bold', labelpad=3)
    ax.set_ylabel('OEF value', fontsize=8.5, fontweight='bold', labelpad=3)
    ax.set_title(region_name, pad=5, fontweight='bold', fontsize=9)
    add_panel_letter(ax, panel, 32)
    sns.despine(ax=ax)


def save_violin_figure(frame, entries, groups, output_path, age_min=None, s4=False, figsize=(20.48, 14.08), dpi=100):
    rows = 4 if s4 else 2
    columns = 3
    fig, axes = plt.subplots(rows, columns, figsize=figsize, dpi=dpi)
    for ax, entry in zip(axes.flat, entries):
        panel, outcome, region_name = entry
        plot_violin_axis(ax, frame, outcome, region_name, panel, groups, age_min=age_min, s4=s4)
    fig.subplots_adjust(left=0.035 if s4 else 0.045, right=0.99, top=0.96, bottom=0.045 if s4 else 0.055, wspace=0.14, hspace=0.22)
    fig.savefig(output_path, dpi=dpi, facecolor='white')
    plt.close(fig)


def save_age_figure(frame, entries, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(20.48, 11.52), dpi=100)
    for ax, (panel, outcome, region_name) in zip(axes.flat, entries):
        plot_age_axis(ax, frame, outcome, region_name, panel)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.96, bottom=0.07, wspace=0.24, hspace=0.25)
    fig.savefig(output_path, dpi=100, facecolor='white')
    plt.close(fig)


def main():
    frame = pd.read_excel(INPUT_FILE, sheet_name='Sheet1')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure2_regions = region_entries('E')
    supp_regions = region_entries('A')
    supp4_markers = marker_entries()
    save_violin_figure(
        frame,
        figure2_regions,
        GROUPS,
        OUTPUT_DIR / 'Figure2_E-J.png'
    )
    save_violin_figure(
        frame,
        supp_regions,
        OLDER_GROUPS,
        OUTPUT_DIR / 'Supplementary_Figure2.png',
        age_min=60,
        figsize=(20.48, 14.08)
    )
    save_age_figure(frame, supp_regions, OUTPUT_DIR / 'Supplementary_Figure3.png')
    save_violin_figure(
        frame,
        supp4_markers,
        GROUPS,
        OUTPUT_DIR / 'Supplementary_Figure4.png',
        s4=True,
        figsize=(15.36, 20.48)
    )


if __name__ == '__main__':
    main()

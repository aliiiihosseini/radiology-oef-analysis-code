import numpy as np
import pandas as pd
from scipy import stats
data = pd.read_excel("TRIAD_OEF.xlsx")
tau = "mk_adni_suvr_infcg_metaroi"
outcome = "OEF_PCC_precuneus"
covariates = ["age_at_mri", "Sex", "APOE4"]
mediators = [
    ("AB", "Amyloid-beta PET"),
    ("TSPO", "PBR_PCC_precuneus"),
    ("CCL3", "ccl3"),
    ("IL8", "il8"),
    ("CXCL1", "cxcl1"),
    ("YKL40", "csf_ykl40_pg_ml__elisa"),
]

data["Sex"] = (
    data["gender"].astype(str).str.strip().str.lower().eq("female").astype(float)
)
data["APOE4"] = (
    data["apoe"].astype(str).str.contains("E4", case=False).astype(float)
)


def standardized(values):
    return (values - values.mean()) / values.std(ddof=0)


def regression(frame, outcome_column, predictors):
    design = np.column_stack(
        [np.ones(len(frame)), frame[predictors].to_numpy(dtype=float)]
    )
    values = frame[outcome_column].to_numpy(dtype=float)

    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    residuals = values - design @ beta
    degrees = len(values) - design.shape[1]

    covariance = (
        (residuals @ residuals / degrees)
        * np.linalg.pinv(design.T @ design)
    )
    errors = np.sqrt(np.diag(covariance))
    pvalues = 2 * stats.t.sf(np.abs(beta / errors), degrees)

    return beta, pvalues


def bootstrap_coefficients(design, values):
    cross = np.swapaxes(design, 1, 2) @ design
    product = np.swapaxes(design, 1, 2) @ values[..., None]
    return (np.linalg.pinv(cross) @ product)[..., 0]


def fit_model(frame, mediator_column, outcome_column, rng):
    columns = [tau, mediator_column, outcome_column, *covariates]

    values = (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .reset_index(drop=True)
    )

    scaled = values.copy()
    scaled[[tau, mediator_column, outcome_column]] = standardized(
        values[[tau, mediator_column, outcome_column]]
    )

    a, p_a = regression(scaled, mediator_column, [tau, *covariates])
    b, p_b = regression(
        scaled, outcome_column, [tau, mediator_column, *covariates]
    )

    raw = values.to_numpy(dtype=float)
    indirect_bootstrap = np.empty(10000)

    for start in range(0, 10000, 500):
        size = min(500, 10000 - start)
        sample = raw[rng.integers(len(raw), size=(size, len(raw)))]

        sample_core = sample[:, :, :3]
        standardized_sample = (
            sample_core - sample_core.mean(axis=1, keepdims=True)
        ) / sample_core.std(axis=1, ddof=0, keepdims=True)

        covariate_sample = sample[:, :, 3:]
        intercept = np.ones((size, len(raw), 1))

        a_design = np.concatenate(
            [intercept, standardized_sample[:, :, 0:1], covariate_sample],
            axis=2,
        )
        b_design = np.concatenate(
            [intercept, standardized_sample[:, :, 0:2], covariate_sample],
            axis=2,
        )

        a_bootstrap = bootstrap_coefficients(
            a_design, standardized_sample[:, :, 1]
        )
        b_bootstrap = bootstrap_coefficients(
            b_design, standardized_sample[:, :, 2]
        )

        indirect_bootstrap[start : start + size] = (
            a_bootstrap[:, 1] * b_bootstrap[:, 2]
        )

    lower, upper = np.quantile(indirect_bootstrap, [0.025, 0.975])

    return {
        "N": len(values),
        "a": a[1],
        "p_a": p_a[1],
        "b": b[2],
        "p_b": p_b[2],
        "c_prime": b[1],
        "p_c_prime": p_b[1],
        "indirect_effect": a[1] * b[2],
        "indirect_95CI_lower": lower,
        "indirect_95CI_upper": upper,
    }


rng = np.random.default_rng(42)
results = []

for mediator_label, mediator_column in mediators:
    results.append(
        {
            "Region": "PCC/precuneus",
            "Mediator": mediator_label,
            **fit_model(data, mediator_column, outcome, rng),
        }
    )

pd.DataFrame(results).to_excel(
    "Mediation_analysis_results.xlsx",
    index=False,
    sheet_name="Mediation",
)
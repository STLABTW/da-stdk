"""KAUST benchmark tables, coverage summaries, and paper figures."""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

try:
    from da_stdk.utils.conformal import compute_conformal_coverage, compute_cqr_qhat
except ImportError:
    compute_cqr_qhat = None
    compute_conformal_coverage = None


def _assign_nearest_center(coords, centers):
    coords = np.asarray(coords)
    centers = np.asarray(centers)
    if coords.ndim != 2 or centers.ndim != 2:
        return None
    d = np.sum((coords[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(d, axis=1)


def _width_stats(width, mask):
    if width is None or mask is None:
        return (np.nan, np.nan)
    vals = width[mask]
    if vals.size == 0:
        return (np.nan, np.nan)
    return (float(np.mean(vals)), float(np.percentile(vals, 90)))


def _per_site_coverage(z_full, q_lo, q_hi, test_mask, qhat_per_site=None):
    if z_full is None or q_lo is None or q_hi is None or test_mask is None:
        return None
    test_mask = test_mask.astype(bool)
    if qhat_per_site is not None:
        qhat_expanded = np.broadcast_to(qhat_per_site, q_lo.shape)
        inside = (z_full >= q_lo - qhat_expanded) & (z_full <= q_hi + qhat_expanded)
    else:
        inside = (z_full >= q_lo) & (z_full <= q_hi)
    n_test = test_mask.sum(axis=0)
    cov = np.where(
        n_test > 0, np.where(test_mask, inside, 0).sum(axis=0) / (n_test + 1e-10), np.nan
    )
    return cov


def _cov_spatial_stats(cov):
    if cov is None:
        return (np.nan, np.nan)
    valid = cov[~np.isnan(cov)]
    if valid.size == 0:
        return (np.nan, np.nan)
    k = max(1, int(0.1 * valid.size))
    worst10 = float(np.mean(np.sort(valid)[:k]))
    return (float(np.std(valid)), worst10)


def _compute_experiment_interval_metrics(exp_dir: Path, result: dict):
    pred_path = exp_dir / "predictions.npz"
    if not pred_path.exists():
        return
    preds_npz = np.load(pred_path, allow_pickle=True)
    if "predictions_quantile" not in preds_npz:
        return

    preds_q = preds_npz["predictions_quantile"]  # (T, S, Q)
    z_full = preds_npz["true"]
    coords = preds_npz["coords"]
    test_mask = preds_npz["test_mask"].astype(bool)

    quantile_levels = result.get("quantile_levels")
    conformal_alpha = result.get("conformal_alpha", 0.1)
    conf_path = exp_dir / "conformal_info.npz"
    qhat_per_center = None
    spatial_centers = None
    if conf_path.exists():
        conf_npz = np.load(conf_path, allow_pickle=True)
        if quantile_levels is None and "quantile_levels" in conf_npz.files:
            quantile_levels = conf_npz["quantile_levels"]
        if "conformal_alpha" in conf_npz.files:
            conformal_alpha = float(conf_npz["conformal_alpha"])
        if "qhat_per_center" in conf_npz.files:
            qhat_per_center = conf_npz["qhat_per_center"]
        if "spatial_centers" in conf_npz.files:
            spatial_centers = conf_npz["spatial_centers"]

    if quantile_levels is None:
        return
    q_levels = np.asarray(quantile_levels)
    q_lo = conformal_alpha / 2
    q_hi = 1.0 - conformal_alpha / 2
    idx_lo = int(np.argmin(np.abs(q_levels - q_lo)))
    idx_hi = int(np.argmin(np.abs(q_levels - q_hi)))
    q_lo_grid = preds_q[:, :, idx_lo]
    q_hi_grid = preds_q[:, :, idx_hi]
    width_nominal = q_hi_grid - q_lo_grid

    mean_w_nom, p90_w_nom = _width_stats(width_nominal, test_mask)
    result["pi_width_nominal_mean"] = mean_w_nom
    result["pi_width_nominal_p90"] = p90_w_nom

    # Post-hoc global conformal for STDK (no conformal_info.npz): use valid_mask as calibration
    if (
        result.get("conformal_qhat") is None
        and "valid_mask" in preds_npz.files
        and compute_cqr_qhat is not None
        and compute_conformal_coverage is not None
    ):
        valid_mask = preds_npz["valid_mask"].astype(bool)
        n_cal = int(valid_mask.sum())
        if n_cal >= 10:
            cal_preds = preds_q[valid_mask]
            cal_y_true = np.asarray(z_full)[valid_mask]
            q_levels_list = q_levels.tolist() if hasattr(q_levels, "tolist") else list(q_levels)
            qhat_posthoc, _ = compute_cqr_qhat(
                cal_preds, cal_y_true, q_levels_list, alpha=float(conformal_alpha)
            )
            result["conformal_qhat"] = float(qhat_posthoc)
            test_preds = preds_q[test_mask]
            test_y = np.asarray(z_full)[test_mask]
            if test_preds.size > 0:
                cov_global = compute_conformal_coverage(
                    test_preds, test_y, q_levels_list, qhat_posthoc, alpha=float(conformal_alpha)
                )
                result["test_coverage_90_conformal_global"] = float(cov_global)

    qhat_global = result.get("conformal_qhat")
    if qhat_global is not None:
        width_global = width_nominal + 2.0 * float(qhat_global)
        mean_w_g, p90_w_g = _width_stats(width_global, test_mask)
        result["pi_width_global_mean"] = mean_w_g
        result["pi_width_global_p90"] = p90_w_g

    qhat_per_site = None
    if qhat_per_center is not None and spatial_centers is not None:
        cluster_ids = _assign_nearest_center(coords, spatial_centers)
        if cluster_ids is not None:
            qhat_per_site = qhat_per_center[cluster_ids]
            width_cluster = width_nominal + 2.0 * qhat_per_site[None, :]
            mean_w_c, p90_w_c = _width_stats(width_cluster, test_mask)
            result["pi_width_cluster_mean"] = mean_w_c
            result["pi_width_cluster_p90"] = p90_w_c

    cov_nom = _per_site_coverage(z_full, q_lo_grid, q_hi_grid, test_mask)
    cov_std_nom, cov_worst10_nom = _cov_spatial_stats(cov_nom)
    result["spatial_cov_std_nominal"] = cov_std_nom
    result["worst10_cov_nominal"] = cov_worst10_nom

    if qhat_per_site is not None:
        cov_cluster = _per_site_coverage(
            z_full, q_lo_grid, q_hi_grid, test_mask, qhat_per_site=qhat_per_site
        )
        cov_std_cluster, cov_worst10_cluster = _cov_spatial_stats(cov_cluster)
        result["spatial_cov_std_cluster"] = cov_std_cluster
        result["worst10_cov_cluster"] = cov_worst10_cluster


def load_table_4_4_results(results_dir: Path):
    """
    Load all results from KAUST experiment runs.

    Returns:
        List of result dictionaries.
    """
    results = []
    results_dir = Path(results_dir)

    for scenario_dir in results_dir.iterdir():
        if not scenario_dir.is_dir():
            continue

        name = scenario_dir.name
        if name.endswith("_DA-STDK"):
            scenario_name = name[:-8]
            model_name = "DA-STDK"
        elif name.endswith("_STDK"):
            scenario_name = name[:-5]
            model_name = "STDK"
        else:
            parts = name.split("_")
            if len(parts) >= 2:
                scenario_name = parts[0]
                model_name = "_".join(parts[1:])
            else:
                continue

        scenario_summary_path = scenario_dir / "scenario_summary.json"
        if scenario_summary_path.exists():
            with open(scenario_summary_path, "r") as f:
                scenario_data = json.load(f)
                results.extend(scenario_data.get("results", []))
        else:
            exp_parent = scenario_dir / "experiments"
            if exp_parent.exists():
                search_dirs = list(exp_parent.iterdir())
            else:
                search_dirs = list(scenario_dir.iterdir())
            for exp_dir in search_dirs:
                if not exp_dir.is_dir():
                    continue
                results_path = exp_dir / "results.json"
                if results_path.exists():
                    with open(results_path, "r") as f:
                        result = json.load(f)
                        result["scenario"] = scenario_name
                        result["model"] = model_name
                        result["exp_dir"] = str(exp_dir)
                        _compute_experiment_interval_metrics(exp_dir, result)
                        results.append(result)

    if len(results) == 0:
        for summary_name in ("table_4_4_summary.json", "result.json"):
            summary_path = results_dir / summary_name
            if not summary_path.exists():
                continue
            with open(summary_path, "r") as f:
                payload = json.load(f)
            if isinstance(payload.get("results"), list) and len(payload["results"]) > 0:
                results = payload["results"]
                break

    return results


def create_table_4_4(results: list):
    """Build KAUST experiment summary and pivots (CRPS + coverage). Returns (pivot_df, pivot_std, summary_df)."""
    df = pd.DataFrame(results)
    summary = []
    for scenario in ["Fixed_Uniform", "Fixed_Clustered", "Random_Uniform", "Random_Clustered"]:
        for model in ["STDK", "DA-STDK"]:
            mask = (df["scenario"] == scenario) & (df["model"] == model)
            subset = df[mask]

            if len(subset) > 0:
                crps_values = subset["test_crps"].values
                mean_crps = np.mean(crps_values)
                std_crps = np.std(crps_values)
                test_cov = subset.get("test_picp_90")
                if test_cov is None:
                    test_cov = subset.get("test_coverage_90")
                test_cov_conf = subset.get("test_coverage_90_conformal")
                test_cov_global = subset.get("test_coverage_90_conformal_global")
                test_cov_cluster = subset.get("test_coverage_90_conformal_cluster")
                qhat_vals = subset.get("conformal_qhat")
                mean_qhat_global = subset.get("mean_qhat_global")
                mean_qhat_cluster = subset.get("mean_qhat_cluster")
                cal_cov = subset.get("calibration_coverage_90")
                if cal_cov is None:
                    cal_cov = subset.get("valid_coverage_90_conformal")

                def _mean_std(series):
                    if series is None:
                        return (np.nan, np.nan)
                    values = pd.to_numeric(series, errors="coerce").dropna().values
                    if len(values) == 0:
                        return (np.nan, np.nan)
                    return (np.mean(values), np.std(values))

                mean_test_cov, std_test_cov = _mean_std(test_cov)
                mean_test_cov_conf, std_test_cov_conf = _mean_std(test_cov_conf)
                mean_test_cov_global, std_test_cov_global = _mean_std(test_cov_global)
                mean_test_cov_cluster, std_test_cov_cluster = _mean_std(test_cov_cluster)
                mean_qhat, std_qhat = _mean_std(qhat_vals)
                mqg, _ = _mean_std(mean_qhat_global)
                mqc, _ = _mean_std(mean_qhat_cluster)
                mean_cal_cov, std_cal_cov = _mean_std(cal_cov)
                gap_nom = abs(mean_test_cov - 0.90) if not np.isnan(mean_test_cov) else np.nan
                gap_global = (
                    abs(mean_test_cov_global - 0.90)
                    if not np.isnan(mean_test_cov_global)
                    else np.nan
                )
                gap_cluster = (
                    abs(mean_test_cov_cluster - 0.90)
                    if not np.isnan(mean_test_cov_cluster)
                    else np.nan
                )
                mean_w_nom, std_w_nom = _mean_std(subset.get("pi_width_nominal_mean"))
                mean_wg, std_wg = _mean_std(subset.get("pi_width_global_mean"))
                mean_wc, std_wc = _mean_std(subset.get("pi_width_cluster_mean"))
                p90_w_nom, _ = _mean_std(subset.get("pi_width_nominal_p90"))
                p90_wg, _ = _mean_std(subset.get("pi_width_global_p90"))
                p90_wc, _ = _mean_std(subset.get("pi_width_cluster_p90"))
                mean_cov_std_nom, _ = _mean_std(subset.get("spatial_cov_std_nominal"))
                mean_cov_std_cluster, _ = _mean_std(subset.get("spatial_cov_std_cluster"))
                mean_worst10_nom, _ = _mean_std(subset.get("worst10_cov_nominal"))
                mean_worst10_cluster, _ = _mean_std(subset.get("worst10_cov_cluster"))
                mean_qice, std_qice = _mean_std(subset.get("test_qice"))
                crps_w = subset.get("crps_weighting")
                if crps_w is None or (
                    isinstance(crps_w, (int, float)) and (pd.isna(crps_w) or crps_w != crps_w)
                ):
                    crps_weighting = "trapezoidal"
                else:
                    crps_weighting = (
                        str(list(np.atleast_1d(crps_w).flat)[0])
                        if np.atleast_1d(crps_w).size
                        else "trapezoidal"
                    )

                summary.append(
                    {
                        "Observation Scenario": scenario.replace("_", " "),
                        "Observation Distribution": scenario.split("_")[1],
                        "Model": model,
                        "model_backbone": model,
                        "calibration_mode": "backbone",
                        "method_id": "stdk_backbone" if model == "STDK" else "ours_backbone",
                        "Mean CRPS": mean_crps,
                        "Std CRPS": std_crps,
                        "crps_weighting": crps_weighting,
                        "Mean PICP 90": mean_test_cov,
                        "Std PICP 90": std_test_cov,
                        "Mean Test Coverage 90": mean_test_cov,
                        "Std Test Coverage 90": std_test_cov,
                        "Mean Test Coverage 90 (Conformal)": mean_test_cov_conf,
                        "Std Test Coverage 90 (Conformal)": std_test_cov_conf,
                        "Mean Test Coverage 90 (Conformal Global)": mean_test_cov_global,
                        "Std Test Coverage 90 (Conformal Global)": std_test_cov_global,
                        "Mean Test Coverage 90 (Conformal Cluster)": mean_test_cov_cluster,
                        "Std Test Coverage 90 (Conformal Cluster)": std_test_cov_cluster,
                        "Mean Conformal Qhat": mean_qhat,
                        "Std Conformal Qhat": std_qhat,
                        "Mean Qhat Global": mqg,
                        "Mean Qhat Cluster": mqc,
                        "Mean Calibration Coverage 90": mean_cal_cov,
                        "Std Calibration Coverage 90": std_cal_cov,
                        "Coverage Gap 90 (Nominal)": gap_nom,
                        "Coverage Gap 90 (Conformal Global)": gap_global,
                        "Coverage Gap 90 (Conformal Cluster)": gap_cluster,
                        "Mean PI Width (Nominal)": mean_w_nom,
                        "Mean PI Width (Global)": mean_wg,
                        "Mean PI Width (Cluster)": mean_wc,
                        "P90 PI Width (Nominal)": p90_w_nom,
                        "P90 PI Width (Global)": p90_wg,
                        "P90 PI Width (Cluster)": p90_wc,
                        "Spatial Coverage Std (Nominal)": mean_cov_std_nom,
                        "Spatial Coverage Std (Cluster)": mean_cov_std_cluster,
                        "Worst10 Coverage (Nominal)": mean_worst10_nom,
                        "Worst10 Coverage (Cluster)": mean_worst10_cluster,
                        "Mean QICE": mean_qice,
                        "Std QICE": std_qice,
                        "N": len(subset),
                    }
                )

    summary_df = pd.DataFrame(summary)
    pivot_df = summary_df.pivot_table(
        index=["Observation Scenario", "Observation Distribution"],
        columns="Model",
        values="Mean CRPS",
        aggfunc="first",
    )
    pivot_std = summary_df.pivot_table(
        index=["Observation Scenario", "Observation Distribution"],
        columns="Model",
        values="Std CRPS",
        aggfunc="first",
    )
    row_order = [
        ("Fixed", "Uniform"),
        ("Fixed", "Clustered"),
        ("Random", "Uniform"),
        ("Random", "Clustered"),
    ]
    idx = pd.MultiIndex.from_tuples(
        [(f"{s} {d}", d) for s, d in row_order],
        names=["Observation Scenario", "Observation Distribution"],
    )
    pivot_df = pivot_df.reindex(idx).dropna(how="all")
    pivot_std = pivot_std.reindex(idx).dropna(how="all")
    pivot_df = pivot_df.reindex(columns=["STDK", "DA-STDK"])
    pivot_std = pivot_std.reindex(columns=["STDK", "DA-STDK"])
    return pivot_df, pivot_std, summary_df


def build_summary_by_method(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 4-method summary: one row per (Observation Scenario, method_id).
    method_id in {stdk_nominal, stdk_global, ours_global, ours_cluster}.
    """
    rows = []
    scenarios = summary_df["Observation Scenario"].unique()
    for scenario in scenarios:
        stdk = summary_df[
            (summary_df["Observation Scenario"] == scenario) & (summary_df["Model"] == "STDK")
        ]
        da = summary_df[
            (summary_df["Observation Scenario"] == scenario) & (summary_df["Model"] == "DA-STDK")
        ]
        stdk_row = stdk.iloc[0] if len(stdk) else None
        da_row = da.iloc[0] if len(da) else None
        dist = (
            stdk_row["Observation Distribution"]
            if stdk_row is not None
            else (da_row["Observation Distribution"] if da_row is not None else "")
        )
        crps_stdk = stdk_row["Mean CRPS"] if stdk_row is not None else np.nan
        std_crps_stdk = stdk_row["Std CRPS"] if stdk_row is not None else np.nan
        crps_da = da_row["Mean CRPS"] if da_row is not None else np.nan
        std_crps_da = da_row["Std CRPS"] if da_row is not None else np.nan
        crps_weighting = (
            stdk_row.get("crps_weighting", "trapezoidal")
            if stdk_row is not None
            else (
                da_row.get("crps_weighting", "trapezoidal") if da_row is not None else "trapezoidal"
            )
        )

        def _cov(row, key_mean, key_std):
            if row is None:
                return np.nan, np.nan
            m = row.get(key_mean, np.nan)
            s = row.get(key_std, np.nan)
            return (float(m) if pd.notna(m) else np.nan), (float(s) if pd.notna(s) else np.nan)

        cov_nom_s, std_nom_s = _cov(stdk_row, "Mean Test Coverage 90", "Std Test Coverage 90")
        rows.append(
            {
                "Observation Scenario": scenario,
                "Observation Distribution": dist,
                "method_id": "stdk_nominal",
                "model_backbone": "STDK",
                "calibration_mode": "nominal",
                "Mean CRPS": crps_stdk,
                "Std CRPS": std_crps_stdk,
                "Coverage": cov_nom_s,
                "Std Coverage": std_nom_s,
                "crps_weighting": crps_weighting,
            }
        )
        cov_glob_s, std_glob_s = _cov(
            stdk_row,
            "Mean Test Coverage 90 (Conformal Global)",
            "Std Test Coverage 90 (Conformal Global)",
        )
        rows.append(
            {
                "Observation Scenario": scenario,
                "Observation Distribution": dist,
                "method_id": "stdk_global",
                "model_backbone": "STDK",
                "calibration_mode": "global",
                "Mean CRPS": crps_stdk,
                "Std CRPS": std_crps_stdk,
                "Coverage": cov_glob_s,
                "Std Coverage": std_glob_s,
                "crps_weighting": crps_weighting,
            }
        )
        cov_glob_d, std_glob_d = _cov(
            da_row,
            "Mean Test Coverage 90 (Conformal Global)",
            "Std Test Coverage 90 (Conformal Global)",
        )
        rows.append(
            {
                "Observation Scenario": scenario,
                "Observation Distribution": dist,
                "method_id": "ours_global",
                "model_backbone": "DA-STDK",
                "calibration_mode": "global",
                "Mean CRPS": crps_da,
                "Std CRPS": std_crps_da,
                "Coverage": cov_glob_d,
                "Std Coverage": std_glob_d,
                "crps_weighting": crps_weighting,
            }
        )
        cov_clust_d, std_clust_d = _cov(
            da_row,
            "Mean Test Coverage 90 (Conformal Cluster)",
            "Std Test Coverage 90 (Conformal Cluster)",
        )
        if np.isnan(cov_clust_d) and da_row is not None:
            cov_clust_d = da_row.get("Mean Test Coverage 90 (Conformal)", np.nan)
            std_clust_d = da_row.get("Std Test Coverage 90 (Conformal)", np.nan)
        rows.append(
            {
                "Observation Scenario": scenario,
                "Observation Distribution": dist,
                "method_id": "ours_cluster",
                "model_backbone": "DA-STDK",
                "calibration_mode": "cluster",
                "Mean CRPS": crps_da,
                "Std CRPS": std_crps_da,
                "Coverage": cov_clust_d,
                "Std Coverage": std_clust_d,
                "crps_weighting": crps_weighting,
            }
        )
    return pd.DataFrame(rows)


def create_cluster_gain_plot(summary_df: pd.DataFrame, output_path: Path):
    """
    Cluster scenarios only: gain in PICP / Worst10 / QICE (e.g. Ours cluster vs STDK nominal).
    """
    clustered = summary_df[summary_df["Observation Distribution"] == "Clustered"].copy()
    if clustered.empty:
        return None
    scenarios = clustered["Observation Scenario"].unique()
    rows = []
    for sc in scenarios:
        stdk = clustered[(clustered["Observation Scenario"] == sc) & (clustered["Model"] == "STDK")]
        da = clustered[
            (clustered["Observation Scenario"] == sc) & (clustered["Model"] == "DA-STDK")
        ]
        if stdk.empty or da.empty:
            continue
        s, d = stdk.iloc[0], da.iloc[0]
        picp_stdk = s.get("Mean Test Coverage 90", np.nan)
        picp_da = d.get(
            "Mean Test Coverage 90 (Conformal Cluster)",
            d.get("Mean Test Coverage 90 (Conformal)", np.nan),
        )
        w10_stdk = s.get("Worst10 Coverage (Nominal)", np.nan)
        w10_da = d.get("Worst10 Coverage (Cluster)", np.nan)
        qice_stdk = s.get("Mean QICE", np.nan)
        qice_da = d.get("Mean QICE", np.nan)
        rows.append(
            {
                "Scenario": sc,
                "PICP (STDK nom)": picp_stdk,
                "PICP (Ours cluster)": picp_da,
                "Worst10 (STDK nom)": w10_stdk,
                "Worst10 (Ours cluster)": w10_da,
                "QICE (STDK nom)": qice_stdk,
                "QICE (Ours cluster)": qice_da,
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
    x = np.arange(len(df))
    w = 0.35
    for ax, (metric, lb_nom, lb_ours) in zip(
        axes,
        [
            ("PICP", "PICP (STDK nom)", "PICP (Ours cluster)"),
            ("Worst10", "Worst10 (STDK nom)", "Worst10 (Ours cluster)"),
            ("QICE", "QICE (STDK nom)", "QICE (Ours cluster)"),
        ],
    ):
        ax.bar(x - w / 2, df[lb_nom], width=w, label="STDK nominal", color="#4C78A8")
        ax.bar(x + w / 2, df[lb_ours], width=w, label="Ours (cluster)", color="#E45756")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Scenario"], rotation=15, ha="right")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Clustered scenarios: PICP / Worst10 / QICE", fontsize=12, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_coverage_comparison_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare STDK nominal vs DA-STDK conformal-cluster coverage summary."""

    def _first_float(row: pd.DataFrame, column: str) -> float:
        if column not in row.columns or row.empty:
            return np.nan
        vals = pd.to_numeric(row[column], errors="coerce").dropna().values
        return float(vals[0]) if len(vals) > 0 else np.nan

    row_order = [
        ("Fixed Uniform", "Uniform"),
        ("Fixed Clustered", "Clustered"),
        ("Random Uniform", "Uniform"),
        ("Random Clustered", "Clustered"),
    ]
    rows = []
    for scenario_name, distribution in row_order:
        stdk_row = summary_df[
            (summary_df["Observation Scenario"] == scenario_name)
            & (summary_df["Observation Distribution"] == distribution)
            & (summary_df["Model"] == "STDK")
        ]
        da_row = summary_df[
            (summary_df["Observation Scenario"] == scenario_name)
            & (summary_df["Observation Distribution"] == distribution)
            & (summary_df["Model"] == "DA-STDK")
        ]
        if stdk_row.empty or da_row.empty:
            continue

        stdk_cov = _first_float(stdk_row, "Mean Test Coverage 90")
        stdk_std = _first_float(stdk_row, "Std Test Coverage 90")
        da_cov = _first_float(da_row, "Mean Test Coverage 90 (Conformal Cluster)")
        da_std = _first_float(da_row, "Std Test Coverage 90 (Conformal Cluster)")
        diff_cov = da_cov - stdk_cov
        if np.isfinite(stdk_std) and np.isfinite(da_std):
            diff_std = np.sqrt(stdk_std**2 + da_std**2)
        else:
            diff_std = np.nan

        rows.append(
            {
                "Scenario": f"{scenario_name.split(' ')[0]} / {distribution}",
                "STDK Nominal Coverage": stdk_cov,
                "STDK Nominal Std": stdk_std,
                "DA-STDK Conformal Cluster Coverage": da_cov,
                "DA-STDK Conformal Cluster Std": da_std,
                "Coverage Difference (DA_cluster - STDK_nominal)": diff_cov,
                "Coverage Difference Std (approx)": diff_std,
                "Coverage Difference (pp)": diff_cov * 100.0,
            }
        )
    return pd.DataFrame(rows)


def create_coverage_4methods_plot(summary_by_method_df: pd.DataFrame, output_path: Path):
    """
    Coverage bar plot: 4 methods x 4 scenarios (with error bars).
    method_id: stdk_nominal, stdk_global, ours_global, ours_cluster.
    Returns (output_path, summary_by_method_df).
    """
    if summary_by_method_df.empty or "method_id" not in summary_by_method_df.columns:
        return None, None
    scenario_order = ["Fixed Uniform", "Fixed Clustered", "Random Uniform", "Random Clustered"]
    method_order = ["stdk_nominal", "stdk_global", "ours_global", "ours_cluster"]
    method_labels = {
        "stdk_nominal": "STDK Nominal",
        "stdk_global": "STDK + Global CQR",
        "ours_global": "Ours-global",
        "ours_cluster": "Ours (cluster)",
    }
    colors = {
        "stdk_nominal": "#4C78A8",
        "stdk_global": "#72B7B2",
        "ours_global": "#F58518",
        "ours_cluster": "#E45756",
    }
    df = summary_by_method_df[
        summary_by_method_df["Observation Scenario"].isin(scenario_order)
    ].copy()
    df["Scenario"] = df["Observation Scenario"]
    x_labels = [s.replace(" ", "\n") for s in scenario_order]
    n_scenarios = len(scenario_order)
    n_methods = len(method_order)
    width = 1.0 / (n_methods + 1)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    x = np.arange(n_scenarios)
    for i, mid in enumerate(method_order):
        vals = []
        errs = []
        for sc in scenario_order:
            row = df[(df["Observation Scenario"] == sc) & (df["method_id"] == mid)]
            if len(row) > 0:
                c = row["Coverage"].iloc[0]
                e = (
                    row["Std Coverage"].iloc[0]
                    if "Std Coverage" in row.columns and pd.notna(row["Std Coverage"].iloc[0])
                    else 0.0
                )
                vals.append(float(c) if pd.notna(c) else np.nan)
                errs.append(float(e) if pd.notna(e) else 0.0)
            else:
                vals.append(np.nan)
                errs.append(0.0)
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width=width * 0.9,
            yerr=errs,
            capsize=2,
            label=method_labels.get(mid, mid),
            color=colors.get(mid, None),
        )
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.0, alpha=0.8, label="Nominal 90%")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Coverage")
    cov_vals = df["Coverage"].dropna()
    y_min = float(cov_vals.min() - 0.06) if len(cov_vals) else 0.7
    ax.set_ylim(max(0.0, y_min), min(1.0, 0.98))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Coverage @90%: 4 methods × 4 scenarios", fontsize=12, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path, summary_by_method_df


def create_coverage_comparison_plot(summary_df: pd.DataFrame, output_path: Path):
    """
    Plot STDK nominal vs DA-STDK conformal-cluster coverage and coverage difference.
    Returns (output_path, comp_df) or (None, None) if no data.
    """
    comp_df = _build_coverage_comparison_df(summary_df)
    if comp_df.empty:
        return None, None

    x = np.arange(len(comp_df))
    width = 0.36
    nominal_target = 0.90

    stdk_cov = comp_df["STDK Nominal Coverage"].to_numpy()
    da_cov = comp_df["DA-STDK Conformal Cluster Coverage"].to_numpy()
    stdk_std = np.nan_to_num(comp_df["STDK Nominal Std"].to_numpy(), nan=0.0)
    da_std = np.nan_to_num(comp_df["DA-STDK Conformal Cluster Std"].to_numpy(), nan=0.0)
    diff_cov = comp_df["Coverage Difference (DA_cluster - STDK_nominal)"].to_numpy()
    diff_std = np.nan_to_num(comp_df["Coverage Difference Std (approx)"].to_numpy(), nan=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)

    ax = axes[0]
    ax.bar(
        x - width / 2,
        stdk_cov,
        width=width,
        yerr=stdk_std,
        capsize=3,
        color="#4C78A8",
        label="STDK Nominal",
    )
    ax.bar(
        x + width / 2,
        da_cov,
        width=width,
        yerr=da_std,
        capsize=3,
        color="#F58518",
        label="DA-STDK Conformal Cluster",
    )
    ax.axhline(
        nominal_target, color="black", linestyle="--", linewidth=1.0, alpha=0.8, label="Nominal 90%"
    )
    y_min = max(0.0, min(np.min(stdk_cov), np.min(da_cov), nominal_target) - 0.06)
    y_max = min(1.0, max(np.max(stdk_cov), np.max(da_cov), nominal_target) + 0.04)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Scenario"], rotation=15, ha="right")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage @90%")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    ax = axes[1]
    colors_bar = np.where(diff_cov >= 0, "#54A24B", "#E45756")
    ax.bar(x, diff_cov, width=0.55, yerr=diff_std, capsize=3, color=colors_bar)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Scenario"], rotation=15, ha="right")
    ax.set_ylabel("Coverage Difference")
    ax.set_title("DA-STDK Cluster - STDK Nominal")
    ax.grid(axis="y", alpha=0.25)

    for i, val in enumerate(diff_cov):
        ax.text(
            i,
            val + (0.002 if val >= 0 else -0.002),
            f"{val * 100:.2f} pp",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )

    fig.suptitle("KAUST experiment: Coverage Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path, comp_df


def _nearest_site_qhat(
    coords: np.ndarray, spatial_centers: np.ndarray, qhat_per_center: np.ndarray
):
    """Map per-center qhat to per-site qhat via nearest-center assignment."""
    cluster_ids = _assign_nearest_center(coords, spatial_centers)
    if cluster_ids is None:
        return None
    return qhat_per_center[cluster_ids]


def _load_quantile_interval(exp_dir: Path, result: dict):
    """Load quantile interval bounds and test mask for one experiment."""
    pred_path = exp_dir / "predictions.npz"
    if not pred_path.exists():
        return None
    preds_npz = np.load(pred_path, allow_pickle=True)
    if "predictions_quantile" not in preds_npz.files:
        return None

    preds_q = preds_npz["predictions_quantile"]  # (T, S, Q)
    z_full = preds_npz["true"]
    coords = preds_npz["coords"]
    test_mask = preds_npz["test_mask"].astype(bool)

    quantile_levels = result.get("quantile_levels")
    conformal_alpha = result.get("conformal_alpha", 0.1)
    conf_path = exp_dir / "conformal_info.npz"
    if conf_path.exists():
        conf_npz = np.load(conf_path, allow_pickle=True)
        if quantile_levels is None and "quantile_levels" in conf_npz.files:
            quantile_levels = conf_npz["quantile_levels"]
        if "conformal_alpha" in conf_npz.files:
            conformal_alpha = float(conf_npz["conformal_alpha"])
    if quantile_levels is None:
        return None

    q_levels = np.asarray(quantile_levels)
    q_lo = conformal_alpha / 2.0
    q_hi = 1.0 - conformal_alpha / 2.0
    idx_lo = int(np.argmin(np.abs(q_levels - q_lo)))
    idx_hi = int(np.argmin(np.abs(q_levels - q_hi)))
    q_lo_grid = preds_q[:, :, idx_lo]
    q_hi_grid = preds_q[:, :, idx_hi]

    return {
        "q_lo_grid": q_lo_grid,
        "q_hi_grid": q_hi_grid,
        "z_full": z_full,
        "coords": coords,
        "test_mask": test_mask,
    }


def _compute_scenario_spatial_comparison(results: list, scenario: str):
    """
    Build per-site averaged coverage arrays:
      STDK nominal, DA-STDK conformal cluster, delta = DA cluster - STDK nominal.
    """
    stdk_results = [
        r for r in results if r.get("scenario") == scenario and r.get("model") == "STDK"
    ]
    da_results = [
        r for r in results if r.get("scenario") == scenario and r.get("model") == "DA-STDK"
    ]
    if len(stdk_results) == 0 or len(da_results) == 0:
        return None

    stdk_cov_list = []
    da_cov_cluster_list = []
    coords_ref = None
    centers_ref = None

    for result in stdk_results:
        exp_dir = Path(result.get("exp_dir", ""))
        if not exp_dir.exists():
            continue
        loaded = _load_quantile_interval(exp_dir, result)
        if loaded is None:
            continue
        cov_nom = _per_site_coverage(
            loaded["z_full"],
            loaded["q_lo_grid"],
            loaded["q_hi_grid"],
            loaded["test_mask"],
            qhat_per_site=None,
        )
        if cov_nom is None:
            continue
        stdk_cov_list.append(cov_nom)
        if coords_ref is None:
            coords_ref = loaded["coords"]

    for result in da_results:
        exp_dir = Path(result.get("exp_dir", ""))
        if not exp_dir.exists():
            continue
        loaded = _load_quantile_interval(exp_dir, result)
        conf_path = exp_dir / "conformal_info.npz"
        if loaded is None or not conf_path.exists():
            continue
        conf_npz = np.load(conf_path, allow_pickle=True)
        if "qhat_per_center" not in conf_npz.files or "spatial_centers" not in conf_npz.files:
            continue
        qhat_per_center = conf_npz["qhat_per_center"]
        spatial_centers = conf_npz["spatial_centers"]
        qhat_per_site = _nearest_site_qhat(loaded["coords"], spatial_centers, qhat_per_center)
        if qhat_per_site is None:
            continue
        cov_cluster = _per_site_coverage(
            loaded["z_full"],
            loaded["q_lo_grid"],
            loaded["q_hi_grid"],
            loaded["test_mask"],
            qhat_per_site=qhat_per_site,
        )
        if cov_cluster is None:
            continue
        da_cov_cluster_list.append(cov_cluster)
        if coords_ref is None:
            coords_ref = loaded["coords"]
        if centers_ref is None:
            centers_ref = spatial_centers

    if len(stdk_cov_list) == 0 or len(da_cov_cluster_list) == 0 or coords_ref is None:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stdk_nom_avg = np.nanmean(np.asarray(stdk_cov_list), axis=0)
        da_cluster_avg = np.nanmean(np.asarray(da_cov_cluster_list), axis=0)
    delta = da_cluster_avg - stdk_nom_avg

    return {
        "coords": coords_ref,
        "centers": centers_ref,
        "stdk_nominal_avg": stdk_nom_avg,
        "da_cluster_avg": da_cluster_avg,
        "delta": delta,
        "n_stdk": len(stdk_cov_list),
        "n_da": len(da_cov_cluster_list),
    }


def _scenario_label(scenario: str) -> str:
    return scenario.replace("_", " / ")


def _create_paper_main_figures(
    results: list, results_dir: Path, summary_by_method: pd.DataFrame, out_dir: Path
):
    """Produce the 2 extra main paper figures (cluster_gain, spatial_delta_clustered)."""
    _, _, summary_df = create_table_4_4(results)
    create_cluster_gain_plot(summary_df, out_dir / "fig_main_cluster_gain.png")
    scenarios_clustered = ["Fixed_Clustered", "Random_Clustered"]
    fig_list = []
    xi = np.linspace(0.0, 1.0, 200)
    yi = np.linspace(0.0, 1.0, 200)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    for scenario in scenarios_clustered:
        comp = _compute_scenario_spatial_comparison(results, scenario)
        if comp is None:
            continue
        coords, delta = comp["coords"], comp["delta"]
        valid = ~np.isnan(delta)
        if not np.any(valid):
            continue
        delta_grid = griddata(coords[valid], delta[valid], (xi_grid, yi_grid), method="nearest")
        fig_list.append((scenario, delta_grid))
    if len(fig_list) >= 1:
        n_plots = len(fig_list)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), dpi=160)
        if n_plots == 1:
            axes = [axes]
        for ax, (scenario, data) in zip(axes, fig_list):
            finite = data[np.isfinite(data)]
            v = max(np.max(np.abs(finite)) if finite.size else 0.05, 0.05)
            im = ax.pcolormesh(xi_grid, yi_grid, data, cmap="RdBu", shading="auto", vmin=-v, vmax=v)
            ax.set_title(_scenario_label(scenario))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.suptitle(
            "Spatial coverage delta (DA cluster − STDK nominal), clustered scenarios",
            fontsize=12,
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "fig_main_spatial_delta_clustered.png", bbox_inches="tight")
        plt.close(fig)


def create_spatial_coverage_comparison_plots(results: list, results_dir: Path):
    """
    For each scenario, create a 3-panel map:
      STDK nominal | DA-STDK conformal cluster | Delta (DA - STDK).
    Returns list of (png_path, csv_path) generated.
    """
    scenarios = ["Fixed_Uniform", "Fixed_Clustered", "Random_Uniform", "Random_Clustered"]
    out_dir = Path(results_dir) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    xi = np.linspace(0.0, 1.0, 200)
    yi = np.linspace(0.0, 1.0, 200)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    for scenario in scenarios:
        comp = _compute_scenario_spatial_comparison(results, scenario)
        if comp is None:
            continue
        coords = comp["coords"]
        centers = comp["centers"]
        stdk_nom = comp["stdk_nominal_avg"]
        da_cluster = comp["da_cluster_avg"]
        delta = comp["delta"]

        valid = ~np.isnan(stdk_nom) & ~np.isnan(da_cluster) & ~np.isnan(delta)
        if not np.any(valid):
            continue

        stdk_grid = griddata(coords[valid], stdk_nom[valid], (xi_grid, yi_grid), method="nearest")
        da_grid = griddata(coords[valid], da_cluster[valid], (xi_grid, yi_grid), method="nearest")
        delta_grid = griddata(coords[valid], delta[valid], (xi_grid, yi_grid), method="nearest")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=160)
        title_base = _scenario_label(scenario)

        panels = [
            (axes[0], stdk_grid, "STDK Nominal Coverage", "RdYlGn", 0.5, 1.0),
            (axes[1], da_grid, "DA-STDK Conformal Cluster Coverage", "RdYlGn", 0.5, 1.0),
            (axes[2], delta_grid, "Delta (DA Cluster - STDK Nominal)", "RdBu", None, None),
        ]

        for ax, data, panel_title, cmap, vmin, vmax in panels:
            if vmin is None or vmax is None:
                finite_data = data[np.isfinite(data)]
                if finite_data.size == 0:
                    vmin_local, vmax_local = -0.05, 0.05
                else:
                    max_abs = max(float(np.max(np.abs(finite_data))), 1e-4)
                    vmin_local, vmax_local = -max_abs, max_abs
            else:
                vmin_local, vmax_local = vmin, vmax

            im = ax.pcolormesh(
                xi_grid,
                yi_grid,
                data,
                cmap=cmap,
                shading="auto",
                vmin=vmin_local,
                vmax=vmax_local,
            )
            if centers is not None:
                ax.scatter(centers[:, 0], centers[:, 1], c="red", s=15, marker="x", alpha=0.6)
            ax.set_title(panel_title, fontsize=12, fontweight="bold")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

        fig.suptitle(
            f"Spatial Coverage Comparison ({title_base})  [N: STDK={comp['n_stdk']}, DA={comp['n_da']}]",
            fontsize=14,
            y=0.98,
        )
        fig.tight_layout()

        png_path = out_dir / f"spatial_coverage_compare_{scenario.lower()}.png"
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)

        csv_path = out_dir / f"spatial_coverage_compare_{scenario.lower()}.csv"
        summary_rows = pd.DataFrame(
            [
                {
                    "Scenario": scenario,
                    "N_STDK": comp["n_stdk"],
                    "N_DA": comp["n_da"],
                    "Mean STDK Nominal Coverage": float(np.nanmean(stdk_nom)),
                    "Mean DA Cluster Coverage": float(np.nanmean(da_cluster)),
                    "Mean Delta (DA- STDK)": float(np.nanmean(delta)),
                    "Median Delta (DA- STDK)": float(np.nanmedian(delta)),
                    "P10 Delta (DA- STDK)": float(np.nanpercentile(delta, 10)),
                    "P90 Delta (DA- STDK)": float(np.nanpercentile(delta, 90)),
                    "Max Abs Delta": float(np.nanmax(np.abs(delta))),
                }
            ]
        )
        summary_rows.to_csv(csv_path, index=False)
        generated.append((png_path, csv_path))

    return generated


def create_paper_main_figures(
    results: list, results_dir: Path, summary_by_method: pd.DataFrame, out_dir: Path
):
    """
    Produce the 2 extra main paper figures (cluster_gain, spatial_delta_clustered).
    The coverage 4-methods figure is produced by the caller.
    """
    _create_paper_main_figures(results, Path(results_dir), summary_by_method, Path(out_dir))


def print_table_4_4(pivot_df, pivot_std, summary_df):
    """Print KAUST experiment CRPS and coverage summary to stdout."""
    print("\n" + "=" * 80)
    print("KAUST EXPERIMENT: CRPS Comparison between STDK and DA-STDK")
    print("(Multi-quantile joint training, 2b-8 dataset)")
    print("=" * 80)
    print()
    print("KAUST experiment (Mean ± Std CRPS):")
    print("-" * 80)
    print(f"{'Scenario':<24} {'STDK':>14} {'DA-STDK':>14}")
    print("-" * 80)
    for idx in pivot_df.index:
        scenario, distribution = idx
        design = scenario.replace(" " + distribution, "")
        label = f"{design} / {distribution}"

        def _fmt(col):
            m, s = pivot_df.loc[idx, col], pivot_std.loc[idx, col]
            if pd.isna(m) or pd.isna(s):
                return "      —"
            return f"{m:.4f} ± {s:.4f}"

        print(f"{label:<24} {_fmt('STDK'):>14}  {_fmt('DA-STDK'):>14}")
    print()
    print("Summary Statistics (all scenarios × models):")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    print()
    print("Pivot (Mean CRPS only):")
    print("-" * 80)
    print(pivot_df.to_string())
    print()
    if "Mean Test Coverage 90 (Conformal)" in summary_df.columns:
        print("Coverage Summary (Mean ± Std):")
        print("-" * 80)
        cols = [
            "Observation Scenario",
            "Observation Distribution",
            "Model",
            "N",
            "Mean PICP 90",
            "Std PICP 90",
            "Mean Test Coverage 90",
            "Std Test Coverage 90",
            "Mean Test Coverage 90 (Conformal)",
            "Std Test Coverage 90 (Conformal)",
            "Mean Test Coverage 90 (Conformal Global)",
            "Std Test Coverage 90 (Conformal Global)",
            "Mean Test Coverage 90 (Conformal Cluster)",
            "Std Test Coverage 90 (Conformal Cluster)",
            "Mean QICE",
            "Std QICE",
            "Mean Conformal Qhat",
            "Std Conformal Qhat",
            "Mean Qhat Global",
            "Mean Qhat Cluster",
            "Mean Calibration Coverage 90",
            "Std Calibration Coverage 90",
            "Coverage Gap 90 (Nominal)",
            "Coverage Gap 90 (Conformal Global)",
            "Coverage Gap 90 (Conformal Cluster)",
            "Mean PI Width (Nominal)",
            "Mean PI Width (Global)",
            "Mean PI Width (Cluster)",
            "P90 PI Width (Nominal)",
            "P90 PI Width (Global)",
            "P90 PI Width (Cluster)",
            "Spatial Coverage Std (Nominal)",
            "Spatial Coverage Std (Cluster)",
            "Worst10 Coverage (Nominal)",
            "Worst10 Coverage (Cluster)",
        ]
        available = [c for c in cols if c in summary_df.columns]
        print(summary_df[available].to_string(index=False))

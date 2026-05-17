"""Spatial MSE and coverage maps."""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

from da_stdk.utils.conformal import _assign_nearest_center


def plot_spatial_mse(
    model,
    z_full,
    coords,
    train_mask,
    device,
    output_dir,
    return_predictions=False,
    valid_mask=None,
    test_mask=None,
):
    """
    Plot spatial MSE heatmap averaged over all time points.

    Args:
        model: trained model
        z_full: (T, S) full data
        coords: (S, 2) coordinates
        train_mask: (T, S) training mask
        device: torch device
        output_dir: output directory
        return_predictions: if True, return true and pred values
        valid_mask: (T, S) validation mask (optional, for saving)
        test_mask: (T, S) test mask (optional, for saving)

    Returns:
        If return_predictions=True: (all_predictions, z_full, coords, train_mask, valid_mask, test_mask)
        Otherwise: None
    """
    output_dir = Path(output_dir)
    T, S = z_full.shape

    # Get spatial basis centers from model
    spatial_centers = model.spatial_basis.centers.detach().cpu().numpy()  # (k_spatial, 2)
    spatial_bandwidths = model.spatial_basis.bandwidths.detach().cpu().numpy()  # (k_spatial,)

    # Size basis markers proportional to bandwidth
    bw_normalized = (spatial_bandwidths - spatial_bandwidths.min()) / (
        spatial_bandwidths.max() - spatial_bandwidths.min() + 1e-8
    )
    basis_sizes = 10 + bw_normalized * 90  # Range [10, 100]

    # Generate predictions for all sites at all times
    model.eval()
    all_predictions = np.zeros((T, S))

    with torch.no_grad():
        for t_idx in range(T):
            t_normalized = t_idx / (T - 1) if T > 1 else 0.0
            t_tensor = torch.tensor([[t_normalized]], dtype=torch.float32).repeat(S, 1).to(device)
            coords_tensor = torch.from_numpy(coords).float().to(device)
            X_tensor = torch.zeros(S, 0).to(device)  # No covariates

            y_pred = model(X_tensor, coords_tensor, t_tensor).cpu().numpy()  # (S, output_dim)

            # For multi-quantile, use median quantile; otherwise use single output
            if y_pred.shape[1] > 1:  # Multi-quantile
                median_idx = y_pred.shape[1] // 2
                y_pred = y_pred[:, median_idx]
            else:
                y_pred = y_pred.flatten()

            all_predictions[t_idx, :] = y_pred

    # Compute MSE per site (averaged over time)
    squared_errors = (all_predictions - z_full) ** 2
    site_mse = np.nanmean(squared_errors, axis=0)  # (S,)

    # Get all train sites (any time)
    train_sites_any = np.where(train_mask.any(axis=0))[0]
    train_coords_any = coords[train_sites_any]

    # Valid sites (not all NaN)
    valid_sites = ~np.isnan(site_mse)
    coords_valid = coords[valid_sites]
    site_mse_valid = site_mse[valid_sites]

    # Create grid for interpolation
    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate MSE to grid using nearest neighbor
    mse_grid = griddata(coords_valid, site_mse_valid, (xi_grid, yi_grid), method="nearest")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.pcolormesh(xi_grid, yi_grid, mse_grid, cmap="YlOrRd", shading="auto")
    ax.scatter(
        train_coords_any[:, 0],
        train_coords_any[:, 1],
        c="black",
        s=25,
        alpha=0.6,
        label="Train sites",
        edgecolors="white",
        linewidths=0.5,
    )
    ax.scatter(
        spatial_centers[:, 0],
        spatial_centers[:, 1],
        c="red",
        s=basis_sizes,
        marker="x",
        alpha=0.5,
        label="Basis centers",
        linewidths=1.5,
    )

    ax.set_title("Spatial MSE (Averaged over Time)", fontsize=18, fontweight="bold")
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=13)
    cbar = plt.colorbar(im, ax=ax, label="MSE")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("MSE", fontsize=14)

    plt.tight_layout()
    save_path = output_dir / "spatial_mse.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Spatial MSE plot saved to {save_path}")

    if return_predictions:
        return all_predictions, z_full, coords, train_mask, valid_mask, test_mask
    return None


def plot_spatial_coverage_and_qhat(output_dir):
    """
    Plot spatial coverage (nominal vs cluster-aware) and qhat spatial distribution.
    Reads quantile_levels and conformal_alpha from conformal_info.npz (source of truth).
    Requires: predictions.npz with predictions_quantile, conformal_info.npz.
    """
    output_dir = Path(output_dir)
    pred_path = output_dir / "predictions.npz"
    conformal_path = output_dir / "conformal_info.npz"
    if not pred_path.exists() or not conformal_path.exists():
        return
    preds_npz = np.load(pred_path, allow_pickle=True)
    conf_npz = np.load(conformal_path, allow_pickle=True)
    if "predictions_quantile" not in preds_npz:
        return
    preds_q = preds_npz["predictions_quantile"]  # (T, S, Q)
    z_full = preds_npz["true"]
    coords = preds_npz["coords"]
    test_mask = preds_npz["test_mask"]
    qhat_per_center = conf_npz["qhat_per_center"]  # (C,)
    spatial_centers = conf_npz["spatial_centers"]  # (C, 2)
    quantile_levels = np.asarray(conf_npz["quantile_levels"])
    conformal_alpha = (
        float(conf_npz["conformal_alpha"]) if "conformal_alpha" in conf_npz.files else 0.1
    )
    T, S, Q = preds_q.shape
    q_lo, q_hi = conformal_alpha / 2, 1.0 - conformal_alpha / 2
    idx_lo = np.argmin(np.abs(quantile_levels - q_lo))
    idx_hi = np.argmin(np.abs(quantile_levels - q_hi))
    q_lo_grid = preds_q[:, :, idx_lo]
    q_hi_grid = preds_q[:, :, idx_hi]
    cluster_ids = _assign_nearest_center(coords, spatial_centers)  # (S,)
    qhat_per_site = qhat_per_center[cluster_ids]  # (S,)

    test_mask = test_mask.astype(bool)
    n_test_per_site = test_mask.sum(axis=0)  # (S,)
    inside_nominal = (z_full >= q_lo_grid) & (z_full <= q_hi_grid)
    qhat_expanded = np.broadcast_to(qhat_per_site, (T, S))
    inside_cluster = (z_full >= q_lo_grid - qhat_expanded) & (z_full <= q_hi_grid + qhat_expanded)
    cov_nominal = np.where(
        n_test_per_site > 0,
        np.where(test_mask, inside_nominal, 0).sum(axis=0) / (n_test_per_site + 1e-10),
        np.nan,
    )
    cov_cluster = np.where(
        n_test_per_site > 0,
        np.where(test_mask, inside_cluster, 0).sum(axis=0) / (n_test_per_site + 1e-10),
        np.nan,
    )
    valid_sites = n_test_per_site > 0
    coords_valid = coords[valid_sites]
    cov_nominal_valid = cov_nominal[valid_sites]
    cov_cluster_valid = cov_cluster[valid_sites]
    qhat_per_site[valid_sites]
    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    cov_nominal_grid = griddata(
        coords_valid, cov_nominal_valid, (xi_grid, yi_grid), method="nearest"
    )
    cov_cluster_grid = griddata(
        coords_valid, cov_cluster_valid, (xi_grid, yi_grid), method="nearest"
    )
    qhat_grid = griddata(coords, qhat_per_site, (xi_grid, yi_grid), method="nearest")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, data, title, vmin, vmax in [
        (axes[0], cov_nominal_grid, "Spatial Coverage (Nominal 90% PI)", 0.5, 1.0),
        (axes[1], cov_cluster_grid, "Spatial Coverage (Cluster-aware)", 0.5, 1.0),
        (axes[2], qhat_grid, "Conformal qhat (per cluster)", 0, None),
    ]:
        vmax = vmax if vmax is not None else np.nanmax(data) * 1.05
        im = ax.pcolormesh(
            xi_grid,
            yi_grid,
            data,
            cmap="RdYlGn" if "Coverage" in title else "viridis",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.scatter(
            spatial_centers[:, 0], spatial_centers[:, 1], c="red", s=15, marker="x", alpha=0.7
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_path = output_dir / "spatial_coverage_and_qhat.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Spatial coverage/qhat plot saved to {save_path}")


def create_averaged_spatial_coverage(
    all_results, summary_dir, quantile_levels=None, conformal_alpha=0.1
):
    """
    Create averaged spatial coverage map (nominal + cluster-aware) from all experiments.
    Uses first experiment's spatial_centers as reference for both coverage and qhat.
    Qhat panel: averaged across experiments when centers align; else representative (exp 1).
    quantile_levels/conformal_alpha from config (fallback); prefer first exp's conformal_info when loading.
    Skips if no experiment has predictions_quantile and conformal_info.
    """
    summary_dir = Path(summary_dir)
    all_cov_nominal = []
    all_cov_cluster = []
    all_qhat_per_center = []
    coords_ref = None
    spatial_centers_ref = None
    first_exp_dir = None
    quantile_levels_set = False
    q_lo, q_hi, idx_lo, idx_hi = 0.05, 0.95, 0, -1  # defaults

    for result in all_results:
        exp_dir = Path(result.get("config", {}).get("output_dir", result.get("output_dir", "")))
        if not exp_dir:
            continue
        pred_path = exp_dir / "predictions.npz"
        conf_path = exp_dir / "conformal_info.npz"
        if not pred_path.exists() or not conf_path.exists():
            continue
        preds_npz = np.load(pred_path, allow_pickle=True)
        conf_npz = np.load(conf_path, allow_pickle=True)
        if "predictions_quantile" not in preds_npz:
            continue
        if not quantile_levels_set:
            ql = (
                np.asarray(conf_npz["quantile_levels"])
                if "quantile_levels" in conf_npz.files
                else quantile_levels
            )
            ca = (
                float(conf_npz["conformal_alpha"])
                if "conformal_alpha" in conf_npz.files
                else conformal_alpha
            )
            ql = np.asarray(ql) if ql is not None else np.array([0.05, 0.25, 0.5, 0.75, 0.95])
            q_lo, q_hi = ca / 2, 1.0 - ca / 2
            idx_lo = np.argmin(np.abs(ql - q_lo))
            idx_hi = np.argmin(np.abs(ql - q_hi))
            quantile_levels_set = True
        if first_exp_dir is None:
            first_exp_dir = exp_dir
        preds_q = preds_npz["predictions_quantile"]
        z_full = preds_npz["true"]
        coords = preds_npz["coords"]
        test_mask = preds_npz["test_mask"].astype(bool)
        qhat_per_center = conf_npz["qhat_per_center"]
        spatial_centers = conf_npz["spatial_centers"]
        T, S, Q = preds_q.shape
        q_lo_grid = preds_q[:, :, idx_lo]
        q_hi_grid = preds_q[:, :, idx_hi]
        cluster_ids = _assign_nearest_center(coords, spatial_centers)
        qhat_per_site = qhat_per_center[cluster_ids]
        n_test_per_site = test_mask.sum(axis=0)
        inside_nominal = (z_full >= q_lo_grid) & (z_full <= q_hi_grid)
        qhat_expanded = np.broadcast_to(qhat_per_site, (T, S))
        inside_cluster = (z_full >= q_lo_grid - qhat_expanded) & (
            z_full <= q_hi_grid + qhat_expanded
        )
        cov_nominal = np.where(
            n_test_per_site > 0,
            np.where(test_mask, inside_nominal, 0).sum(axis=0) / (n_test_per_site + 1e-10),
            np.nan,
        )
        cov_cluster = np.where(
            n_test_per_site > 0,
            np.where(test_mask, inside_cluster, 0).sum(axis=0) / (n_test_per_site + 1e-10),
            np.nan,
        )
        all_cov_nominal.append(cov_nominal)
        all_cov_cluster.append(cov_cluster)
        if coords_ref is None:
            coords_ref = coords
            spatial_centers_ref = spatial_centers
        if (
            spatial_centers_ref is not None
            and len(spatial_centers) == len(spatial_centers_ref)
            and np.allclose(spatial_centers, spatial_centers_ref, atol=1e-6)
        ):
            all_qhat_per_center.append(qhat_per_center)

    if len(all_cov_nominal) == 0:
        print("No experiments with conformal info found. Skipping averaged spatial coverage.")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cov_nominal_avg = np.nanmean(np.array(all_cov_nominal), axis=0)
        cov_cluster_avg = np.nanmean(np.array(all_cov_cluster), axis=0)
    valid_sites = ~np.isnan(cov_nominal_avg)
    S = len(coords_ref)
    cluster_ids = _assign_nearest_center(coords_ref, spatial_centers_ref)
    if len(all_qhat_per_center) > 0:
        qhat_avg = np.mean(np.array(all_qhat_per_center), axis=0)
        qhat_title = "Conformal qhat (per cluster, mean over exps)"
    else:
        qhat_avg = np.load(first_exp_dir / "conformal_info.npz")["qhat_per_center"]
        qhat_title = "Conformal qhat (per cluster, representative exp 1)"
    qhat_per_site = qhat_avg[cluster_ids]
    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    cov_nominal_grid = griddata(
        coords_ref[valid_sites], cov_nominal_avg[valid_sites], (xi_grid, yi_grid), method="nearest"
    )
    cov_cluster_grid = griddata(
        coords_ref[valid_sites], cov_cluster_avg[valid_sites], (xi_grid, yi_grid), method="nearest"
    )
    qhat_grid = griddata(coords_ref, qhat_per_site, (xi_grid, yi_grid), method="nearest")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, data, title in [
        (axes[0], cov_nominal_grid, "Averaged Spatial Coverage (Nominal)"),
        (axes[1], cov_cluster_grid, "Averaged Spatial Coverage (Cluster-aware)"),
        (axes[2], qhat_grid, qhat_title),
    ]:
        vmin = 0.5 if "Coverage" in title else 0
        vmax = 1.0 if "Coverage" in title else np.nanmax(data) * 1.05
        im = ax.pcolormesh(
            xi_grid,
            yi_grid,
            data,
            cmap="RdYlGn" if "Coverage" in title else "viridis",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.scatter(
            spatial_centers_ref[:, 0],
            spatial_centers_ref[:, 1],
            c="red",
            s=15,
            marker="x",
            alpha=0.7,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_path = summary_dir / "spatial_coverage_aggregated.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Averaged spatial coverage map saved to {save_path}")

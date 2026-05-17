"""
Temporal series visualizations: per-site time series, quantile bands, conformal highlight.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from da_stdk.utils.conformal import _assign_nearest_center


def plot_temporal_series(
    model,
    z_full,
    coords,
    train_mask,
    device,
    output_dir,
    valid_mask=None,
    test_mask=None,
    n_sites=4,
    quantile_models=None,
    quantile_levels=None,
    conformal_qhat=None,
    conformal_alpha=0.1,
    conformal_qhat_per_cluster=None,
    conformal_centers=None,
    site_selection_seed=None,
):
    """
    Plot temporal series for selected spatial locations.

    Args:
        model: trained model (for mean regression or single quantile), can be None if only quantile_models provided
        z_full: (T, S) full data
        coords: (S, 2) coordinates
        train_mask: (T, S) training mask
        device: computation device
        output_dir: output directory
        valid_mask: (T, S) validation mask (optional)
        test_mask: (T, S) test mask (optional)
        n_sites: number of sites to plot
        quantile_models: dict of {quantile_level: model} for quantile regression (optional)
        quantile_levels: list of quantile levels (optional)
        conformal_qhat: global CQR expansion (optional); if set, plot global 90% PI band [q_lo-qhat, q_hi+qhat]
        conformal_alpha: nominal miscoverage for conformal (0.1 -> 90% PI)
        conformal_qhat_per_cluster: dict or array of per-cluster qhat (optional)
        conformal_centers: (C, 2) spatial centers for cluster assignment (optional)
        site_selection_seed: seed for selecting highlight sites (optional)
    """
    output_dir = Path(output_dir)
    T, S = z_full.shape

    if valid_mask is None:
        valid_mask = np.zeros_like(train_mask, dtype=bool)
    if test_mask is None:
        test_mask = ~(train_mask | valid_mask)

    # Select sites with good spatial coverage
    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords

    sites_with_train = np.where(train_mask.sum(axis=0) > 0)[0]
    selected_sites = []

    if len(sites_with_train) > 0:
        center = np.array([0.5, 0.5])
        dists_to_center = np.linalg.norm(coords_np[sites_with_train] - center, axis=1)
        train_site = sites_with_train[np.argmin(dists_to_center)]
        selected_sites.append(train_site)

    n_grid = int(np.ceil(np.sqrt(n_sites)))
    for i in range(n_grid):
        for j in range(n_grid):
            if len(selected_sites) >= n_sites:
                break
            x_min, x_max = i / n_grid, (i + 1) / n_grid
            y_min, y_max = j / n_grid, (j + 1) / n_grid
            in_region = (
                (coords_np[:, 0] >= x_min)
                & (coords_np[:, 0] < x_max)
                & (coords_np[:, 1] >= y_min)
                & (coords_np[:, 1] < y_max)
            )
            if in_region.sum() > 0:
                region_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
                dists = np.linalg.norm(coords_np[in_region] - region_center, axis=1)
                local_idx = np.argmin(dists)
                global_idx = np.where(in_region)[0][local_idx]
                if global_idx not in selected_sites:
                    selected_sites.append(global_idx)

    all_predictions = None
    quantile_predictions = {}

    if model is not None:
        model.eval()
        with torch.no_grad():
            t_grid = torch.linspace(0, 1, T, device=device)
            coords_tensor = torch.tensor(coords_np, dtype=torch.float32, device=device)
            t_expanded = t_grid.repeat_interleave(S).unsqueeze(1)
            coords_expanded = coords_tensor.repeat(T, 1)
            if hasattr(model, "p") and model.p > 0:
                X_expanded = torch.zeros(T * S, model.p, device=device)
            else:
                X_expanded = torch.zeros(T * S, 0, device=device)
            all_predictions_raw = model(X_expanded, coords_expanded, t_expanded)
            if all_predictions_raw.shape[1] > 1 and quantile_levels is not None:
                for q_idx, q_level in enumerate(quantile_levels):
                    q_pred_flat = all_predictions_raw[:, q_idx]
                    quantile_predictions[q_level] = q_pred_flat.reshape(T, S).cpu().numpy()
                median_idx = all_predictions_raw.shape[1] // 2
                all_predictions_flat = all_predictions_raw[:, median_idx]
            else:
                all_predictions_flat = all_predictions_raw.squeeze()
            all_predictions = all_predictions_flat.reshape(T, S).cpu().numpy()

    if quantile_models is not None and quantile_levels is not None:
        t_grid = torch.linspace(0, 1, T, device=device)
        coords_tensor = torch.tensor(coords_np, dtype=torch.float32, device=device)
        t_expanded = t_grid.repeat_interleave(S).unsqueeze(1)
        coords_expanded = coords_tensor.repeat(T, 1)
        for q_level, q_model in quantile_models.items():
            q_model.eval()
            with torch.no_grad():
                if hasattr(q_model, "p_covariates") and q_model.p_covariates > 0:
                    X_q = torch.zeros(T * S, q_model.p_covariates, device=device)
                else:
                    X_q = torch.zeros(T * S, 0, device=device)
                q_predictions_flat = q_model(X_q, coords_expanded, t_expanded).squeeze()
                quantile_predictions[q_level] = q_predictions_flat.reshape(T, S).cpu().numpy()

    if model is not None and all_predictions is not None:
        n_rows = len(selected_sites)
        n_cols = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
        if len(selected_sites) == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        time_points = np.arange(1, T + 1)
        for idx, site_idx in enumerate(selected_sites):
            ax = axes[idx]
            true_values = z_full[:, site_idx]
            pred_values = all_predictions[:, site_idx]
            train_obs = train_mask[:, site_idx]
            valid_obs = valid_mask[:, site_idx]
            test_obs = test_mask[:, site_idx]
            ax.plot(time_points, pred_values, "b-", linewidth=2, label="Prediction", alpha=0.8)
            if test_obs.sum() > 0:
                ax.scatter(
                    time_points[test_obs],
                    true_values[test_obs],
                    c="gray",
                    s=40,
                    marker="o",
                    alpha=0.7,
                    label="Test (unobserved)",
                    zorder=3,
                )
            observed_mask = train_obs | valid_obs
            if observed_mask.sum() > 0:
                ax.scatter(
                    time_points[observed_mask],
                    true_values[observed_mask],
                    c="black",
                    s=40,
                    marker="o",
                    alpha=0.7,
                    label="Train (observed)",
                    zorder=3,
                )
            site_coord = coords_np[site_idx]
            ax.set_title(
                f"Site {site_idx} at ({site_coord[0]:.3f}, {site_coord[1]:.3f})",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel("Time", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
            ax.grid(True, alpha=0.3)
        for idx in range(len(selected_sites), len(axes)):
            axes[idx].axis("off")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        save_path = output_dir / "temporal_series.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Temporal series plot saved to {save_path}")

    if quantile_predictions and quantile_levels:
        n_rows = len(selected_sites)
        n_cols = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
        if len(selected_sites) == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        if len(quantile_levels) == 3:
            colors = ["#0000FF", "#00CC00", "#FF0000"]
        elif len(quantile_levels) == 5:
            colors = ["#0000FF", "#00CCCC", "#00CC00", "#FF8800", "#FF0000"]
        elif len(quantile_levels) == 7:
            colors = ["#8B00FF", "#0000FF", "#00CCCC", "#00CC00", "#FFCC00", "#FF8800", "#FF0000"]
        else:
            colors = plt.cm.tab10(np.linspace(0, 0.9, len(quantile_levels)))
        time_points = np.arange(1, T + 1)
        for idx, site_idx in enumerate(selected_sites):
            ax = axes[idx]
            true_values = z_full[:, site_idx]
            train_obs = train_mask[:, site_idx]
            valid_obs = valid_mask[:, site_idx]
            test_obs = test_mask[:, site_idx]
            for q_idx, q_level in enumerate(quantile_levels):
                pred_values = quantile_predictions[q_level][:, site_idx]
                ax.plot(
                    time_points,
                    pred_values,
                    color=colors[q_idx],
                    linewidth=2,
                    label=f"τ={q_level}",
                    alpha=0.8,
                )
            has_global = conformal_qhat is not None
            has_cluster = conformal_qhat_per_cluster is not None and conformal_centers is not None
            if has_global or has_cluster:
                q_lo = conformal_alpha / 2
                q_hi = 1.0 - conformal_alpha / 2
                q_levels_arr = np.asarray(quantile_levels)
                idx_lo = np.argmin(np.abs(q_levels_arr - q_lo))
                idx_hi = np.argmin(np.abs(q_levels_arr - q_hi))
                q_lo_level = quantile_levels[idx_lo]
                q_hi_level = quantile_levels[idx_hi]
                q_lo_line = quantile_predictions[q_lo_level][:, site_idx]
                q_hi_line = quantile_predictions[q_hi_level][:, site_idx]
                if has_global:
                    conf_low = q_lo_line - conformal_qhat
                    conf_high = q_hi_line + conformal_qhat
                    ax.fill_between(
                        time_points,
                        conf_low,
                        q_lo_line,
                        alpha=0.35,
                        color="purple",
                        label="90% PI (global conformal)" if idx == 0 else None,
                        zorder=1,
                    )
                    ax.fill_between(
                        time_points, q_hi_line, conf_high, alpha=0.35, color="purple", zorder=1
                    )
                    ax.plot(
                        time_points,
                        conf_low,
                        "--",
                        color="purple",
                        linewidth=2,
                        alpha=0.9,
                        zorder=2,
                    )
                    ax.plot(
                        time_points,
                        conf_high,
                        "--",
                        color="purple",
                        linewidth=2,
                        alpha=0.9,
                        zorder=2,
                    )
                if has_cluster:
                    centers = np.asarray(conformal_centers)
                    cluster_id = int(_assign_nearest_center(coords_np[[site_idx]], centers)[0])
                    if isinstance(conformal_qhat_per_cluster, dict):
                        qhat_site = conformal_qhat_per_cluster.get(
                            cluster_id, conformal_qhat or 0.0
                        )
                    else:
                        qhat_arr = np.asarray(conformal_qhat_per_cluster)
                        qhat_site = (
                            float(qhat_arr[cluster_id])
                            if cluster_id < len(qhat_arr)
                            else float(conformal_qhat or 0.0)
                        )
                    conf_low_c = q_lo_line - qhat_site
                    conf_high_c = q_hi_line + qhat_site
                    ax.fill_between(
                        time_points,
                        conf_low_c,
                        q_lo_line,
                        alpha=0.30,
                        color="#1b9e77",
                        label="90% PI (cluster-aware conformal)" if idx == 0 else None,
                        zorder=1,
                    )
                    ax.fill_between(
                        time_points, q_hi_line, conf_high_c, alpha=0.30, color="#1b9e77", zorder=1
                    )
                    ax.plot(
                        time_points,
                        conf_low_c,
                        "--",
                        color="#1b9e77",
                        linewidth=2,
                        alpha=0.9,
                        zorder=2,
                    )
                    ax.plot(
                        time_points,
                        conf_high_c,
                        "--",
                        color="#1b9e77",
                        linewidth=2,
                        alpha=0.9,
                        zorder=2,
                    )
            if test_obs.sum() > 0:
                ax.scatter(
                    time_points[test_obs],
                    true_values[test_obs],
                    c="gray",
                    s=40,
                    marker="o",
                    alpha=0.7,
                    label="Test",
                    zorder=3,
                )
            observed_mask = train_obs | valid_obs
            if observed_mask.sum() > 0:
                ax.scatter(
                    time_points[observed_mask],
                    true_values[observed_mask],
                    c="black",
                    s=40,
                    marker="o",
                    alpha=0.7,
                    label="Train",
                    zorder=3,
                )
            site_coord = coords_np[site_idx]
            ax.set_title(
                f"Site {site_idx} at ({site_coord[0]:.3f}, {site_coord[1]:.3f}) - All Quantiles",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel("Time", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
            ax.grid(True, alpha=0.3)
        for idx in range(len(selected_sites), len(axes)):
            axes[idx].axis("off")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        save_path = output_dir / "temporal_series_quantiles_combined.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Combined quantile temporal series plot saved to {save_path}")

        if conformal_qhat_per_cluster is not None and conformal_centers is not None:
            plot_temporal_series_conformal_highlight(
                quantile_predictions,
                quantile_levels,
                z_full,
                coords_np,
                train_mask,
                valid_mask,
                test_mask,
                output_dir,
                conformal_qhat=conformal_qhat,
                conformal_alpha=conformal_alpha,
                conformal_qhat_per_cluster=conformal_qhat_per_cluster,
                conformal_centers=conformal_centers,
                site_selection_seed=site_selection_seed,
            )


def plot_temporal_series_conformal_highlight(
    quantile_predictions,
    quantile_levels,
    z_full,
    coords,
    train_mask,
    valid_mask,
    test_mask,
    output_dir,
    conformal_qhat=None,
    conformal_alpha=0.1,
    conformal_qhat_per_cluster=None,
    conformal_centers=None,
    site_selection_seed=None,
    n_sites=3,
    n_top=2,
):
    """
    Highlight sites with large conformal qhat: compare q05/q95 vs conformal intervals.
    Produces temporal_series_conformal_highlight.png.
    """
    if not quantile_predictions or not quantile_levels:
        return
    if conformal_qhat_per_cluster is None or conformal_centers is None:
        return

    output_dir = Path(output_dir)
    coords_np = coords if isinstance(coords, np.ndarray) else np.asarray(coords)
    T, S = z_full.shape
    q_levels_arr = np.asarray(quantile_levels)
    q_lo = conformal_alpha / 2
    q_hi = 1.0 - conformal_alpha / 2
    idx_lo = np.argmin(np.abs(q_levels_arr - q_lo))
    idx_hi = np.argmin(np.abs(q_levels_arr - q_hi))
    q_lo_level = quantile_levels[idx_lo]
    q_hi_level = quantile_levels[idx_hi]

    centers = np.asarray(conformal_centers)
    cluster_ids = _assign_nearest_center(coords_np, centers)
    if isinstance(conformal_qhat_per_cluster, dict):
        qhat_per_site = np.array(
            [conformal_qhat_per_cluster.get(int(c), conformal_qhat or 0.0) for c in cluster_ids]
        )
    else:
        qhat_arr = np.asarray(conformal_qhat_per_cluster)
        qhat_per_site = np.array(
            [
                qhat_arr[int(c)] if int(c) < len(qhat_arr) else (conformal_qhat or 0.0)
                for c in cluster_ids
            ]
        )

    test_counts = test_mask.sum(axis=0)
    candidates = np.where(test_counts > 0)[0]
    if len(candidates) == 0:
        candidates = np.arange(S)

    sorted_idx = candidates[np.argsort(qhat_per_site[candidates])[::-1]]
    top_sites = [int(s) for s in sorted_idx[:n_top]]
    rng = np.random.RandomState(site_selection_seed or 0)
    remaining = [int(s) for s in candidates if int(s) not in top_sites]
    if remaining:
        random_site = int(rng.choice(remaining))
        selected_sites = top_sites + [random_site]
    else:
        selected_sites = top_sites
    selected_sites = selected_sites[:n_sites]

    time_points = np.arange(1, T + 1)
    fig, axes = plt.subplots(len(selected_sites), 1, figsize=(14, 3.5 * len(selected_sites)))
    if len(selected_sites) == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, site_idx in enumerate(selected_sites):
        ax = axes[idx]
        true_values = z_full[:, site_idx]
        train_obs = train_mask[:, site_idx]
        valid_obs = valid_mask[:, site_idx]
        test_obs = test_mask[:, site_idx]

        q_lo_line = quantile_predictions[q_lo_level][:, site_idx]
        q_hi_line = quantile_predictions[q_hi_level][:, site_idx]
        ax.fill_between(
            time_points,
            q_lo_line,
            q_hi_line,
            color="gray",
            alpha=0.2,
            label="90% PI (quantile)" if idx == 0 else None,
            zorder=1,
        )
        ax.plot(
            time_points,
            q_lo_line,
            color="blue",
            linewidth=1.5,
            alpha=0.8,
            label="q=0.05" if idx == 0 else None,
        )
        ax.plot(
            time_points,
            q_hi_line,
            color="red",
            linewidth=1.5,
            alpha=0.8,
            label="q=0.95" if idx == 0 else None,
        )

        if conformal_qhat is not None:
            conf_low_g = q_lo_line - conformal_qhat
            conf_high_g = q_hi_line + conformal_qhat
            ax.fill_between(
                time_points,
                conf_low_g,
                conf_high_g,
                color="purple",
                alpha=0.12,
                label="90% PI (global conformal)" if idx == 0 else None,
                zorder=0,
            )
            ax.plot(time_points, conf_low_g, "--", color="purple", linewidth=1.6, alpha=0.8)
            ax.plot(time_points, conf_high_g, "--", color="purple", linewidth=1.6, alpha=0.8)

        qhat_site = float(qhat_per_site[site_idx])
        conf_low_c = q_lo_line - qhat_site
        conf_high_c = q_hi_line + qhat_site
        ax.fill_between(
            time_points,
            conf_low_c,
            conf_high_c,
            color="#1b9e77",
            alpha=0.18,
            label="90% PI (cluster-aware conformal)" if idx == 0 else None,
            zorder=0,
        )
        ax.plot(time_points, conf_low_c, "--", color="#1b9e77", linewidth=1.8, alpha=0.9)
        ax.plot(time_points, conf_high_c, "--", color="#1b9e77", linewidth=1.8, alpha=0.9)

        if test_obs.sum() > 0:
            ax.scatter(
                time_points[test_obs],
                true_values[test_obs],
                c="gray",
                s=30,
                alpha=0.7,
                label="Test" if idx == 0 else None,
                zorder=3,
            )
        observed_mask = train_obs | valid_obs
        if observed_mask.sum() > 0:
            ax.scatter(
                time_points[observed_mask],
                true_values[observed_mask],
                c="black",
                s=30,
                alpha=0.7,
                label="Train" if idx == 0 else None,
                zorder=3,
            )

        site_coord = coords_np[site_idx]
        ax.set_title(
            f"Site {site_idx} at ({site_coord[0]:.3f}, {site_coord[1]:.3f}) "
            f"- qhat={qhat_site:.3f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_path = output_dir / "temporal_series_conformal_highlight.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Conformal highlight temporal series plot saved to {save_path}")

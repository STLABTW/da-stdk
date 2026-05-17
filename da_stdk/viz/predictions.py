"""True / predicted / bias heatmaps at selected times."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata


def plot_predictions(model, z_full, coords, train_mask, device, output_dir, n_times=3):
    """
    Plot true/pred/bias heatmaps for random time points.

    Args:
        model: trained model
        z_full: (T, S) full data
        coords: (S, 2) coordinates
        train_mask: (T, S) training mask
        device: torch device
        output_dir: output directory (Path or str)
        n_times: number of time points to plot
    """
    output_dir = Path(output_dir)
    T, S = z_full.shape

    # Select random time points
    np.random.seed(42)
    time_indices = np.random.choice(T, size=min(n_times, T), replace=False)
    time_indices = sorted(time_indices)

    # Get spatial basis centers from model
    spatial_centers = model.spatial_basis.centers.detach().cpu().numpy()  # (k_spatial, 2)
    spatial_bandwidths = model.spatial_basis.bandwidths.detach().cpu().numpy()  # (k_spatial,)

    # Size basis markers proportional to bandwidth
    bw_normalized = (spatial_bandwidths - spatial_bandwidths.min()) / (
        spatial_bandwidths.max() - spatial_bandwidths.min() + 1e-8
    )
    basis_sizes = 10 + bw_normalized * 90  # Range [10, 100]

    # Generate predictions for all sites at selected times
    model.eval()
    predictions = {}

    with torch.no_grad():
        for t_idx in time_indices:
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

            predictions[t_idx] = y_pred

    # Create grid for interpolation (higher resolution for smoother heatmap)
    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Create plots
    fig, axes = plt.subplots(n_times, 3, figsize=(20, 5 * n_times))
    if n_times == 1:
        axes = axes.reshape(1, -1)

    for i, t_idx in enumerate(time_indices):
        y_true = z_full[t_idx, :]
        y_pred = predictions[t_idx]
        bias = y_pred - y_true

        # Get train sites at this time
        train_sites_t = np.where(train_mask[t_idx, :])[0]
        train_coords_t = coords[train_sites_t]

        # Valid indices (non-NaN)
        valid_idx = ~np.isnan(y_true)
        coords_valid = coords[valid_idx]

        # Interpolate to grid using nearest neighbor (to fill space like Voronoi)
        y_true_grid = griddata(
            coords_valid, y_true[valid_idx], (xi_grid, yi_grid), method="nearest"
        )
        y_pred_grid = griddata(
            coords_valid, y_pred[valid_idx], (xi_grid, yi_grid), method="nearest"
        )
        bias_grid = griddata(coords_valid, bias[valid_idx], (xi_grid, yi_grid), method="nearest")

        # True values
        ax = axes[i, 0]
        im = ax.pcolormesh(xi_grid, yi_grid, y_true_grid, cmap="viridis", shading="auto")
        ax.scatter(
            train_coords_t[:, 0],
            train_coords_t[:, 1],
            c="black",
            s=20,
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
        ax.set_title(f"t={t_idx+1} - True", fontsize=16, fontweight="bold")
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=11)
        plt.colorbar(im, ax=ax)

        # Predicted values
        ax = axes[i, 1]
        im = ax.pcolormesh(xi_grid, yi_grid, y_pred_grid, cmap="viridis", shading="auto")
        ax.scatter(
            train_coords_t[:, 0],
            train_coords_t[:, 1],
            c="black",
            s=20,
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
        ax.set_title(f"t={t_idx+1} - Predicted", fontsize=16, fontweight="bold")
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=11)
        plt.colorbar(im, ax=ax)

        # Bias (pred - true)
        ax = axes[i, 2]
        bias_max = np.nanmax(np.abs(bias[valid_idx]))
        im = ax.pcolormesh(
            xi_grid,
            yi_grid,
            bias_grid,
            cmap="RdBu_r",
            shading="auto",
            vmin=-bias_max,
            vmax=bias_max,
        )
        ax.scatter(
            train_coords_t[:, 0],
            train_coords_t[:, 1],
            c="black",
            s=20,
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
        ax.set_title(f"t={t_idx+1} - Bias (Pred - True)", fontsize=16, fontweight="bold")
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=11)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    save_path = output_dir / "prediction_maps.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction maps saved to {save_path}")


__all__ = ["plot_predictions"]

"""
Observation pattern visualization (train/valid/test per site) and observation density maps.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def plot_observation_pattern(coords, obs_mask, train_mask, valid_mask, output_dir):
    """
    Plot the spatial pattern of observations (train/valid/test per site).

    Args:
        coords: (S, 2) coordinates
        obs_mask: (T, S) observation mask
        train_mask: (T, S) training mask
        valid_mask: (T, S) validation mask
        output_dir: output directory (Path or str)
    """
    T, S = obs_mask.shape
    point_size = max(5, min(100, 13 * np.sqrt(1000 / S)))

    obs_counts = obs_mask.sum(axis=0)
    train_counts = train_mask.sum(axis=0)
    valid_counts = valid_mask.sum(axis=0)
    test_counts = T - obs_counts

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=obs_counts, cmap="viridis", s=point_size, alpha=0.7
    )
    ax.set_title(
        f"Total Observations per Site\n(Total: {obs_mask.sum()} obs)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.colorbar(scatter, ax=ax, label="# observations").ax.tick_params(labelsize=11)

    ax = axes[0, 1]
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=train_counts, cmap="Blues", s=point_size, alpha=0.7
    )
    ax.set_title(
        f"Train Observations per Site\n(Total: {train_mask.sum()} obs)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.colorbar(scatter, ax=ax, label="# observations").ax.tick_params(labelsize=11)

    ax = axes[1, 0]
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=valid_counts, cmap="Greens", s=point_size, alpha=0.7
    )
    ax.set_title(
        f"Valid Observations per Site\n(Total: {valid_mask.sum()} obs)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.colorbar(scatter, ax=ax, label="# observations").ax.tick_params(labelsize=11)

    ax = axes[1, 1]
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=test_counts, cmap="Reds", s=point_size, alpha=0.7
    )
    ax.set_title(
        f"Test (Unobserved) per Site\n(Total: {(~obs_mask).sum()} obs)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.colorbar(scatter, ax=ax, label="# unobserved").ax.tick_params(labelsize=11)

    plt.tight_layout()
    save_path = Path(output_dir) / "observation_pattern.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Observation pattern plot saved to {save_path}")


def create_observation_density_map(all_train_masks, coords, summary_dir):
    """
    Create observation density heatmap to compare with spatial MSE.

    Args:
        all_train_masks: list of (T, S) training masks from all experiments
        coords: (S, 2) coordinates
        summary_dir: directory to save the map (Path or str)
    """
    summary_dir = Path(summary_dir)
    n_experiments = len(all_train_masks)

    if n_experiments == 0:
        print("No train masks found. Skipping observation density map.")
        return

    all_masks_array = np.array(all_train_masks)
    T, S = all_masks_array.shape[1], all_masks_array.shape[2]
    total_possible_obs = n_experiments * T
    total_obs_per_site = all_masks_array.sum(axis=(0, 1))
    obs_ratio_per_site = total_obs_per_site / total_possible_obs

    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    obs_ratio_grid = griddata(coords, obs_ratio_per_site, (xi_grid, yi_grid), method="nearest")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(
        xi_grid, yi_grid, obs_ratio_grid, cmap="Blues", shading="auto", vmin=0, vmax=1
    )
    ax.set_title(f"Observation Ratio per Site (n={n_experiments} experiments)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.colorbar(im, ax=ax, label="Observation Ratio")
    plt.tight_layout()
    save_path = summary_dir / "observation_density.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Observation density map saved to {save_path}")


__all__ = ["plot_observation_pattern", "create_observation_density_map"]

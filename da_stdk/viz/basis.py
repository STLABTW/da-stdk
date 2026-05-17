"""Spatial basis center trajectories during training."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch, Rectangle


def plot_basis_evolution(
    model_initial, model_final, train_coords, output_dir, config, basis_centers_history=None
):
    """
    Plot spatial basis centers before and after training.

    Args:
        model_initial: model with initial basis centers
        model_final: trained model with final basis centers
        train_coords: (N, 2) training coordinates
        output_dir: output directory (Path or str)
        config: configuration dict
        basis_centers_history: list of (epoch, centers) tuples recording trajectory every 100 epochs
    """
    output_dir = Path(output_dir)

    # Extract basis information
    centers_init = model_initial.spatial_basis.centers.detach().cpu().numpy()
    centers_final = model_final.spatial_basis.centers.detach().cpu().numpy()

    bandwidths_init = model_initial.spatial_basis.bandwidths.detach().cpu().numpy()
    bandwidths_final = model_final.spatial_basis.bandwidths.detach().cpu().numpy()

    learnable = config.get("spatial_learnable", False)
    init_method = config.get("spatial_init_method", "uniform")
    k_spatial_centers = config.get("k_spatial_centers", [25, 81, 121])

    # Detect inactive basis (zero weights from Group Lasso)
    inactive_basis_mask = None
    sparsity_type = config.get("sparsity_penalty_type", "none")
    if sparsity_type in ["group", "sparse_group"]:
        if (
            hasattr(model_final, "use_delta_reparameterization")
            and model_final.use_delta_reparameterization
            and model_final.mlp_trunk is not None
        ):
            first_layer_weight = model_final.mlp_trunk[0].weight.data  # (hidden_dim, input_dim)
        else:
            first_layer_weight = model_final.mlp[0].weight.data  # (hidden_dim, input_dim)

        p = config.get("p_covariates", 0)
        k_spatial = model_final.k_spatial
        spatial_weights = first_layer_weight[:, p : p + k_spatial].T  # (k_spatial, hidden_dim)
        basis_norms = torch.norm(spatial_weights, dim=1).cpu().numpy()  # (k_spatial,)

        if basis_norms.max() > 0:
            relative_threshold = config.get("sparsity_threshold_ratio", 1e-2)
            threshold = relative_threshold * basis_norms.max()
        else:
            threshold = 0.0

        inactive_basis_mask = basis_norms < threshold
        n_inactive = inactive_basis_mask.sum()
        n_active = (~inactive_basis_mask).sum()

        print(f"\n  [INFO] Sparsity Analysis ({sparsity_type} penalty):")
        print(f"     Active basis: {n_active}/{len(inactive_basis_mask)}")
        print(
            f"     Removed basis: {n_inactive}/{len(inactive_basis_mask)} (norm < {threshold:.4f})"
        )
        print(
            f"     Basis norms: min={basis_norms.min():.4f}, max={basis_norms.max():.4f}, "
            f"median={np.median(basis_norms):.4f}, mean={basis_norms.mean():.4f}"
        )
        sorted_norms = np.sort(basis_norms)
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = np.percentile(sorted_norms, percentiles)
        print(f"     Percentiles: ", end="")
        for pv, v in zip(percentiles, percentile_values):
            print(f"P{pv}={v:.4f} ", end="")
        print()
        very_small_1e3 = (basis_norms < 1e-3 * basis_norms.max()).sum()
        very_small_1e2 = (basis_norms < 1e-2 * basis_norms.max()).sum()
        very_small_1e1 = (basis_norms < 1e-1 * basis_norms.max()).sum()
        print(
            f"     Small norms: <0.1%max: {very_small_1e3}, <1%max: {very_small_1e2}, "
            f"<10%max: {very_small_1e1}"
        )

    # Sample training data for visualization (max 20000 points)
    max_train_vis = 20000
    if len(train_coords) > max_train_vis:
        indices = np.random.choice(len(train_coords), max_train_vis, replace=False)
        train_coords_vis = train_coords[indices]
    else:
        train_coords_vis = train_coords

    if learnable:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes = [axes[0], axes[1], None]

    def bw_to_size(bw):
        bw_norm = (bw - bw.min()) / (bw.max() - bw.min() + 1e-8)
        return 20 + bw_norm * 180

    sizes_init = bw_to_size(bandwidths_init)
    sizes_final = bw_to_size(bandwidths_final)

    colors = []
    color_map = ["red", "blue", "green"]
    for i, k in enumerate(k_spatial_centers):
        colors.extend([color_map[i % 3]] * k)

    legend_elements = [
        Patch(facecolor=color_map[i], label=f"Resolution {i+1} (k={k})")
        for i, k in enumerate(k_spatial_centers)
    ]
    legend_elements.insert(0, Patch(facecolor="lightgray", label="Train data"))

    # Plot 1: Initial basis
    ax = axes[0]
    ax.scatter(
        train_coords_vis[:, 0],
        train_coords_vis[:, 1],
        c="lightgray",
        s=2,
        alpha=0.3,
        label="Train data",
        rasterized=True,
    )
    rect = Rectangle(
        (0, 0),
        1,
        1,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        label="Domain [0,1]²",
    )
    ax.add_patch(rect)
    for i, (center, size, color) in enumerate(zip(centers_init, sizes_init, colors)):
        ax.scatter(
            center[0],
            center[1],
            c=color,
            s=size,
            marker="o",
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )
    ax.set_title(
        f"Initial Basis Centers\n({init_method} initialization)", fontsize=20, fontweight="bold"
    )
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14)
    ax.grid(True, alpha=0.2)

    # Plot 2: Final basis (after training)
    ax = axes[1]
    ax.scatter(
        train_coords_vis[:, 0],
        train_coords_vis[:, 1],
        c="lightgray",
        s=2,
        alpha=0.3,
        label="Train data",
        rasterized=True,
    )
    rect = Rectangle((0, 0), 1, 1, linewidth=2, edgecolor="black", facecolor="none", linestyle="--")
    ax.add_patch(rect)
    out_of_domain = ((centers_final < 0) | (centers_final > 1)).any(axis=1).sum()
    for i, (center, size, color) in enumerate(zip(centers_final, sizes_final, colors)):
        if inactive_basis_mask is not None and inactive_basis_mask[i]:
            alpha = 0.15
        else:
            alpha = 0.6
        ax.scatter(
            center[0],
            center[1],
            c=color,
            s=size,
            marker="o",
            alpha=alpha,
            edgecolors="black",
            linewidths=0.5,
        )
    title_suffix = " (LEARNED)" if learnable else " (FIXED - same as initial)"
    if learnable and out_of_domain > 0:
        title_suffix += f"\n[WARNING] {out_of_domain} centers out-of-domain"
    if inactive_basis_mask is not None:
        n_inactive = inactive_basis_mask.sum()
        n_active = (~inactive_basis_mask).sum()
        title_suffix += f"\n[INFO] {n_active} active, {n_inactive} removed (sparsity)"
    ax.set_title(f"Final Basis Centers{title_suffix}", fontsize=20, fontweight="bold")
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14)
    ax.grid(True, alpha=0.2)

    # Plot 3: Movement (only if learnable)
    if learnable and axes[2] is not None:
        ax = axes[2]
        ax.scatter(
            train_coords_vis[:, 0],
            train_coords_vis[:, 1],
            c="lightgray",
            s=2,
            alpha=0.3,
            label="Train data",
            rasterized=True,
        )
        if basis_centers_history is not None and len(basis_centers_history) > 0:
            trajectory = [(0, centers_init)]
            trajectory.extend(basis_centers_history)
            trajectory.append((-1, centers_final))
            for i in range(len(centers_init)):
                if inactive_basis_mask is not None and inactive_basis_mask[i]:
                    continue
                path = np.array([traj[1][i] for traj in trajectory])
                total_distance = np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))
                if total_distance > 0.005:
                    ax.plot(
                        path[:, 0], path[:, 1], color=colors[i], alpha=0.5, linewidth=1.5, zorder=1
                    )
        else:
            for i, (c_init, c_final, color) in enumerate(zip(centers_init, centers_final, colors)):
                if inactive_basis_mask is not None and inactive_basis_mask[i]:
                    continue
                dx = c_final[0] - c_init[0]
                dy = c_final[1] - c_init[1]
                distance = np.sqrt(dx**2 + dy**2)
                if distance > 0.005:
                    ax.arrow(
                        c_init[0],
                        c_init[1],
                        dx,
                        dy,
                        head_width=0.015,
                        head_length=0.01,
                        fc=color,
                        ec="black",
                        alpha=0.5,
                        linewidth=0.5,
                    )
        if inactive_basis_mask is not None:
            active_mask = ~inactive_basis_mask
            if active_mask.any():
                ax.scatter(
                    centers_init[active_mask, 0],
                    centers_init[active_mask, 1],
                    c="white",
                    s=16,
                    marker="o",
                    alpha=0.8,
                    edgecolors="black",
                    linewidths=1.5,
                    label="Initial (active)",
                    zorder=2,
                )
                ax.scatter(
                    centers_final[active_mask, 0],
                    centers_final[active_mask, 1],
                    c=np.array(colors)[active_mask],
                    s=80,
                    marker="o",
                    alpha=0.8,
                    edgecolors="black",
                    linewidths=1.5,
                    label="Final (active)",
                    zorder=2,
                )
            if inactive_basis_mask.any():
                ax.scatter(
                    centers_init[inactive_basis_mask, 0],
                    centers_init[inactive_basis_mask, 1],
                    c="white",
                    s=8,
                    marker="o",
                    alpha=0.2,
                    edgecolors="gray",
                    linewidths=0.5,
                    label="Removed by sparsity",
                    zorder=2,
                )
                ax.scatter(
                    centers_final[inactive_basis_mask, 0],
                    centers_final[inactive_basis_mask, 1],
                    c=np.array(colors)[inactive_basis_mask],
                    s=40,
                    marker="o",
                    alpha=0.2,
                    edgecolors="gray",
                    linewidths=0.5,
                    zorder=2,
                )
        else:
            ax.scatter(
                centers_init[:, 0],
                centers_init[:, 1],
                c="white",
                s=16,
                marker="o",
                alpha=0.8,
                edgecolors="black",
                linewidths=1.5,
                label="Initial",
                zorder=2,
            )
            ax.scatter(
                centers_final[:, 0],
                centers_final[:, 1],
                c=colors,
                s=80,
                marker="o",
                alpha=0.8,
                edgecolors="black",
                linewidths=1.5,
                label="Final",
                zorder=2,
            )
        if inactive_basis_mask is not None:
            active_mask = ~inactive_basis_mask
            movements = np.linalg.norm(
                centers_final[active_mask] - centers_init[active_mask], axis=1
            )
        else:
            movements = np.linalg.norm(centers_final - centers_init, axis=1)
        mean_movement = movements.mean() if len(movements) > 0 else 0
        max_movement = movements.max() if len(movements) > 0 else 0
        median_movement = np.median(movements) if len(movements) > 0 else 0
        movement_penalty = config.get("movement_penalty_weight", 0.0)
        title_text = f"Basis Movement\nMean: {mean_movement:.4f}, Max: {max_movement:.4f}, "
        title_text += f"Median: {median_movement:.4f}\n"
        if movement_penalty > 0:
            title_text += f"Movement Penalty: λ={movement_penalty}"
        ax.set_title(title_text, fontsize=18, fontweight="bold")
        ax.set_xlabel("x", fontsize=18)
        ax.set_ylabel("y", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=14)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = output_dir / "basis_evolution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Basis evolution plot saved to {save_path}")


__all__ = ["plot_basis_evolution"]

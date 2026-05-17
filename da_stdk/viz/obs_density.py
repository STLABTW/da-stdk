"""Observation density grids and 2×2 scenario comparison plots."""

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from da_stdk.data.obs_sampling import create_spatial_obs_prob_fn, sample_observations


def compute_observation_density(
    z_data: np.ndarray,
    coords: np.ndarray,
    obs_method: str,
    spatial_pattern: str,
    obs_ratio: float = 0.1,
    intensity: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """Per-site observation frequency under one sampling scenario."""
    T, S = z_data.shape
    obs_prob_fn = create_spatial_obs_prob_fn(pattern=spatial_pattern, intensity=intensity)
    obs_mask, _ = sample_observations(
        z_data,
        coords,
        obs_method=obs_method,
        obs_ratio=obs_ratio,
        obs_prob_fn=obs_prob_fn,
        seed=seed,
    )
    if obs_method == "site-wise":
        density = obs_mask.any(axis=0).astype(float)
    else:
        density = obs_mask.sum(axis=0) / T
    return density


def plot_observation_density_maps(
    data_path: Optional[Union[str, Path]] = None,
    z_data: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
    obs_ratio: float = 0.1,
    intensity: float = 10.0,
    n_samples: int = 100,
    seed: int = 42,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot 2x2 observation density maps for 4 scenarios (Fixed/Random × Uniform/Clustered).

    Call with either data_path (CSV with columns t, x, y, z) or (z_data, coords).

    Args:
        data_path: path to CSV data file (e.g. '2b_8_train.csv'); used if z_data/coords not provided
        z_data: (T, S) array; required if data_path is None
        coords: (S, 2) array; required if data_path is None
        obs_ratio: observation ratio
        intensity: intensity for corner pattern
        n_samples: unused, kept for API compatibility
        seed: base random seed
        save_path: path to save figure

    Returns:
        fig: matplotlib figure
    """
    if data_path is not None:
        df = pd.read_csv(data_path)
        coords_df = df[["x", "y"]].drop_duplicates().sort_values(["x", "y"])
        coords = coords_df.values
        S = len(coords)
        T = df["t"].nunique()
        df_sorted = df.sort_values(["t", "x", "y"])
        z_data = df_sorted["z"].values.reshape(T, S)
    elif z_data is None or coords is None:
        raise ValueError("Provide either data_path or both z_data and coords")

    T, S = z_data.shape
    print(f"Data shape: T={T}, S={S}")
    print(f"Observation ratio: {obs_ratio}")
    print("Computing densities for single experiment...")

    scenarios = [
        {"obs_method": "site-wise", "spatial_pattern": "uniform", "title": "Fixed + Uniform"},
        {"obs_method": "site-wise", "spatial_pattern": "corner", "title": "Fixed + Clustered"},
        {"obs_method": "random", "spatial_pattern": "uniform", "title": "Random + Uniform"},
        {"obs_method": "random", "spatial_pattern": "corner", "title": "Random + Clustered"},
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for idx, (ax, scenario) in enumerate(zip(axes, scenarios)):
        print(f"  Scenario {idx+1}: {scenario['title']}")
        density = compute_observation_density(
            z_data,
            coords,
            obs_method=scenario["obs_method"],
            spatial_pattern=scenario["spatial_pattern"],
            obs_ratio=obs_ratio,
            intensity=intensity,
            seed=seed + idx,
        )
        ax.scatter(coords[:, 0], coords[:, 1], c="red", s=5, alpha=density)
        ax.set_title(scenario["title"], fontsize=27, fontweight="bold")
        ax.set_xlabel("x", fontsize=21)
        ax.set_ylabel("y", fontsize=21)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nDensity map saved to {save_path}")
    return fig


__all__ = ["compute_observation_density", "plot_observation_density_maps"]

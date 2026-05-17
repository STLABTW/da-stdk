"""Spatial observation masks (uniform or corner-biased density)."""

import numpy as np


def create_spatial_obs_prob_fn(pattern="uniform", intensity=1.0):
    """Return site weight fn, or None for uniform. ``corner``: p(s) ∝ (1 + intensity‖s‖)⁻²."""
    if pattern == "uniform" or pattern is None:
        return None

    if pattern == "corner":
        # Corner: p(s) ∝ (1 + intensity * ||s||)^{-2}; paper intensity = 10
        def obs_prob_fn(coord):
            x, y = coord
            dist = np.sqrt(x**2 + y**2)
            prob = 1.0 / (1.0 + float(intensity) * dist) ** 2
            return prob

        return obs_prob_fn

    raise ValueError(f"Unknown pattern: {pattern}")


def sample_observations(
    z_data, coords, obs_method="site-wise", obs_ratio=0.5, obs_prob_fn=None, seed=None, config=None
):
    """Sample ``obs_mask`` (T, S) and observed site indices."""
    if seed is not None:
        np.random.seed(seed)

    T, S = z_data.shape

    if obs_prob_fn is not None:
        obs_weights = np.array([obs_prob_fn(coords[i]) for i in range(S)])
        obs_weights_normalized = obs_weights / obs_weights.mean()
        obs_probs = obs_weights_normalized * obs_ratio
        obs_probs = np.clip(obs_probs, 0, 1)
    else:
        obs_probs = np.ones(S) * obs_ratio

    if obs_method == "site-wise":
        n_obs_sites = int(S * obs_ratio)
        obs_weights_normalized = obs_probs / obs_probs.sum()
        obs_sites = np.random.choice(S, size=n_obs_sites, replace=False, p=obs_weights_normalized)
        obs_mask = np.zeros((T, S), dtype=bool)
        obs_mask[:, obs_sites] = True
        return obs_mask, obs_sites

    if obs_method == "random":
        obs_probs_expanded = obs_probs[np.newaxis, :].repeat(T, axis=0)
        obs_mask = np.random.rand(T, S) < obs_probs_expanded
        obs_sites = np.where(obs_mask.any(axis=0))[0]
        return obs_mask, obs_sites

    raise ValueError(f"Unknown obs_method: {obs_method}")


__all__ = ["create_spatial_obs_prob_fn", "sample_observations"]

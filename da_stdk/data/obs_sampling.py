"""
Observation sampling: spatial probability function and observation mask sampling.

Used by train_default and visualize_obs_density. KAUST experiment: obs_ratio 0.1,
obs_method site-wise/random, obs_spatial_pattern uniform/corner (intensity 10).
"""

import numpy as np


def create_spatial_obs_prob_fn(pattern="uniform", intensity=1.0):
    """
    Create spatial observation probability function.

    Args:
        pattern: 'uniform', 'corner', or custom function
        intensity: for 'corner': p(s) ∝ (1 + intensity * ||s||)^{-2}; paper uses 10.0

    Returns:
        obs_prob_fn: function(coord) -> probability, or None for uniform
    """
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
    """
    Sample observations from full spatio-temporal data.

    Args:
        z_data: (T, S) full data
        coords: (S, 2) spatial coordinates in [0,1]^2
        obs_method: 'site-wise' (fixed sites) or 'random' (Bernoulli per (t,s))
        obs_ratio: target observation ratio (~10% in paper)
        obs_prob_fn: function(coord) -> prob; if None, uniform
        seed: random seed
        config: optional config dict (unused; for API compatibility)

    Returns:
        obs_mask: (T, S) boolean observed locations
        obs_sites: 1d array of site indices observed at least once (for site-wise: selected sites)
    """
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

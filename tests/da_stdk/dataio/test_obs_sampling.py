"""
Tests for observation sampling: create_spatial_obs_prob_fn, sample_observations.

Section 8.4: Confirm obs mask logic with fixed seed (reproducibility).
"""

import numpy as np
import pytest

from da_stdk.dataio.obs_sampling import create_spatial_obs_prob_fn, sample_observations


def test_create_spatial_obs_prob_fn_uniform():
    """Uniform pattern returns None (no spatial weighting)."""
    assert create_spatial_obs_prob_fn(pattern="uniform") is None
    assert create_spatial_obs_prob_fn(pattern=None) is None


def test_create_spatial_obs_prob_fn_corner():
    """Corner pattern returns callable; prob at origin > prob at (1,1)."""
    fn = create_spatial_obs_prob_fn(pattern="corner", intensity=10.0)
    assert callable(fn)
    p0 = fn((0.0, 0.0))
    p1 = fn((1.0, 1.0))
    assert p0 > p1
    assert 0 < p0 <= 1 and 0 < p1 <= 1


def test_create_spatial_obs_prob_fn_unknown_pattern():
    """Unknown pattern raises ValueError."""
    with pytest.raises(ValueError, match="Unknown pattern"):
        create_spatial_obs_prob_fn(pattern="invalid")


def test_sample_observations_site_wise_reproducible():
    """Same seed yields same obs_mask and obs_sites (site-wise)."""
    np.random.seed(123)
    z_data = np.random.randn(20, 50).astype(np.float32)
    coords = np.random.rand(50, 2).astype(np.float32)
    mask1, sites1 = sample_observations(
        z_data, coords, obs_method="site-wise", obs_ratio=0.2, seed=42
    )
    mask2, sites2 = sample_observations(
        z_data, coords, obs_method="site-wise", obs_ratio=0.2, seed=42
    )
    np.testing.assert_array_equal(mask1, mask2)
    np.testing.assert_array_equal(sites1, sites2)
    assert mask1.shape == (20, 50)
    # Site-wise: exactly n_obs_sites columns True per row
    n_obs_sites = int(50 * 0.2)
    assert mask1.sum(axis=0).astype(int).max() == 20  # selected sites observed every t
    assert mask1.sum(axis=0).astype(int).min() == 0
    assert len(sites1) == n_obs_sites


def test_sample_observations_random_reproducible():
    """Same seed yields same obs_mask (random)."""
    z_data = np.random.randn(15, 40).astype(np.float32)
    coords = np.random.rand(40, 2).astype(np.float32)
    mask1, _ = sample_observations(z_data, coords, obs_method="random", obs_ratio=0.1, seed=99)
    mask2, _ = sample_observations(z_data, coords, obs_method="random", obs_ratio=0.1, seed=99)
    np.testing.assert_array_equal(mask1, mask2)
    assert mask1.shape == (15, 40)
    # Random: approximate ratio
    ratio = mask1.sum() / (15 * 40)
    assert 0.02 < ratio < 0.25


def test_sample_observations_random_ratio_approximate():
    """Random method: mean observation ratio is close to obs_ratio."""
    z_data = np.random.randn(30, 100).astype(np.float32)
    coords = np.random.rand(100, 2).astype(np.float32)
    ratios = []
    for seed in range(5):
        mask, _ = sample_observations(
            z_data, coords, obs_method="random", obs_ratio=0.15, seed=seed
        )
        ratios.append(mask.sum() / mask.size)
    mean_ratio = np.mean(ratios)
    assert 0.08 < mean_ratio < 0.25


def test_sample_observations_with_corner_prob_fn():
    """sample_observations with corner prob_fn runs and returns valid mask."""
    z_data = np.random.randn(10, 20).astype(np.float32)
    coords = np.random.rand(20, 2).astype(np.float32)
    prob_fn = create_spatial_obs_prob_fn(pattern="corner", intensity=10.0)
    mask, sites = sample_observations(
        z_data,
        coords,
        obs_method="site-wise",
        obs_ratio=0.3,
        obs_prob_fn=prob_fn,
        seed=0,
    )
    assert mask.shape == (10, 20)
    assert mask.dtype == bool
    assert len(sites) <= 20
    # site-wise: selected sites have mask True for all T
    assert (mask.sum(axis=0) == 10).sum() == len(sites)

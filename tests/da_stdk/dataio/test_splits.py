"""Tests for da_stdk/dataio/splits.py."""

import numpy as np
import pytest

from da_stdk.dataio.splits import split_train_valid

RNG = np.random.default_rng(42)


def _make_obs_mask(T=10, S=20, p=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((T, S)) < p


class TestSplitTrainValid:
    def test_site_wise_shapes(self):
        mask = _make_obs_mask()
        obs_sites = np.arange(20)
        train_m, valid_m = split_train_valid(mask, obs_sites, split_method="site-wise", seed=0)
        assert train_m.shape == mask.shape
        assert valid_m.shape == mask.shape

    def test_site_wise_no_overlap(self):
        T, S = 8, 20
        mask = np.ones((T, S), dtype=bool)
        obs_sites = np.arange(S)
        train_m, valid_m = split_train_valid(mask, obs_sites, split_method="site-wise", seed=0)
        assert not np.any(train_m & valid_m)

    def test_site_wise_ratio(self):
        T, S = 5, 20
        mask = np.ones((T, S), dtype=bool)
        obs_sites = np.arange(S)
        train_m, valid_m = split_train_valid(
            mask, obs_sites, split_method="site-wise", train_ratio=0.8, seed=1
        )
        n_train_sites = int(np.any(train_m, axis=0).sum())
        n_valid_sites = int(np.any(valid_m, axis=0).sum())
        assert n_train_sites == 16
        assert n_valid_sites == 4

    def test_random_shapes(self):
        mask = _make_obs_mask(T=10, S=15)
        obs_sites = np.arange(15)
        train_m, valid_m = split_train_valid(mask, obs_sites, split_method="random", seed=42)
        assert train_m.shape == mask.shape
        assert valid_m.shape == mask.shape

    def test_random_no_overlap(self):
        mask = _make_obs_mask()
        obs_sites = np.arange(20)
        train_m, valid_m = split_train_valid(mask, obs_sites, split_method="random", seed=0)
        assert not np.any(train_m & valid_m)

    def test_random_covers_all_obs(self):
        mask = _make_obs_mask()
        obs_sites = np.arange(20)
        train_m, valid_m = split_train_valid(mask, obs_sites, split_method="random", seed=7)
        combined = train_m | valid_m
        assert np.array_equal(combined, mask)

    def test_invalid_method_raises(self):
        mask = _make_obs_mask()
        obs_sites = np.arange(20)
        with pytest.raises(ValueError):
            split_train_valid(mask, obs_sites, split_method="unknown")

    def test_reproducible_with_seed(self):
        mask = _make_obs_mask()
        obs_sites = np.arange(20)
        t1, v1 = split_train_valid(mask, obs_sites, split_method="random", seed=5)
        t2, v2 = split_train_valid(mask, obs_sites, split_method="random", seed=5)
        assert np.array_equal(t1, t2)

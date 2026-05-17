"""Tests for da_stdk/losses/crps.py."""

import numpy as np
import pytest
import torch

from da_stdk.losses.crps import (
    check_loss_numpy,
    compute_coverage,
    compute_crps,
    compute_crps_multi_quantile,
    compute_picp,
    compute_qice,
    quantile_loss,
    trapezoidal_weights_for_quantiles,
)

QL = [0.05, 0.25, 0.5, 0.75, 0.95]
RNG = np.random.default_rng(0)


class TestCheckLossNumpy:
    def test_nonneg(self):
        y_pred = np.array([0.0, 1.0, 2.0])
        y_true = np.array([1.0, 1.0, 1.0])
        assert check_loss_numpy(y_pred, y_true, 0.5) >= 0

    def test_perfect_median_zero(self):
        y = np.array([1.0, 1.0, 1.0])
        assert check_loss_numpy(y, y, 0.5) == pytest.approx(0.0)


class TestQuantileLoss:
    def test_returns_scalar(self):
        y_pred = torch.randn(10)
        y_true = torch.randn(10)
        loss = quantile_loss(y_pred, y_true, 0.5)
        assert loss.shape == ()

    def test_nonneg(self):
        y_pred = torch.randn(20)
        y_true = torch.randn(20)
        for q in [0.1, 0.5, 0.9]:
            assert quantile_loss(y_pred, y_true, q) >= 0

    def test_2d_input(self):
        y_pred = torch.randn(10, 1)
        y_true = torch.randn(10, 1)
        loss = quantile_loss(y_pred, y_true, 0.5)
        assert loss.shape == ()


class TestTrapezoidalWeights:
    def test_sums_to_one(self):
        w = trapezoidal_weights_for_quantiles(QL)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-6)

    def test_single_quantile(self):
        w = trapezoidal_weights_for_quantiles([0.5])
        assert w[0] == pytest.approx(1.0)

    def test_length_matches(self):
        w = trapezoidal_weights_for_quantiles(QL)
        assert len(w) == len(QL)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            trapezoidal_weights_for_quantiles([])


class TestComputeCrps:
    def _preds_dict(self, n=50):
        preds = np.sort(RNG.standard_normal((n, len(QL))), axis=1)
        y = RNG.standard_normal(n)
        return {q: preds[:, i] for i, q in enumerate(QL)}, y

    def test_nonneg(self):
        d, y = self._preds_dict()
        assert compute_crps(d, y) >= 0

    def test_single_quantile(self):
        preds = {0.5: np.array([0.0, 0.0])}
        y = np.array([1.0, -1.0])
        score = compute_crps(preds, y)
        assert score >= 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_crps({}, np.array([1.0]))

    def test_wrong_weights_length_raises(self):
        d, y = self._preds_dict()
        with pytest.raises(ValueError):
            compute_crps(d, y, weights=[1.0, 2.0])

    def test_with_weights(self):
        d, y = self._preds_dict()
        w = np.ones(len(QL)) / len(QL)
        score = compute_crps(d, y, weights=w)
        assert score >= 0


class TestComputeCrpsMultiQuantile:
    def test_uniform_weighting(self):
        n = 100
        preds = np.sort(RNG.standard_normal((n, len(QL))), axis=1)
        y = RNG.standard_normal(n)
        score = compute_crps_multi_quantile(preds, y, QL, crps_weighting="uniform")
        assert score >= 0

    def test_trapezoidal_weighting(self):
        n = 100
        preds = np.sort(RNG.standard_normal((n, len(QL))), axis=1)
        y = RNG.standard_normal(n)
        score = compute_crps_multi_quantile(preds, y, QL, crps_weighting="trapezoidal")
        assert score >= 0

    def test_2d_y(self):
        preds = np.sort(RNG.standard_normal((50, len(QL))), axis=1)
        y = RNG.standard_normal((50, 1))
        score = compute_crps_multi_quantile(preds, y, QL)
        assert score >= 0


class TestComputePicp:
    def test_large_interval_full_coverage(self):
        n = 100
        preds = np.column_stack(
            [np.full(n, -1e6), np.full(n, 0), np.full(n, 0), np.full(n, 0), np.full(n, 1e6)]
        )
        y = RNG.standard_normal(n)
        cov = compute_picp(preds, y, QL)
        assert cov == pytest.approx(1.0)

    def test_2d_y_accepted(self):
        n = 50
        preds = np.sort(RNG.standard_normal((n, len(QL))), axis=1)
        y = RNG.standard_normal(n)
        c1 = compute_picp(preds, y, QL)
        c2 = compute_picp(preds, y.reshape(-1, 1), QL)
        assert c1 == pytest.approx(c2, abs=1e-6)

    def test_alias_coverage(self):
        n = 50
        preds = np.sort(RNG.standard_normal((n, len(QL))), axis=1)
        y = RNG.standard_normal(n)
        assert compute_picp(preds, y, QL) == pytest.approx(compute_coverage(preds, y, QL))


class TestComputeQice:
    def test_nonneg(self):
        n = 100
        preds = np.sort(RNG.standard_normal((n, len(QL))), axis=1)
        y = RNG.standard_normal(n)
        score = compute_qice(preds, y, QL)
        assert score >= 0

    def test_empty_y_nan(self):
        score = compute_qice(np.empty((0, 5)), np.array([]), QL)
        assert np.isnan(score)

    def test_too_few_quantiles_nan(self):
        preds = np.sort(RNG.standard_normal((50, 2)), axis=1)
        y = RNG.standard_normal(50)
        score = compute_qice(preds, y, [0.1, 0.9], M=4)
        assert np.isnan(score)

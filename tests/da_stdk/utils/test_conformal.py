"""Tests for da_stdk/utils/conformal.py."""

import numpy as np
import pytest

from da_stdk.utils.conformal import (
    _assign_nearest_center,
    compute_cluster_aware_cqr,
    compute_cluster_conformal_coverage,
    compute_conformal_coverage,
    compute_cqr_qhat,
)

RNG = np.random.default_rng(0)
QL = [0.05, 0.25, 0.5, 0.75, 0.95]


def _make_preds_y(n=100, q=5):
    preds = np.sort(RNG.standard_normal((n, q)), axis=1).astype(np.float32)
    y = RNG.standard_normal(n).astype(np.float32)
    return preds, y


# ---------------------------------------------------------------------------
# compute_cqr_qhat
# ---------------------------------------------------------------------------
class TestComputeCqrQhat:
    def test_returns_nonneg(self):
        preds, y = _make_preds_y()
        qhat, n = compute_cqr_qhat(preds, y, QL)
        assert qhat >= 0.0
        assert n == 100

    def test_2d_y_flattened(self):
        preds, y = _make_preds_y()
        qhat1, _ = compute_cqr_qhat(preds, y, QL)
        qhat2, _ = compute_cqr_qhat(preds, y.reshape(-1, 1), QL)
        assert qhat1 == pytest.approx(qhat2, abs=1e-6)

    def test_perfect_predictions_qhat_zero(self):
        """When all predictions exactly cover y, scores=0 so qhat=0."""
        n = 50
        y = np.zeros(n)
        preds = np.column_stack(
            [
                y - 1.0,  # q_lo
                y - 0.5,
                y,
                y + 0.5,
                y + 1.0,  # q_hi
            ]
        )
        qhat, _ = compute_cqr_qhat(preds, y, QL)
        assert qhat == 0.0

    def test_alpha_0p05(self):
        preds, y = _make_preds_y()
        qhat, n = compute_cqr_qhat(preds, y, QL, alpha=0.05)
        assert qhat >= 0.0

    def test_single_sample(self):
        preds = np.array([[0.1, 0.3, 0.5, 0.7, 0.9]])
        y = np.array([0.5])
        qhat, n = compute_cqr_qhat(preds, y, QL)
        assert n == 1
        assert qhat >= 0.0


# ---------------------------------------------------------------------------
# compute_conformal_coverage
# ---------------------------------------------------------------------------
class TestComputeConformalCoverage:
    def test_coverage_in_0_1(self):
        preds, y = _make_preds_y()
        qhat, _ = compute_cqr_qhat(preds, y, QL)
        cov = compute_conformal_coverage(preds, y, QL, qhat)
        assert 0.0 <= cov <= 1.0

    def test_large_qhat_full_coverage(self):
        preds, y = _make_preds_y()
        cov = compute_conformal_coverage(preds, y, QL, qhat=1e6)
        assert cov == pytest.approx(1.0)

    def test_2d_y_accepted(self):
        preds, y = _make_preds_y()
        qhat, _ = compute_cqr_qhat(preds, y, QL)
        cov1 = compute_conformal_coverage(preds, y, QL, qhat)
        cov2 = compute_conformal_coverage(preds, y.reshape(-1, 1), QL, qhat)
        assert cov1 == pytest.approx(cov2, abs=1e-6)


# ---------------------------------------------------------------------------
# _assign_nearest_center
# ---------------------------------------------------------------------------
class TestAssignNearestCenter:
    def test_simple_assignment(self):
        coords = np.array([[0.1, 0.1], [0.9, 0.9], [0.1, 0.2]])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        ids = _assign_nearest_center(coords, centers)
        assert list(ids) == [0, 1, 0]

    def test_1d_input_reshaped(self):
        coords = np.array([0.5, 0.5])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        ids = _assign_nearest_center(coords, centers)
        assert ids.shape == (1,)


# ---------------------------------------------------------------------------
# compute_cluster_aware_cqr
# ---------------------------------------------------------------------------
class TestComputeClusterAwareCqr:
    def _setup(self, n=200, n_centers=4):
        preds, y = _make_preds_y(n)
        coords = RNG.uniform(0, 1, (n, 2)).astype(np.float32)
        centers = RNG.uniform(0, 1, (n_centers, 2)).astype(np.float32)
        return preds, y, coords, centers

    def test_returns_four_values(self):
        preds, y, coords, centers = self._setup()
        result = compute_cluster_aware_cqr(preds, y, coords, centers, QL)
        assert len(result) == 4

    def test_global_qhat_nonneg(self):
        preds, y, coords, centers = self._setup()
        qhat_per_cluster, global_qhat, mean_qhat, n_fallback = compute_cluster_aware_cqr(
            preds, y, coords, centers, QL
        )
        assert global_qhat >= 0.0
        assert mean_qhat >= 0.0

    def test_per_cluster_nonneg(self):
        preds, y, coords, centers = self._setup()
        qhat_per_cluster, *_ = compute_cluster_aware_cqr(preds, y, coords, centers, QL)
        for v in qhat_per_cluster.values():
            assert v >= 0.0

    def test_2d_y_accepted(self):
        preds, y, coords, centers = self._setup()
        r1 = compute_cluster_aware_cqr(preds, y, coords, centers, QL)
        r2 = compute_cluster_aware_cqr(preds, y.reshape(-1, 1), coords, centers, QL)
        assert r1[1] == pytest.approx(r2[1], abs=1e-5)

    def test_global_qhat_fallback_used(self):
        preds, y, coords, centers = self._setup(n=200, n_centers=4)
        _, global_qhat, _, _ = compute_cluster_aware_cqr(
            preds, y, coords, centers, QL, min_n=10000, global_qhat_fallback=42.0
        )
        # all clusters fall back; provided fallback used
        qhat_per_cluster, *_ = compute_cluster_aware_cqr(
            preds, y, coords, centers, QL, min_n=10000, global_qhat_fallback=42.0
        )
        for v in qhat_per_cluster.values():
            assert v == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# compute_cluster_conformal_coverage
# ---------------------------------------------------------------------------
class TestComputeClusterConformalCoverage:
    def test_coverage_in_0_1(self):
        n = 200
        preds, y = _make_preds_y(n)
        coords = RNG.uniform(0, 1, (n, 2)).astype(np.float32)
        centers = RNG.uniform(0, 1, (4, 2)).astype(np.float32)
        qhat_per_cluster = {c: 0.5 for c in range(4)}
        cov = compute_cluster_conformal_coverage(preds, y, coords, centers, qhat_per_cluster, QL)
        assert 0.0 <= cov <= 1.0

    def test_large_qhat_full_coverage(self):
        n = 50
        preds, y = _make_preds_y(n)
        coords = RNG.uniform(0, 1, (n, 2)).astype(np.float32)
        centers = RNG.uniform(0, 1, (2, 2)).astype(np.float32)
        qhat_per_cluster = {0: 1e6, 1: 1e6}
        cov = compute_cluster_conformal_coverage(preds, y, coords, centers, qhat_per_cluster, QL)
        assert cov == pytest.approx(1.0)

    def test_missing_cluster_uses_fallback(self):
        n = 10
        preds, y = _make_preds_y(n)
        coords = np.zeros((n, 2), dtype=np.float32)
        centers = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        qhat_per_cluster = {}  # empty, all use global fallback
        cov = compute_cluster_conformal_coverage(
            preds, y, coords, centers, qhat_per_cluster, QL, global_qhat_fallback=1e6
        )
        assert cov == pytest.approx(1.0)

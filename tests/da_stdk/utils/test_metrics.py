"""Tests for da_stdk/utils/metrics.py."""

import numpy as np
import pytest
import torch

from da_stdk.utils.metrics import compute_metrics, compute_spatial_metrics, print_metrics

RNG = np.random.default_rng(1)


class TestComputeMetrics:
    def test_basic_shape(self):
        y = RNG.standard_normal(100).astype(np.float32)
        yp = RNG.standard_normal(100).astype(np.float32)
        m = compute_metrics(y, yp)
        assert set(m.keys()) >= {"rmse", "mae", "r2", "mse"}

    def test_perfect_predictions(self):
        y = np.arange(10, dtype=np.float32)
        m = compute_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-5)
        assert m["mae"] == pytest.approx(0.0, abs=1e-5)
        assert m["r2"] == pytest.approx(1.0, abs=1e-4)

    def test_rmse_positive(self):
        y = np.ones(50, dtype=np.float32)
        yp = np.zeros(50, dtype=np.float32)
        m = compute_metrics(y, yp)
        assert m["rmse"] == pytest.approx(1.0, abs=1e-5)

    def test_tensor_input(self):
        y = torch.randn(30)
        yp = torch.randn(30)
        m = compute_metrics(y, yp)
        assert "rmse" in m

    def test_nan_ignored(self):
        y = np.array([1.0, 2.0, np.nan, 4.0])
        yp = np.array([1.0, 2.0, 99.0, 4.0])
        m = compute_metrics(y, yp)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-5)

    def test_per_horizon(self):
        B, H, S = 2, 3, 4
        y = RNG.standard_normal((B, H, S, 1)).astype(np.float32)
        yp = RNG.standard_normal((B, H, S, 1)).astype(np.float32)
        m = compute_metrics(y, yp, per_horizon=True)
        assert "rmse_per_horizon" in m
        assert len(m["rmse_per_horizon"]) == H

    def test_per_horizon_false_no_extra_keys(self):
        y = RNG.standard_normal((2, 3, 4, 1)).astype(np.float32)
        yp = y.copy()
        m = compute_metrics(y, yp, per_horizon=False)
        assert "rmse_per_horizon" not in m


class TestComputeSpatialMetrics:
    def test_returns_three_keys(self):
        B, H, S = 2, 2, 10
        y = RNG.standard_normal((B, H, S, 1)).astype(np.float32)
        yp = RNG.standard_normal((B, H, S, 1)).astype(np.float32)
        coords = RNG.uniform(0, 1, (S, 2)).astype(np.float32)
        m = compute_spatial_metrics(y, yp, coords)
        assert "bin_centers" in m
        assert "rmse_by_distance" in m
        assert "mae_by_distance" in m

    def test_tensor_input(self):
        B, H, S = 2, 2, 8
        y = torch.randn(B, H, S, 1)
        yp = torch.randn(B, H, S, 1)
        coords = np.random.rand(S, 2).astype(np.float32)
        m = compute_spatial_metrics(y, yp, coords)
        assert len(m["bin_centers"]) > 0

    def test_bins_count(self):
        B, H, S = 3, 2, 20
        y = RNG.standard_normal((B, H, S, 1)).astype(np.float32)
        yp = y.copy()
        coords = RNG.uniform(0, 1, (S, 2)).astype(np.float32)
        m = compute_spatial_metrics(y, yp, coords, n_bins=3)
        assert len(m["bin_centers"]) <= 3


class TestPrintMetrics:
    def test_runs_without_error(self, capsys):
        m = {"rmse": 0.5, "mae": 0.4, "r2": 0.9, "mse": 0.25}
        print_metrics(m, prefix="Test")
        out = capsys.readouterr().out
        assert "RMSE" in out

    def test_per_horizon_printed(self, capsys):
        m = {
            "rmse": 0.5,
            "mae": 0.4,
            "r2": 0.9,
            "mse": 0.25,
            "rmse_per_horizon": [0.4, 0.5, 0.6],
            "mae_per_horizon": [0.3, 0.4, 0.5],
        }
        print_metrics(m)
        out = capsys.readouterr().out
        assert "horizon" in out

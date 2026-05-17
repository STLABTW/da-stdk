"""Tests for da_stdk/viz/table44.py helper functions and create_table_4_4."""

import numpy as np
import pytest

from da_stdk.viz.kaust_analysis import (
    _assign_nearest_center,
    _cov_spatial_stats,
    _per_site_coverage,
    _scenario_label,
    _width_stats,
    create_table_4_4,
    print_table_4_4,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# _assign_nearest_center
# ---------------------------------------------------------------------------
class TestAssignNearestCenter:
    def test_simple(self):
        coords = np.array([[0.1, 0.1], [0.9, 0.9]])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        ids = _assign_nearest_center(coords, centers)
        assert list(ids) == [0, 1]

    def test_invalid_shape_returns_none(self):
        result = _assign_nearest_center(np.array([0.1]), np.array([[0.0, 0.0]]))
        assert result is None


# ---------------------------------------------------------------------------
# _width_stats
# ---------------------------------------------------------------------------
class TestWidthStats:
    def test_basic(self):
        width = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, True, True, True])
        mean, p90 = _width_stats(width, mask)
        assert mean == pytest.approx(2.5)

    def test_none_inputs(self):
        mean, p90 = _width_stats(None, None)
        assert np.isnan(mean)

    def test_empty_mask(self):
        width = np.array([1.0, 2.0])
        mask = np.array([False, False])
        mean, p90 = _width_stats(width, mask)
        assert np.isnan(mean)


# ---------------------------------------------------------------------------
# _per_site_coverage
# ---------------------------------------------------------------------------
class TestPerSiteCoverage:
    def _setup(self, T=10, S=20):
        z = RNG.standard_normal((T, S)).astype(np.float32)
        q_lo = z - 1.0
        q_hi = z + 1.0
        test_mask = np.ones((T, S), dtype=bool)
        return z, q_lo, q_hi, test_mask

    def test_perfect_coverage(self):
        z, q_lo, q_hi, test_mask = self._setup()
        cov = _per_site_coverage(z, q_lo, q_hi, test_mask)
        assert cov is not None
        assert np.allclose(cov, 1.0, atol=1e-5)

    def test_none_input_returns_none(self):
        assert _per_site_coverage(None, None, None, None) is None

    def test_with_qhat(self):
        z, q_lo, q_hi, test_mask = self._setup(T=5, S=10)
        qhat_per_site = np.zeros((1, 10))
        cov = _per_site_coverage(z, q_lo, q_hi, test_mask, qhat_per_site=qhat_per_site)
        assert cov is not None
        assert len(cov) == 10


# ---------------------------------------------------------------------------
# _cov_spatial_stats
# ---------------------------------------------------------------------------
class TestCovSpatialStats:
    def test_basic(self):
        cov = np.array([0.8, 0.9, 0.85, 0.7, 0.95])
        std, worst10 = _cov_spatial_stats(cov)
        assert std >= 0.0

    def test_none_returns_nan(self):
        std, worst10 = _cov_spatial_stats(None)
        assert np.isnan(std)

    def test_all_nan_returns_nan(self):
        cov = np.array([np.nan, np.nan])
        std, worst10 = _cov_spatial_stats(cov)
        assert np.isnan(std)


# ---------------------------------------------------------------------------
# create_table_4_4
# ---------------------------------------------------------------------------
def _make_results(n_each=3):
    """Create minimal synthetic results list."""
    scenarios = ["Fixed_Uniform", "Fixed_Clustered", "Random_Uniform", "Random_Clustered"]
    models = ["STDK", "DA-STDK"]
    results = []
    rng = np.random.default_rng(1)
    for scenario in scenarios:
        for model in models:
            for _ in range(n_each):
                results.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "test_crps": float(rng.uniform(0.1, 0.5)),
                        "test_picp_90": float(rng.uniform(0.85, 0.95)),
                        "test_coverage_90_conformal": float(rng.uniform(0.85, 0.95)),
                        "test_coverage_90_conformal_global": float(rng.uniform(0.85, 0.95)),
                        "test_coverage_90_conformal_cluster": float(rng.uniform(0.85, 0.95)),
                        "conformal_qhat": float(rng.uniform(0.1, 0.3)),
                    }
                )
    return results


class TestCreateTable44:
    def test_returns_three_values(self):
        results = _make_results()
        pivot_df, pivot_std, summary_df = create_table_4_4(results)
        assert pivot_df is not None
        assert pivot_std is not None
        assert summary_df is not None

    def test_summary_has_rows(self):
        results = _make_results()
        _, _, summary_df = create_table_4_4(results)
        assert len(summary_df) > 0

    def test_pivot_has_model_columns(self):
        results = _make_results()
        pivot_df, _, _ = create_table_4_4(results)
        assert "STDK" in pivot_df.columns or "DA-STDK" in pivot_df.columns

    def test_single_row_results(self):
        results = _make_results(n_each=1)
        pivot_df, pivot_std, summary_df = create_table_4_4(results)
        assert len(summary_df) > 0


# ---------------------------------------------------------------------------
# print_table_4_4
# ---------------------------------------------------------------------------
class TestPrintTable44:
    def test_runs_without_error(self, capsys):
        results = _make_results()
        pivot_df, pivot_std, summary_df = create_table_4_4(results)
        print_table_4_4(pivot_df, pivot_std, summary_df)
        out = capsys.readouterr().out
        assert "CRPS" in out

    def test_coverage_summary_printed(self, capsys):
        results = _make_results()
        pivot_df, pivot_std, summary_df = create_table_4_4(results)
        print_table_4_4(pivot_df, pivot_std, summary_df)
        out = capsys.readouterr().out
        assert "Coverage" in out


# ---------------------------------------------------------------------------
# _scenario_label
# ---------------------------------------------------------------------------
class TestScenarioLabel:
    def test_fixed_uniform(self):
        label = _scenario_label("Fixed_Uniform")
        assert isinstance(label, str)
        assert len(label) > 0

    def test_random_clustered(self):
        label = _scenario_label("Random_Clustered")
        assert isinstance(label, str)

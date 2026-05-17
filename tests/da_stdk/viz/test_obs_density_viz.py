"""Tests for da_stdk/viz/obs_density.py."""

import numpy as np
import pytest

from da_stdk.viz.obs_density import compute_observation_density, plot_observation_density_maps

RNG = np.random.default_rng(0)


def _make_data(T=10, S=20):
    z = RNG.standard_normal((T, S)).astype(np.float32)
    coords = RNG.uniform(0, 1, (S, 2)).astype(np.float32)
    return z, coords


class TestComputeObservationDensity:
    def test_site_wise_uniform_shape(self):
        z, coords = _make_data()
        density = compute_observation_density(z, coords, "site-wise", "uniform")
        assert density.shape == (20,)

    def test_random_corner_shape(self):
        z, coords = _make_data()
        density = compute_observation_density(z, coords, "random", "corner")
        assert density.shape == (20,)

    def test_values_in_0_1(self):
        z, coords = _make_data()
        density = compute_observation_density(z, coords, "random", "uniform")
        assert np.all(density >= 0) and np.all(density <= 1)


class TestPlotObservationDensityMaps:
    def test_returns_figure_no_save(self):
        z, coords = _make_data()
        fig = plot_observation_density_maps(z_data=z, coords=coords)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_saves_file(self, tmp_path):
        z, coords = _make_data()
        out = tmp_path / "density.png"
        fig = plot_observation_density_maps(z_data=z, coords=coords, save_path=out)
        assert out.exists()
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_missing_data_raises(self):
        with pytest.raises(ValueError):
            plot_observation_density_maps(z_data=None, coords=None)

    def test_csv_data_path(self, tmp_path):
        import pandas as pd

        T, S = 5, 10
        rng = np.random.default_rng(7)
        coords = rng.uniform(0, 1, (S, 2))
        records = []
        for t in range(T):
            for s in range(S):
                records.append(
                    {"t": t, "x": coords[s, 0], "y": coords[s, 1], "z": rng.standard_normal()}
                )
        df = pd.DataFrame(records)
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        fig = plot_observation_density_maps(data_path=csv_path)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

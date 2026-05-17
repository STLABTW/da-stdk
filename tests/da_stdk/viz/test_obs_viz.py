"""Tests for da_stdk/viz/obs.py."""

import numpy as np

from da_stdk.viz.obs import create_observation_density_map, plot_observation_pattern


class TestPlotObservationPattern:
    def test_creates_file(self, tmp_path):
        T, S = 10, 20
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 1, (S, 2)).astype(np.float32)
        obs_mask = rng.random((T, S)) < 0.6
        train_mask = obs_mask.copy()
        train_mask[:, :4] = False
        valid_mask = np.zeros((T, S), dtype=bool)
        valid_mask[:, :4] = obs_mask[:, :4]
        plot_observation_pattern(coords, obs_mask, train_mask, valid_mask, tmp_path)
        assert (tmp_path / "observation_pattern.png").exists()


class TestCreateObservationDensityMap:
    def test_creates_file(self, tmp_path):
        T, S = 5, 10
        rng = np.random.default_rng(1)
        coords = rng.uniform(0, 1, (S, 2)).astype(np.float32)
        masks = [rng.random((T, S)) < 0.5 for _ in range(3)]
        create_observation_density_map(masks, coords, tmp_path)
        assert (tmp_path / "observation_density.png").exists()

    def test_empty_masks_skips(self, tmp_path, capsys):
        coords = np.zeros((5, 2))
        create_observation_density_map([], coords, tmp_path)
        out = capsys.readouterr().out
        assert "No train masks" in out

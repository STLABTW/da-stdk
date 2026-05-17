"""Tests for da_stdk/viz/training.py."""

from da_stdk.viz.training import plot_training_curves


class TestPlotTrainingCurves:
    def _history(self, n=5):
        return {
            "train_loss": [1.0 - i * 0.1 for i in range(n)],
            "val_loss": [1.1 - i * 0.09 for i in range(n)],
            "val_rmse": [0.9 - i * 0.05 for i in range(n)],
            "lr": [0.01 * (0.9**i) for i in range(n)],
        }

    def test_creates_file(self, tmp_path):
        history = self._history(5)
        out = tmp_path / "curves.png"
        plot_training_curves(history, out)
        assert out.exists()

    def test_single_epoch(self, tmp_path):
        history = self._history(1)
        out = tmp_path / "curves.png"
        plot_training_curves(history, out)
        assert out.exists()

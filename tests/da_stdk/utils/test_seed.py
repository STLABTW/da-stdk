"""Tests for da_stdk/utils/seed.py."""

import random

import numpy as np
import torch

from da_stdk.utils.seed import set_seed


class TestSetSeed:
    def test_reproducible_random(self):
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_reproducible_numpy(self):
        set_seed(7)
        a = np.random.rand(5)
        set_seed(7)
        b = np.random.rand(5)
        assert np.allclose(a, b)

    def test_reproducible_torch(self):
        set_seed(123)
        a = torch.rand(5)
        set_seed(123)
        b = torch.rand(5)
        assert torch.allclose(a, b)

    def test_cuda_path_mocked(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "manual_seed", lambda s: None)
        monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda s: None)
        set_seed(0)  # just ensure it doesn't raise

    def test_prints_seed(self, capsys):
        set_seed(99)
        out = capsys.readouterr().out
        assert "99" in out

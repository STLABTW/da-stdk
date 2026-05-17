"""Tests for da_stdk.training.config (TrainConfig, from_dict, from_yaml)."""

from pathlib import Path

import pytest

from da_stdk.training.config import from_dict, from_yaml


def test_from_dict_minimal():
    d = {"data_file": "data/2b/2b_8.csv", "epochs": 10, "lr": 0.01}
    cfg = from_dict(d)
    assert cfg.data_file == "data/2b/2b_8.csv"
    assert cfg.epochs == 10
    assert cfg.lr == 0.01
    assert cfg.quantile_levels == [0.05, 0.25, 0.5, 0.75, 0.95]
    assert cfg.k_spatial_centers == [25, 81, 121]


def test_config_get_dict_compat():
    """TrainConfig.get(key, default) should behave like dict.get for compatibility."""
    cfg = from_dict({"lr": 0.01, "patience": 50})
    assert cfg.get("lr", 99) == 0.01
    assert cfg.get("patience", 0) == 50
    assert cfg.get("nonexistent_key", 42) == 42
    assert cfg.get("epochs", 100) == 500  # default from dataclass


def test_from_yaml_loads_main_config():
    path = Path(__file__).resolve().parents[3] / "configs" / "config_default.yaml"
    if not path.exists():
        pytest.skip("config_default.yaml not found")
    cfg = from_yaml(path)
    assert cfg.data_file == "data/2a/2a_8.csv"
    assert cfg.epochs == 500
    assert cfg.get("patience", 0) == 50


def test_to_dict():
    cfg = from_dict({"tag": "test", "epochs": 3})
    d = cfg.to_dict()
    assert d["tag"] == "test"
    assert d["epochs"] == 3
    assert "quantile_levels" in d


def test_from_dict_override_behavior():
    """Partial dict overrides specified keys; unspecified keys use paper defaults."""
    cfg = from_dict({"lr": 0.02, "epochs": 100, "data_file": "data/2a/2a_8.csv"})
    assert cfg.lr == 0.02
    assert cfg.epochs == 100
    assert cfg.data_file == "data/2a/2a_8.csv"
    assert cfg.quantile_levels == [0.05, 0.25, 0.5, 0.75, 0.95]
    assert cfg.get("patience", 0) == 50

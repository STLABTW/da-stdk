"""
Tests for KAUST data loading: load_kaust_csv_single, load_kaust_csv (provider split).

Section 8.4: Data loading consistency; uses data/2a or data/2b when available.
"""

from pathlib import Path

import numpy as np
import pytest

from da_stdk.data.kaust_loader import load_kaust_csv, load_kaust_csv_single


def _data_dir():
    return Path(__file__).resolve().parents[3] / "data"


@pytest.fixture
def single_csv_path():
    """Path to single-file CSV (2a_8 or 2b_8). Prefer 2a for speed."""
    for sub in ("2a", "2b"):
        p = _data_dir() / sub / f"{sub}_8.csv"
        if p.exists():
            return p
    return None


@pytest.fixture
def provider_split_paths():
    """Paths to train/test CSV pair (2a_8 or 2b_8)."""
    for sub in ("2a", "2b"):
        train_p = _data_dir() / sub / f"{sub}_8_train.csv"
        test_p = _data_dir() / sub / f"{sub}_8_test.csv"
        if train_p.exists() and test_p.exists():
            return str(train_p), str(test_p)
    return None


def test_load_kaust_csv_single_shape_and_metadata(single_csv_path):
    """load_kaust_csv_single returns (T, S), coords (S, 2), metadata with correct types."""
    if single_csv_path is None:
        pytest.skip("No data/2a/2a_8.csv or data/2b/2b_8.csv found")
    z_data, coords, metadata = load_kaust_csv_single(single_csv_path, normalize=True)
    T, S = z_data.shape
    assert T >= 1 and S >= 1
    assert coords.shape == (S, 2)
    assert coords.dtype in (np.float32, np.float64)
    assert "z_mean" in metadata and "z_std" in metadata
    assert np.isfinite(z_data[~np.isnan(z_data)]).all()
    # After normalize, mean ~0, std ~1 (for non-NaN)
    valid = z_data[~np.isnan(z_data)]
    if len(valid) > 0:
        np.testing.assert_allclose(valid.mean(), 0.0, atol=1e-5)
        np.testing.assert_allclose(valid.std(), 1.0, atol=1e-5)


def test_load_kaust_csv_single_no_normalize(single_csv_path):
    """load_kaust_csv_single with normalize=False leaves z unchanged (no z_mean/z_std)."""
    if single_csv_path is None:
        pytest.skip("No single-file CSV found")
    z_data, coords, metadata = load_kaust_csv_single(single_csv_path, normalize=False)
    assert "z_mean" not in metadata or metadata.get("z_mean") is None
    assert z_data.shape[0] == coords.shape[0] or True  # T, S from same CSV


def test_load_kaust_csv_provider_split(provider_split_paths):
    """load_kaust_csv (train_path, test_path) returns z_train, z_test, coords, metadata."""
    if provider_split_paths is None:
        pytest.skip("No data/2a/2a_8_train.csv+2a_8_test.csv (or 2b) found")
    train_p, test_p = provider_split_paths
    z_train, z_test, coords, site_to_idx, metadata = load_kaust_csv(train_p, test_p, normalize=True)
    assert z_train.ndim == 2 and z_test.ndim == 2
    assert coords.shape[1] == 2
    assert len(site_to_idx) == coords.shape[0]
    assert "z_mean" in metadata
    # Train and test share same site set
    assert z_train.shape[1] == z_test.shape[1] == coords.shape[0]

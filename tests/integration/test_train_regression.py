"""
End-to-end regression test for training pipeline (Phase 0 safety net).

Runs a short training (fixed seed, 5 epochs) and asserts test_crps is finite
and in a reasonable range. Use data/2a/2a_8.csv for speed.
Marked as slow; run with: pytest tests/integration/ -m slow
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Project root (parent of tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "tests" / "fixtures" / "config_regression.yaml"
DATA_FILE = PROJECT_ROOT / "data" / "2a" / "2a_8.csv"

# Reasonable CRPS bounds for 5-epoch STDK on 2a_8 (seed 2025)
# Looser than full training; purpose is to catch NaN/Inf or broken pipeline
TEST_CRPS_MIN = 0.01
TEST_CRPS_MAX = 2.0


@pytest.fixture(scope="module")
def data_available():
    if not DATA_FILE.exists():
        pytest.skip(f"Regression test data not found: {DATA_FILE}")


@pytest.mark.slow
@pytest.mark.integration
def test_train_regression_stdk_short_run(data_available, tmp_path_factory):
    """Run one experiment (STDK, 5 epochs, seed=2025) and check test_crps."""
    out_dir = tmp_path_factory.mktemp("regression_out")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_default.py"),
        "--config",
        str(CONFIG_PATH),
        "--output_dir",
        str(out_dir),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert (
        result.returncode == 0
    ), f"Training failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    results_file = out_dir / "experiments" / "1" / "results.json"
    assert results_file.exists(), f"Expected {results_file} after training"

    with open(results_file) as f:
        results = json.load(f)

    test_crps = results.get("test_crps")
    assert test_crps is not None, "results.json must contain 'test_crps'"

    assert isinstance(test_crps, (int, float)), "test_crps must be numeric"
    assert not (test_crps != test_crps), "test_crps must not be NaN"
    assert abs(test_crps) != float("inf"), "test_crps must not be Inf"
    assert (
        TEST_CRPS_MIN <= test_crps <= TEST_CRPS_MAX
    ), f"test_crps={test_crps} outside expected range [{TEST_CRPS_MIN}, {TEST_CRPS_MAX}]"

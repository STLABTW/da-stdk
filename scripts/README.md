# Scripts Usage Guide

This directory contains executable scripts for training, KAUST experiment runs, analysis, and visualization.

## Table of Contents

- [Training Scripts](#training-scripts)
- [KAUST experiment and Analysis](#kaust-experiment-and-analysis)
- [Visualization Scripts](#visualization-scripts)
- [Configuration Files](#configuration-files)

## Training Scripts

### `train_st_interp.py`

Train a single spatio-temporal interpolation model.

**Basic Usage:**
```bash
python scripts/train_st_interp.py --config configs/config_st_interp.yaml
```

**Arguments:**
- `--config` (str, default: `configs/config_st_interp.yaml`): Path to configuration YAML file
- `--data_file` (str, optional): Override data file path from config
- `--n_experiments` (int, optional): Override number of experiments from config
- `--base_seed` (int, optional): Override base seed from config
- `--output_dir` (str, optional): Override base output directory
- `--parallel` (flag): Run multiple experiments in parallel
- `--n_jobs` (int, default: -1): Number of parallel jobs (-1 for all CPUs, 0 for sequential)
- `--start_exp_id` (int, optional): Starting experiment ID (1-based)
- `--end_exp_id` (int, optional): Ending experiment ID (inclusive)
- `--skip-existing` (flag): Skip experiments that already have results.json

**Examples:**
```bash
# Train with default config (KAUST experiment uses data/2b/2b_8.csv)
python scripts/train_st_interp.py --config configs/config_st_interp.yaml

# Train with custom data file
python scripts/train_st_interp.py --config configs/config_st_interp.yaml --data_file data/2b/2b_8.csv

# Run 10 experiments in parallel
python scripts/train_st_interp.py --config configs/config_st_interp.yaml --n_experiments 10 --parallel --n_jobs 4

# Resume experiments 5-10
python scripts/train_st_interp.py --config configs/config_st_interp.yaml --start_exp_id 5 --end_exp_id 10 --skip-existing
```

## KAUST experiment and Analysis

### `run_kaust_data.py`

Run the KAUST data experiment: 4 scenarios (Fixed/Random × Uniform/Clustered) × 2 backbones (STDK, DA-STDK). Overrides config (data_file, obs_method, obs_spatial_pattern, spatial_learnable, etc.) per scenario. Non-crossing λ is tuned via grid search (`grid_search_non_crossing_lambda.py`); pass `--non_crossing_lambda` for the chosen value.

**Usage:**
```bash
python scripts/run_kaust_data.py
```

See script help for optional arguments (e.g. output base dir, n_experiments).

### `analyze_table_4_4.py`

Analyze KAUST experiment results: coverage tables, CRPS, and summary figures.

**Usage:**
```bash
python scripts/analyze_table_4_4.py <results_dir>
```

### `grid_search_non_crossing_lambda.py`

Grid search over non-crossing penalty λ. Run this to tune λ; then pass `--non_crossing_lambda` to `run_kaust_data.py` for the KAUST data experiment.

## Visualization Scripts

### `visualize_2b_data.py`

Visualize 2b dataset characteristics.

```bash
python scripts/visualize_2b_data.py
```

### `visualize_obs_density.py`

Visualize observation density patterns.

```bash
python scripts/visualize_obs_density.py
```

## Configuration Files

Configs are in `configs/`:

- **config_st_interp.yaml**: Main config (paper-aligned defaults; KAUST experiment uses 2b_8, lr 0.01, crps_weighting trapezoidal). Non-crossing λ is tuned via grid search.
- **configs/experimental/**: Ablation/experiment configs (conformal ratio sweep, demo). Use by path, e.g. `--config configs/experimental/config_conformal_demo.yaml`.

Key sections in the main config:
- Data (data_file, obs_ratio, split_method, train_ratio)
- Model (k_spatial_centers, spatial_learnable, spatial_init_method)
- Training (epochs, lr, basis_lr_ratio)
- Conformal (conformal_alpha, conformal_mode, crps_weighting)

## Makefile

- `make train`: Train with default config
- `make table44`: Run KAUST experiment
- `make test`: Run unit tests
- `make test-cov`: Run tests with coverage

For full list: `make help`.

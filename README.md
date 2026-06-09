# DA-STDK

[![PyPI](https://img.shields.io/pypi/v/da-stdk)](https://pypi.org/project/da-stdk/)
[![Python](https://img.shields.io/pypi/pyversions/da-stdk)](https://pypi.org/project/da-stdk/)
[![Tests](https://github.com/STLABTW/da-stdk/actions/workflows/test.yaml/badge.svg)](https://github.com/STLABTW/da-stdk/actions/workflows/test.yaml)
[![Code Quality](https://github.com/STLABTW/da-stdk/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/STLABTW/da-stdk/actions/workflows/code-quality.yaml)

Reference code for **cluster-aware conformal calibration** in spatio-temporal distributional prediction.

- **Cluster-adaptive spatial bases** — centers and scales initialized from sampling density, so capacity follows heterogeneous observation patterns instead of a fixed grid.
- **Cluster-aware conformal calibration** — interval widths are calibrated within spatial clusters, with a global fallback when local samples are scarce.

Benchmarks in this repo use the KAUST spatio-temporal datasets (scenarios 2a/2b).

## Architecture

![Model backbone](https://raw.githubusercontent.com/STLABTW/da-stdk/main/artifacts/backbone.png)

Cluster-adaptive spatial basis, temporal basis, and covariates are concatenated and passed through a shared MLP trunk. Quantile heads predict multiple levels; cluster-aware CQR produces calibrated prediction intervals.

## Install

Python 3.10+ and [Poetry](https://python-poetry.org/) are enough for most use:

```bash
poetry install --with dev
```

Optional: Conda env via `bash envs/conda/build_conda_env.sh` then `conda activate st-dadk`.

```bash
pip install da-stdk   # after a PyPI release
# or locally:
pip install -e .
```

```python
import da_stdk
from da_stdk.models import STDKMLP, create_model
from da_stdk.data.kaust_loader import load_kaust_csv_single
```

## Run

**Single training run**

```bash
poetry run python scripts/train_default.py
```

**KAUST benchmark (multiple scenarios / models)**

```bash
make kaust
# or (train only, then analyze manually):
poetry run python scripts/run_kaust_data.py --config configs/config_default.yaml
poetry run python scripts/analyze_kaust_results.py --results_dir results/kaust_data_<timestamp>
```

`make kaust` runs all scenario×model combos and calls `analyze_kaust_results.py` when finished (`--analyze`). Use `make kaust-dry` to preview commands.

More scripts and flags: [`scripts/README.md`](scripts/README.md).

## Layout

| Path | Contents |
|------|----------|
| `da_stdk/` | Models, training, data I/O, conformal utils, viz |
| `scripts/` | Training and experiment drivers |
| `configs/` | YAML configs |
| `data/` | KAUST CSVs (large; not on PyPI) |

## Dev

```bash
make test          # pytest
make lint          # black, isort, mypy
pre-commit run --all-files
```

## Citation

If you use this code, please cite:

> **Cluster-Aware Conformal Calibration for Spatio-Temporal Distributional Prediction**
> Gooyoung Kim, Chae Young Lim, Wen-Ting Wang, Hao-Yun Huang, Wei-Ying Wu
> arXiv preprint (link forthcoming)

```bibtex
@misc{kim2026clusterawareconformalcalibrationspatiotemporal,
      title={Cluster-Aware Conformal Calibration for Spatio-Temporal Distributional Prediction}, 
      author={Gooyoung Kim and Chae Young Lim and Wen-Ting Wang and Hao-Yun Huang and Wei-Ying Wu},
      year={2026},
      eprint={2606.06753},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2606.06753}, 
}
```


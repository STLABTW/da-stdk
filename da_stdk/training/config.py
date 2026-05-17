"""Typed training config (YAML / dict compatible via ``.get``)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml


def _default_k_spatial() -> List[int]:
    return [25, 81, 121]


def _default_k_temporal() -> List[int]:
    return [10, 15, 45]


def _default_hidden_dims() -> List[int]:
    return [256, 256, 128]


def _default_quantile_levels() -> List[float]:
    return [0.05, 0.25, 0.5, 0.75, 0.95]


@dataclass
class TrainConfig:
    """All training hyperparameters; load via ``from_yaml`` / ``from_dict``."""

    # Experiment / tag
    tag: str = "integrated"
    output_dir: Optional[str] = None  # Often set at runtime

    # Data (KAUST experiment)
    data_file: str = "data/2b/2b_8.csv"
    use_provider_split: bool = False
    obs_method: str = "site-wise"
    obs_ratio: float = 0.1
    obs_spatial_pattern: str = "corner"
    obs_spatial_intensity: float = 10.0
    split_method: str = "random"
    train_ratio: float = 0.8
    calibration_ratio_from_train: float = 0.3
    calibration_split_method: str = "random"
    normalize_target: bool = False

    # Model
    k_spatial_centers: List[int] = field(default_factory=_default_k_spatial)
    k_temporal_centers: List[int] = field(default_factory=_default_k_temporal)
    spatial_basis_function: str = "wendland"
    spatial_init_method: str = "uniform"
    spatial_learnable: bool = False
    gradient_damping: bool = True
    damping_threshold: float = 0.0
    damping_strength: float = 5.0
    domain_penalty_weight: float = 0.01
    movement_penalty_weight: float = 0.0
    sparsity_penalty_type: str = "none"
    sparsity_lambda_l1: float = 0.0
    sparsity_lambda_group: float = 0.0
    sparsity_apply_to_spatial: bool = True
    sparsity_apply_to_temporal: bool = False
    sparsity_threshold_ratio: float = 0.01
    non_crossing_lambda: float = 0.0
    non_crossing_l1_penalization: bool = True
    non_crossing_weight: float = 0.0
    non_crossing_power: int = 1
    hidden_dims: List[int] = field(default_factory=_default_hidden_dims)
    dropout: float = 0.1
    layernorm: bool = True
    p_covariates: int = 0

    # Conformal / quantile
    conformal_mode: str = "both"
    conformal_center_source: str = "init"
    conformal_cluster_min_n: int = 30
    conformal_alpha: float = 0.1
    crps_weighting: str = "trapezoidal"
    save_predictions_quantile: bool = True
    regression_type: str = "multi-quantile"
    use_delta_reparameterization: bool = False
    current_quantile: Optional[float] = None
    quantile_levels: List[float] = field(default_factory=_default_quantile_levels)

    # Training
    epochs: int = 500
    lr: float = 0.01
    basis_lr_ratio: float = 0.05
    weight_decay: float = 5e-4
    batch_size: int = 4096
    patience: int = 50
    grad_clip: float = 10.0
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    basis_unfreeze_epoch: int = 10
    basis_lr_rampup_epochs: int = 10

    # Experiment
    n_experiments: int = 10
    base_seed: int = 2025
    num_workers: int = 0

    # Optional / backward compat
    use_ema: bool = True
    use_positive_p_nc_penalty: bool = False
    use_check_loss_early_stop: bool = False
    device: str = "cpu"

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get for compatibility with code that uses config.get('key', default)."""
        field_names = {f.name for f in fields(self)}
        if key not in field_names:
            return default
        return getattr(self, key)

    def __getitem__(self, key: str) -> Any:
        """Dict-like access: config['key']."""
        field_names = {f.name for f in fields(self)}
        if key not in field_names:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like assignment: config['key'] = value (for overrides at runtime)."""
        field_names = {f.name for f in fields(self)}
        if key not in field_names:
            raise KeyError(f"Unknown config key: {key}")
        object.__setattr__(self, key, value)

    def to_dict(self) -> dict:
        """Export to a plain dict (e.g. for serialization or merging)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def copy(self) -> "TrainConfig":
        """Return a new TrainConfig with the same values (for overrides like config.copy())."""
        return TrainConfig(**self.to_dict())


def from_dict(d: dict) -> TrainConfig:
    """Build TrainConfig from a dict (e.g. loaded from YAML). Unknown keys are ignored."""
    field_names = {f.name for f in fields(TrainConfig)}
    kwargs = {k: v for k, v in d.items() if k in field_names}
    return TrainConfig(**kwargs)


def from_yaml(path: Union[str, Path]) -> TrainConfig:
    """Load config from a YAML file."""
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML to load as dict, got {type(data)}")
    return from_dict(data)

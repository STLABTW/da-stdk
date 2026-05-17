"""CRPS (Eq. 4.6), check loss, PICP, and QICE."""

import numpy as np
import torch


def check_loss_numpy(y_pred, y_true, quantile):
    """Mean quantile (check) loss, NumPy."""
    errors = y_true - y_pred
    return np.mean(np.maximum((quantile - 1) * errors, quantile * errors))


def quantile_loss(y_pred: torch.Tensor, y_true: torch.Tensor, quantile: float) -> torch.Tensor:
    """Mean quantile (check) loss, PyTorch."""
    if y_pred.dim() > 1:
        y_pred = y_pred.squeeze(1)
    if y_true.dim() > 1:
        y_true = y_true.squeeze(1)
    errors = y_true - y_pred
    return torch.mean(torch.max((quantile - 1) * errors, quantile * errors))


def trapezoidal_weights_for_quantiles(quantile_levels):
    """Normalized trapezoidal weights for CRPS over quantile levels."""
    q = np.asarray(quantile_levels, dtype=float)
    q = np.sort(q)
    K = len(q)
    if K == 0:
        raise ValueError("quantile_levels cannot be empty")
    if K == 1:
        return np.ones(1)
    w = np.empty(K)
    w[0] = (q[1] - q[0]) / 2.0
    for k in range(1, K - 1):
        w[k] = (q[k + 1] - q[k - 1]) / 2.0
    w[K - 1] = (q[K - 1] - q[K - 2]) / 2.0
    w = w / w.sum()
    return w


def compute_crps(predictions_dict, y_true, weights=None):
    """CRPS from ``{τ: predictions}``; 2 Σ_k w_k ρ_τ(y - Q_τ)."""
    quantiles = sorted(predictions_dict.keys())
    K = len(quantiles)
    if K == 0:
        raise ValueError("predictions_dict cannot be empty")
    if K == 1:
        q = quantiles[0]
        check_loss = check_loss_numpy(predictions_dict[q], y_true, q)
        return 2.0 * check_loss
    if weights is None:
        weights = np.ones(K) / K
    else:
        weights = np.asarray(weights)
        if len(weights) != K:
            raise ValueError(
                f"weights length ({len(weights)}) must match number of quantiles ({K})"
            )
        weights = weights / weights.sum()
    crps_sum = 0.0
    for i, q in enumerate(quantiles):
        pred = predictions_dict[q]
        check_loss_q = check_loss_numpy(pred, y_true, q)
        crps_sum += weights[i] * check_loss_q
    return 2.0 * crps_sum


def compute_crps_multi_quantile(
    preds, y_true, quantile_levels, weights=None, crps_weighting="uniform"
):
    """CRPS from (N, Q) preds; ``crps_weighting``: ``uniform`` or ``trapezoidal``."""
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if weights is None and crps_weighting == "trapezoidal":
        weights = trapezoidal_weights_for_quantiles(quantile_levels)
    predictions_dict = {q: preds[:, i] for i, q in enumerate(quantile_levels)}
    return compute_crps(predictions_dict, y_true, weights=weights)


def compute_picp(preds, y_true, quantile_levels, alpha=0.1):
    """Empirical coverage of nominal (1 - alpha) quantile interval."""
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    q_lo = alpha / 2
    q_hi = 1.0 - alpha / 2
    quantile_levels = np.asarray(quantile_levels)
    idx_lo = np.argmin(np.abs(quantile_levels - q_lo))
    idx_hi = np.argmin(np.abs(quantile_levels - q_hi))
    low = preds[:, idx_lo]
    high = preds[:, idx_hi]
    inside = (y_true >= low) & (y_true <= high)
    return float(np.mean(inside))


def compute_coverage(preds, y_true, quantile_levels, alpha=0.1):
    """Alias for ``compute_picp``."""
    return compute_picp(preds, y_true, quantile_levels, alpha=alpha)


def compute_qice(preds, y_true, quantile_levels, M=4):
    """QICE: mean |empirical coverage - 1/M| over M quantile bins."""
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    N = len(y_true)
    if N == 0 or preds.shape[1] < M + 1:
        return float("nan")
    r_m = np.zeros(M)
    target = 1.0 / M
    for m in range(M):
        low = preds[:, m]
        high = preds[:, m + 1]
        in_m = (y_true >= low) & (y_true <= high)
        r_m[m] = np.mean(in_m)
    return float(np.mean(np.abs(r_m - target)))


__all__ = [
    "check_loss_numpy",
    "quantile_loss",
    "trapezoidal_weights_for_quantiles",
    "compute_crps",
    "compute_crps_multi_quantile",
    "compute_picp",
    "compute_coverage",
    "compute_qice",
]

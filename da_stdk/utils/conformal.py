"""CQR conformal calibration (symmetric interval expansion)."""

import math

import numpy as np


def compute_cqr_qhat(cal_preds, cal_y_true, quantile_levels, alpha=0.1):
    """CQR expansion ``qhat`` from calibration scores; returns ``(qhat, n)``."""
    if cal_y_true.ndim > 1:
        cal_y_true = cal_y_true.flatten()
    quantile_levels = np.asarray(quantile_levels)
    q_lo = alpha / 2
    q_hi = 1.0 - alpha / 2
    idx_lo = np.argmin(np.abs(quantile_levels - q_lo))
    idx_hi = np.argmin(np.abs(quantile_levels - q_hi))
    q_lo_pred = cal_preds[:, idx_lo]
    q_hi_pred = cal_preds[:, idx_hi]
    # Standard CQR: score = max(q_lo - y, y - q_hi, 0) so in-interval points get 0
    scores = np.maximum.reduce(
        [q_lo_pred - cal_y_true, cal_y_true - q_hi_pred, np.zeros_like(cal_y_true)]
    )
    n = scores.size
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(k, n)  # if k > n use max score
    qhat = float(np.partition(scores, k - 1)[k - 1]) if k >= 1 else float(np.max(scores))
    qhat = max(qhat, 0.0)  # guarantee nonnegative expansion
    return qhat, n


def compute_conformal_coverage(preds, y_true, quantile_levels, qhat, alpha=0.1):
    """Coverage of [q_lo - qhat, q_hi + qhat]."""
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    quantile_levels = np.asarray(quantile_levels)
    q_lo = alpha / 2
    q_hi = 1.0 - alpha / 2
    idx_lo = np.argmin(np.abs(quantile_levels - q_lo))
    idx_hi = np.argmin(np.abs(quantile_levels - q_hi))
    low = preds[:, idx_lo] - qhat
    high = preds[:, idx_hi] + qhat
    inside = (y_true >= low) & (y_true <= high)
    return float(np.mean(inside))


def _assign_nearest_center(coords, centers):
    """Assign each point to nearest spatial center. Returns cluster_ids (N,)."""
    coords = np.asarray(coords)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    centers = np.asarray(centers)
    # (N, 2) vs (C, 2) -> distances (N, C)
    d = np.sum((coords[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(d, axis=1)


def compute_cluster_aware_cqr(
    cal_preds,
    cal_y_true,
    cal_coords,
    centers,
    quantile_levels,
    alpha=0.1,
    min_n=30,
    global_qhat_fallback=None,
):
    """Per-cluster CQR ``qhat`` with global fallback when cluster size < ``min_n``."""
    if cal_y_true.ndim > 1:
        cal_y_true = cal_y_true.flatten()
    quantile_levels = np.asarray(quantile_levels)
    q_lo, q_hi = alpha / 2, 1.0 - alpha / 2
    idx_lo = np.argmin(np.abs(quantile_levels - q_lo))
    idx_hi = np.argmin(np.abs(quantile_levels - q_hi))
    q_lo_pred = cal_preds[:, idx_lo]
    q_hi_pred = cal_preds[:, idx_hi]
    scores = np.maximum.reduce(
        [q_lo_pred - cal_y_true, cal_y_true - q_hi_pred, np.zeros_like(cal_y_true)]
    )

    cluster_ids = _assign_nearest_center(cal_coords, centers)
    n_global = len(scores)
    k_global = min(int(np.ceil((n_global + 1) * (1.0 - alpha))), n_global)
    global_qhat = (
        float(np.partition(scores, k_global - 1)[k_global - 1])
        if k_global >= 1
        else float(np.max(scores))
    )
    global_qhat = max(global_qhat, 0.0)
    fallback_val = global_qhat_fallback if global_qhat_fallback is not None else global_qhat

    qhat_per_cluster = {}
    num_fallback_clusters = 0
    for c in range(len(centers)):
        mask = cluster_ids == c
        n_c = mask.sum()
        if n_c < min_n:
            qhat_per_cluster[c] = fallback_val
            num_fallback_clusters += 1
        else:
            sc = scores[mask]
            k_c = min(int(np.ceil((n_c + 1) * (1.0 - alpha))), n_c)
            qhat_c = float(np.partition(sc, k_c - 1)[k_c - 1]) if k_c >= 1 else float(np.max(sc))
            qhat_per_cluster[c] = max(qhat_c, 0.0)

    used_qhats = [qhat_per_cluster[c] for c in range(len(centers)) if (cluster_ids == c).any()]
    mean_qhat_cluster = float(np.mean(used_qhats)) if used_qhats else global_qhat

    return qhat_per_cluster, global_qhat, mean_qhat_cluster, num_fallback_clusters


def compute_cluster_conformal_coverage(
    preds,
    y_true,
    coords,
    centers,
    qhat_per_cluster,
    quantile_levels,
    alpha=0.1,
    global_qhat_fallback=0.0,
):
    """Coverage using each point's cluster-specific ``qhat``."""
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    quantile_levels = np.asarray(quantile_levels)
    q_lo, q_hi = alpha / 2, 1.0 - alpha / 2
    idx_lo = np.argmin(np.abs(quantile_levels - q_lo))
    idx_hi = np.argmin(np.abs(quantile_levels - q_hi))

    cluster_ids = _assign_nearest_center(coords, centers)
    qhats = np.array([qhat_per_cluster.get(c, global_qhat_fallback) for c in cluster_ids])
    low = preds[:, idx_lo] - qhats
    high = preds[:, idx_hi] + qhats
    inside = (y_true >= low) & (y_true <= high)
    return float(np.mean(inside))

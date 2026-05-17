"""Non-crossing penalties: P_nc(δ) and prediction-level quantile crossing."""

import torch


def get_crossing_violation_mask(delta_params: list) -> list:
    """Bool mask per quantile k≥2 in violation set A(δ)."""
    if delta_params is None or len(delta_params) < 2:
        return []
    Q = len(delta_params)
    mask = []
    for k in range(1, Q):
        delta_k = delta_params[k]
        delta_k_0 = delta_k[0]
        delta_k_features = delta_k[1:]
        sum_negative = torch.clamp(-delta_k_features, min=0.0).sum()
        mask.append(delta_k_0 < sum_negative)
    return mask


def compute_p_nc_delta_penalty(
    delta_params: list, use_positive_penalty: bool = False
) -> torch.Tensor:
    """Sum J(δ_k) over k≥2; positive form if ``use_positive_penalty``."""
    if delta_params is None or len(delta_params) < 2:
        device = delta_params[0].device if delta_params else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    Q = len(delta_params)
    penalty = torch.tensor(0.0, device=delta_params[0].device)
    for k in range(1, Q):
        delta_k = delta_params[k]
        delta_k_0 = delta_k[0]
        delta_k_features = delta_k[1:]
        sum_negative = torch.clamp(-delta_k_features, min=0.0).sum()
        if use_positive_penalty:
            J_delta_k = torch.clamp(sum_negative - delta_k_0, min=0.0)
        else:
            max_term = torch.max(delta_k_0, sum_negative)
            J_delta_k = delta_k_0 - max_term
        penalty = penalty + J_delta_k
    return penalty


def compute_p_nc_delta_penalty_conditional(
    delta_params: list, use_positive_penalty: bool = True
) -> torch.Tensor:
    """P_nc(δ) summed only over k ∈ A(δ)."""
    if delta_params is None or len(delta_params) < 2:
        device = delta_params[0].device if delta_params else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    Q = len(delta_params)
    device = delta_params[0].device
    penalty = torch.tensor(0.0, device=device)
    violation_mask = get_crossing_violation_mask(delta_params)
    for k in range(1, Q):
        delta_k = delta_params[k]
        delta_k_0 = delta_k[0]
        delta_k_features = delta_k[1:]
        sum_negative = torch.clamp(-delta_k_features, min=0.0).sum()
        if use_positive_penalty:
            J_delta_k = torch.clamp(sum_negative - delta_k_0, min=0.0)
        else:
            max_term = torch.max(delta_k_0, sum_negative)
            J_delta_k = delta_k_0 - max_term
        m = violation_mask[k - 1].detach().float()
        penalty = penalty + m * J_delta_k
    return penalty


def non_crossing_penalty(
    y_pred_multi_q: torch.Tensor, reduction: str = "mean", power: int = 1
) -> torch.Tensor:
    """ReLU penalty when predicted quantiles decrease in τ order."""
    if y_pred_multi_q.dim() != 2 or y_pred_multi_q.shape[1] < 2:
        return torch.tensor(0.0, device=y_pred_multi_q.device)
    diffs = y_pred_multi_q[:, :-1] - y_pred_multi_q[:, 1:]
    violations = torch.relu(diffs)
    if power == 2:
        violations = violations**2
    elif power != 1:
        raise ValueError(f"Unsupported power={power}; use 1 or 2.")
    per_sample = violations.sum(dim=1)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"Unsupported reduction='{reduction}'; use 'mean' or 'sum'.")


__all__ = [
    "get_crossing_violation_mask",
    "compute_p_nc_delta_penalty",
    "compute_p_nc_delta_penalty_conditional",
    "non_crossing_penalty",
]

"""Tests for da_stdk/losses/non_crossing.py."""

import pytest
import torch

from da_stdk.losses.non_crossing import (
    compute_p_nc_delta_penalty,
    compute_p_nc_delta_penalty_conditional,
    get_crossing_violation_mask,
    non_crossing_penalty,
)


def _make_deltas(Q=5, d=4, requires_grad=False):
    """Create Q delta parameter tensors of size (d+1,)."""
    return [torch.randn(d + 1, requires_grad=requires_grad) for _ in range(Q)]


class TestGetCrossingViolationMask:
    def test_returns_q_minus_1_masks(self):
        deltas = _make_deltas(Q=4)
        mask = get_crossing_violation_mask(deltas)
        assert len(mask) == 3

    def test_empty_or_single(self):
        assert get_crossing_violation_mask([]) == []
        assert get_crossing_violation_mask([torch.randn(5)]) == []

    def test_no_violation_when_large_intercept(self):
        Q, d = 3, 4
        deltas = []
        for _ in range(Q):
            dk = torch.zeros(d + 1)
            dk[0] = 100.0  # large intercept, no sum_negative can exceed it
            deltas.append(dk)
        mask = get_crossing_violation_mask(deltas)
        for m in mask:
            assert not m.item()

    def test_violation_when_negative_features(self):
        d = 3
        dk = torch.zeros(d + 1)
        dk[0] = 0.0  # intercept = 0
        dk[1:] = -10.0  # all negative features -> sum_negative = 30 > 0
        deltas = [torch.zeros(d + 1), dk]
        mask = get_crossing_violation_mask(deltas)
        assert mask[0].item()


class TestComputePNcDeltaPenalty:
    def test_returns_scalar(self):
        deltas = _make_deltas()
        p = compute_p_nc_delta_penalty(deltas)
        assert p.shape == ()

    def test_empty_returns_zero(self):
        p = compute_p_nc_delta_penalty([])
        assert p.item() == pytest.approx(0.0)

    def test_single_delta_returns_zero(self):
        p = compute_p_nc_delta_penalty([torch.randn(5)])
        assert p.item() == pytest.approx(0.0)

    def test_use_positive_penalty_nonneg(self):
        deltas = _make_deltas(Q=5)
        p = compute_p_nc_delta_penalty(deltas, use_positive_penalty=True)
        assert p.item() >= 0.0

    def test_no_violation_large_intercept(self):
        Q, d = 4, 3
        deltas = [torch.zeros(d + 1) for _ in range(Q)]
        for dk in deltas:
            dk[0] = 100.0
        p = compute_p_nc_delta_penalty(deltas, use_positive_penalty=True)
        assert p.item() == pytest.approx(0.0)


class TestComputePNcDeltaPenaltyConditional:
    def test_returns_scalar(self):
        deltas = _make_deltas()
        p = compute_p_nc_delta_penalty_conditional(deltas)
        assert p.shape == ()

    def test_empty_returns_zero(self):
        p = compute_p_nc_delta_penalty_conditional([])
        assert p.item() == pytest.approx(0.0)

    def test_single_delta_returns_zero(self):
        p = compute_p_nc_delta_penalty_conditional([torch.randn(5)])
        assert p.item() == pytest.approx(0.0)

    def test_use_positive_penalty_false(self):
        deltas = _make_deltas(Q=4)
        p = compute_p_nc_delta_penalty_conditional(deltas, use_positive_penalty=False)
        assert p.shape == ()


class TestNonCrossingPenalty:
    def test_sorted_input_zero_penalty(self):
        # Already sorted quantiles: no crossing
        preds = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        p = non_crossing_penalty(preds)
        assert p.item() == pytest.approx(0.0)

    def test_crossing_positive_penalty(self):
        preds = torch.tensor([[4.0, 3.0, 2.0, 1.0]])  # fully reversed
        p = non_crossing_penalty(preds)
        assert p.item() > 0.0

    def test_power_2(self):
        preds = torch.tensor([[3.0, 1.0, 2.0]])
        p1 = non_crossing_penalty(preds, power=1)
        p2 = non_crossing_penalty(preds, power=2)
        assert p1.item() >= 0.0
        assert p2.item() >= 0.0

    def test_sum_reduction(self):
        preds = torch.randn(5, 4)
        p_mean = non_crossing_penalty(preds, reduction="mean")
        p_sum = non_crossing_penalty(preds, reduction="sum")
        assert p_sum.item() == pytest.approx(p_mean.item() * 5, abs=1e-5)

    def test_invalid_reduction_raises(self):
        preds = torch.randn(3, 4)
        with pytest.raises(ValueError):
            non_crossing_penalty(preds, reduction="invalid")

    def test_invalid_power_raises(self):
        preds = torch.randn(3, 4)
        with pytest.raises(ValueError):
            non_crossing_penalty(preds, power=3)

    def test_single_quantile_returns_zero(self):
        preds = torch.randn(5, 1)
        p = non_crossing_penalty(preds)
        assert p.item() == pytest.approx(0.0)

    def test_1d_returns_zero(self):
        preds = torch.randn(5)
        p = non_crossing_penalty(preds)
        assert p.item() == pytest.approx(0.0)

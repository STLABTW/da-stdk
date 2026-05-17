"""Tests for da_stdk/utils/ema.py (ModelEMA)."""

import pytest
import torch
import torch.nn as nn

from da_stdk.utils.ema import ModelEMA


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestModelEMA:
    def test_shadow_initialized(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.999)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.allclose(ema.shadow[name], param.data)

    def test_update_changes_shadow(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.9)
        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}
        with torch.no_grad():
            for p in model.parameters():
                p.data.fill_(99.0)
        ema.update(model)
        for name in ema.shadow:
            assert not torch.allclose(ema.shadow[name], old_shadow[name])

    def test_update_formula(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.0)
        with torch.no_grad():
            for p in model.parameters():
                p.data.fill_(7.0)
        ema.update(model)
        for v in ema.shadow.values():
            assert torch.allclose(v, torch.tensor(7.0))

    def test_apply_and_restore(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.9)
        original = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        with torch.no_grad():
            for p in model.parameters():
                p.data.fill_(5.0)
        ema.update(model)
        ema.apply_shadow()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(param.data, ema.shadow[name])
        ema.restore()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(param.data, torch.tensor(5.0))

    def test_state_dict_roundtrip(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.99)
        sd = ema.state_dict()
        ema2 = ModelEMA(model, decay=0.0)
        ema2.load_state_dict(sd)
        assert ema2.decay == 0.99
        for k in ema.shadow:
            assert torch.allclose(ema.shadow[k], ema2.shadow[k])

    def test_assert_missing_param(self):
        model = TinyModel()
        ema = ModelEMA(model, decay=0.9)
        ema.shadow.pop(next(iter(ema.shadow)))
        with pytest.raises(AssertionError):
            ema.update(model)

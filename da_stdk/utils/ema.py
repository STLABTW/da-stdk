"""Exponential moving average of model parameters."""

import torch
import torch.nn as nn


class ModelEMA:
    """Shadow EMA weights; call ``update`` after each optimizer step."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """``decay`` near 1.0 gives slower, smoother averages."""
        self.decay = decay
        self.model = model

        # Create shadow parameters (deep copy of model state)
        self.shadow = {}
        self.backup = {}  # For temporary storage during validation

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Blend current params into shadow (call after ``optimizer.step()``)."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow, f"Parameter {name} not in shadow"
                    new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                    self.shadow[name] = new_average

    def apply_shadow(self):
        """Swap model weights to EMA shadow for eval; pair with ``restore``."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore training weights after ``apply_shadow``."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """EMA state for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        """Restore EMA from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]

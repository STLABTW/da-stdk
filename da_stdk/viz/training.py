"""
Training curves visualization (loss, RMSE, learning rate).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(history, save_path):
    """Plot training curves (train/val loss, val RMSE, learning rate)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("MSE Loss", fontsize=16)
    ax.set_title("Training and Validation Loss", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    if len(history["train_loss"]) > 1:
        train_loss_from_2 = history["train_loss"][1:]
        val_loss_from_2 = history["val_loss"][1:]
        y_min = min(min(train_loss_from_2), min(val_loss_from_2))
        y_max = max(max(train_loss_from_2), max(val_loss_from_2))
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

    # RMSE
    ax = axes[1]
    ax.plot(epochs, history["val_rmse"], "g-", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("RMSE", fontsize=16)
    ax.set_title("Validation RMSE", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)
    if len(history["val_rmse"]) > 1:
        rmse_from_2 = history["val_rmse"][1:]
        y_min = min(rmse_from_2)
        y_max = max(rmse_from_2)
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

    # Learning Rate
    ax = axes[2]
    ax.plot(epochs, history["lr"], "purple", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Learning Rate", fontsize=16)
    ax.set_title("Learning Rate Schedule", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


__all__ = ["plot_training_curves"]

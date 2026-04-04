"""
Training pipeline for the Neural ODE Autoencoder.

Trains on benign-only windows using MSE reconstruction loss.
Validates on benign windows from the validation set.
Supports early stopping and cosine annealing LR schedule.

Usage:
    python src/train.py
    python src/train.py --config configs/default.yaml
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import NeuralODEAutoencoder


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Select best available device (CUDA > MPS > CPU).

    Note: torchdiffeq may have limited MPS support, so we fall back to CPU
    if MPS causes issues.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS can be unstable with torchdiffeq — use CPU as safe default
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch on benign-only windows.

    Returns mean training loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, _ in loader:
        X_batch = X_batch.to(device)

        optimizer.zero_grad()
        x_hat, _ = model(X_batch)
        loss = nn.functional.mse_loss(x_hat, X_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute validation loss on benign windows only.

    Returns mean validation loss (benign-only MSE).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        # Only evaluate on benign windows
        benign_mask = y_batch == 0
        if benign_mask.sum() == 0:
            continue

        X_benign = X_batch[benign_mask].to(device)
        x_hat, _ = model(X_benign)
        loss = nn.functional.mse_loss(x_hat, X_benign)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True  # improved
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False  # did not improve


def train(config: dict):
    """Full training pipeline."""
    # Setup
    device = get_device()
    print(f"Device: {device}")

    torch.manual_seed(config["preprocessing"]["random_state"])
    np.random.seed(config["preprocessing"]["random_state"])

    # Data
    data_dir = config["data"]["processed_dir"]
    batch_size = config["training"]["batch_size"]
    loaders = get_dataloaders(data_dir, batch_size)
    print(f"Train: {len(loaders['train'].dataset):,} windows")
    print(f"Val:   {len(loaders['val'].dataset):,} windows")

    # Model
    model = NeuralODEAutoencoder(config).to(device)

    # Initialize model with a dummy batch to set input dimensions
    sample_x, _ = next(iter(loaders["train"]))
    with torch.no_grad():
        model(sample_x[:2].to(device))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer and scheduler
    train_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    early_stopping = EarlyStopping(patience=train_cfg["early_stopping_patience"])

    # Checkpoints directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    print(f"\nTraining for up to {train_cfg['epochs']} epochs...")
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device)
        val_loss = validate(model, loaders["val"], device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        improved = early_stopping.step(val_loss)
        marker = " *" if improved else ""

        print(
            f"Epoch {epoch:3d}/{train_cfg['epochs']}  "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={lr:.2e}{marker}"
        )

        if improved:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": config,
                },
                os.path.join(checkpoint_dir, "best.pt"),
            )

        if early_stopping.should_stop:
            print(f"\nEarly stopping at epoch {epoch} (patience={early_stopping.patience})")
            break

    print(f"\nBest val loss: {early_stopping.best_loss:.6f}")
    print(f"Checkpoint saved to {checkpoint_dir}/best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train Neural ODE Autoencoder")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()

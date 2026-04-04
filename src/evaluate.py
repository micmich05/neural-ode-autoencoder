"""
Evaluation pipeline for the Neural ODE Autoencoder.

Computes anomaly scores on val/test sets and produces:
- AUROC, AUPRC, F1-Score at optimal threshold
- Score distributions, ROC curve, Precision-Recall curve
- Confusion matrix

Usage:
    python src/evaluate.py
    python src/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.dataset import get_dataloaders
from src.model import NeuralODEAutoencoder


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def compute_anomaly_scores(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-window anomaly scores for all windows in a DataLoader.

    Returns:
        scores: anomaly scores (N,)
        labels: ground truth binary labels (N,)
    """
    model.eval()
    all_scores = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        scores = model.anomaly_score(X_batch)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(y_batch.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Find the threshold that maximizes F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    # Compute F1 for each threshold
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0,
    )
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    best_idx = np.argmax(f1s[:-1])
    return float(thresholds[best_idx])


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Compute all evaluation metrics.

    If threshold is None, the optimal threshold (max F1) is found automatically.
    """
    if threshold is None:
        threshold = find_optimal_threshold(scores, labels)

    predictions = (scores >= threshold).astype(int)

    metrics = {
        "auroc": roc_auc_score(labels, scores),
        "auprc": average_precision_score(labels, scores),
        "f1": f1_score(labels, predictions),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(labels, predictions),
    }
    return metrics


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
    save_path: str | None = None,
):
    """Plot anomaly score distributions for benign vs attack windows."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores[labels == 0], bins=100, alpha=0.6, label="Benign", density=True)
    ax.hist(scores[labels == 1], bins=100, alpha=0.6, label="Attack", density=True)
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Anomaly Score (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: str | None = None,
):
    """Plot ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_precision_recall_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: str | None = None,
):
    """Plot Precision-Recall curve with AUPRC."""
    precisions, recalls, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recalls, precisions, lw=2, label=f"PR (AP = {ap:.4f})")
    baseline = labels.mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Random ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Evaluate Neural ODE Autoencoder")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NeuralODEAutoencoder(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f})")

    # Load data
    loaders = get_dataloaders(config["data"]["processed_dir"], config["training"]["batch_size"])
    loader = loaders[args.split]

    # Compute scores
    print(f"\nComputing anomaly scores on {args.split} set...")
    scores, labels = compute_anomaly_scores(model, loader, device)
    print(f"  Windows: {len(scores):,}  (benign={int((labels==0).sum()):,}, attack={int((labels==1).sum()):,})")

    # Find threshold on val, apply to test
    if args.split == "test":
        print("\nFinding optimal threshold on validation set...")
        val_scores, val_labels = compute_anomaly_scores(model, loaders["val"], device)
        threshold = find_optimal_threshold(val_scores, val_labels)
    else:
        threshold = find_optimal_threshold(scores, labels)

    # Evaluate
    metrics = evaluate(scores, labels, threshold=threshold)

    print(f"\n{'='*40}")
    print(f"Results on {args.split} set")
    print(f"{'='*40}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.6f}")
    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # Plots
    os.makedirs(args.output_dir, exist_ok=True)
    plot_score_distribution(
        scores, labels, threshold,
        save_path=os.path.join(args.output_dir, f"score_dist_{args.split}.png"),
    )
    plot_roc_curve(
        scores, labels,
        save_path=os.path.join(args.output_dir, f"roc_{args.split}.png"),
    )
    plot_precision_recall_curve(
        scores, labels,
        save_path=os.path.join(args.output_dir, f"pr_{args.split}.png"),
    )
    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

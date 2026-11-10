"""
Evaluation module — comprehensive metrics, confusion matrices,
classification reports, and training history visualization.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model import BreadfruitClassifier, MultiTaskLoss
from datasets import get_dataloaders


@torch.no_grad()
def evaluate_model(
    model: BreadfruitClassifier,
    loader: DataLoader,
    config: Config,
) -> Dict:
    """Run full evaluation on a DataLoader and return all metrics.

    Returns a dictionary with:
        - Per-task accuracy, precision, recall, F1
        - Confusion matrices
        - Classification reports (text)
        - Per-class metrics
    """
    device = torch.device(config.device)
    model.eval()

    all_mat_preds: List[int] = []
    all_mat_targets: List[int] = []
    all_qual_preds: List[int] = []
    all_qual_targets: List[int] = []
    all_filenames: List[str] = []

    for images, targets in tqdm(loader, desc="Evaluating"):
        images = images.to(device)

        if isinstance(targets, dict):
            mat_targets = targets["maturity"]
            qual_targets = targets["quality"]
        else:
            mat_targets = targets
            qual_targets = targets

        predictions = model(images)

        mat_preds = predictions["maturity_logits"].argmax(dim=1).cpu()
        qual_preds = predictions["quality_logits"].argmax(dim=1).cpu()

        all_mat_preds.extend(mat_preds.tolist())
        all_mat_targets.extend(mat_targets.tolist())
        all_qual_preds.extend(qual_preds.tolist())
        all_qual_targets.extend(qual_targets.tolist())

    results = _compute_metrics(
        all_mat_targets, all_mat_preds,
        all_qual_targets, all_qual_preds,
        config.model.maturity_classes,
        config.model.quality_classes,
    )

    return results


def _compute_metrics(
    mat_targets: List[int],
    mat_preds: List[int],
    qual_targets: List[int],
    qual_preds: List[int],
    maturity_classes: List[str],
    quality_classes: List[str],
) -> Dict:
    """Compute all classification metrics for both tasks."""

    def _safe_metric(y_true, y_pred, metric_fn, average="weighted", **kwargs):
        if len(set(y_true)) < 2:
            return 0.0
        return metric_fn(y_true, y_pred, average=average, **kwargs)

    results = {}

    for task_name, targets, preds, classes in [
        ("maturity", mat_targets, mat_preds, maturity_classes),
        ("quality", qual_targets, qual_preds, quality_classes),
    ]:
        task = {}
        task["accuracy"] = accuracy_score(targets, preds)
        task["precision"] = _safe_metric(targets, preds, precision_score, average="weighted", zero_division=0)
        task["recall"] = _safe_metric(targets, preds, recall_score, average="weighted", zero_division=0)
        task["f1_score"] = _safe_metric(targets, preds, f1_score, average="weighted", zero_division=0)

        task["precision_per_class"] = _safe_metric(targets, preds, precision_score, average=None, zero_division=0, labels=range(len(classes))).tolist() if len(classes) > 1 else [0.0]
        task["recall_per_class"] = _safe_metric(targets, preds, recall_score, average=None, zero_division=0, labels=range(len(classes))).tolist() if len(classes) > 1 else [0.0]
        task["f1_per_class"] = _safe_metric(targets, preds, f1_score, average=None, zero_division=0, labels=range(len(classes))).tolist() if len(classes) > 1 else [0.0]

        task["confusion_matrix"] = confusion_matrix(targets, preds, labels=range(len(classes))).tolist()
        task["classification_report"] = classification_report(
            targets, preds,
            target_names=classes,
            zero_division=0,
            digits=4,
        )

        task["class_names"] = classes
        results[task_name] = task

    results["overall_avg_accuracy"] = (results["maturity"]["accuracy"] + results["quality"]["accuracy"]) / 2.0
    results["overall_avg_f1"] = (results["maturity"]["f1_score"] + results["quality"]["f1_score"]) / 2.0

    return results


def plot_confusion_matrices(
    results: Dict,
    save_dir: str = "plots",
    prefix: str = "",
) -> None:
    """Plot and save confusion matrices for both maturity and quality tasks."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for task_name in ["maturity", "quality"]:
        task = results[task_name]
        cm = np.array(task["confusion_matrix"])
        classes = task["class_names"]

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
        ax.set_title(f"{task_name.capitalize()} - Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        plt.tight_layout()

        fname = f"{prefix}_{task_name}_confusion_matrix.png" if prefix else f"{task_name}_confusion_matrix.png"
        fig.savefig(save_path / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path / fname}")


def plot_training_history(history: Dict, save_dir: str = "plots", prefix: str = "") -> None:
    """Plot training/validation loss and accuracy curves."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "val_maturity_accuracy" in history and history["val_maturity_accuracy"]:
        ax.plot(epochs, history["val_maturity_accuracy"], "g-", label="Maturity Accuracy", linewidth=2)
    if "val_quality_accuracy" in history and history["val_quality_accuracy"]:
        ax.plot(epochs, history["val_quality_accuracy"], "m-", label="Quality Accuracy", linewidth=2)
    if "val_avg_accuracy" in history and history["val_avg_accuracy"]:
        ax.plot(epochs, history["val_avg_accuracy"], "k--", label="Avg Accuracy", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Validation Accuracy Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{prefix}_training_history.png" if prefix else "training_history.png"
    fig.savefig(save_path / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path / fname}")

    # Learning rate curve
    if "learning_rate" in history and history["learning_rate"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, history["learning_rate"], "c-", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"{prefix}_learning_rate.png" if prefix else "learning_rate.png"
        fig.savefig(save_path / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path / fname}")

    # Per-task loss curves
    if "train_maturity_loss" in history and history["train_maturity_loss"]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        ax = axes[0]
        ax.plot(epochs, history["train_maturity_loss"], "g-", label="Train Maturity Loss", linewidth=2)
        ax.plot(epochs, history["val_maturity_loss"], "g--", label="Val Maturity Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Maturity Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(epochs, history["train_quality_loss"], "m-", label="Train Quality Loss", linewidth=2)
        ax.plot(epochs, history["val_quality_loss"], "m--", label="Val Quality Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Quality Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"{prefix}_per_task_loss.png" if prefix else "per_task_loss.png"
        fig.savefig(save_path / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path / fname}")


def print_evaluation_report(results: Dict) -> None:
    """Print a structured evaluation report to console."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  COMPREHENSIVE EVALUATION REPORT")
    print(f"{sep}\n")

    for task_name in ["maturity", "quality"]:
        task = results[task_name]
        print(f"  {'─' * 60}")
        print(f"  Task: {task_name.upper()}")
        print(f"  {'─' * 60}")
        print(f"    Accuracy :  {task['accuracy']:.4f}")
        print(f"    Precision:  {task['precision']:.4f}")
        print(f"    Recall   :  {task['recall']:.4f}")
        print(f"    F1-Score :  {task['f1_score']:.4f}")
        print()
        print(f"    Per-Class Metrics:")
        for i, cls in enumerate(task["class_names"]):
            print(f"      {cls:20s}  Prec={task['precision_per_class'][i]:.4f}  "
                  f"Recall={task['recall_per_class'][i]:.4f}  "
                  f"F1={task['f1_per_class'][i]:.4f}")
        print()
        print(f"    Classification Report:")
        for line in task["classification_report"].split("\n"):
            print(f"      {line}")
        print()

    print(f"  {'─' * 60}")
    print(f"  Overall Average Accuracy:  {results['overall_avg_accuracy']:.4f}")
    print(f"  Overall Average F1-Score:  {results['overall_avg_f1']:.4f}")
    print(f"{sep}")

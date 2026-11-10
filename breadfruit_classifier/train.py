"""
Training module — full training pipeline with early stopping,
learning rate scheduling, checkpointing, and progress logging.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model import BreadfruitClassifier, MultiTaskLoss, create_model
from datasets import get_dataloaders


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_epoch(
    model: BreadfruitClassifier,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: Optimizer,
    device: torch.device,
    config: Config,
) -> Dict[str, float]:
    """Run one training epoch over the entire dataset."""
    model.train()
    total_loss = 0.0
    maturity_loss_total = 0.0
    quality_loss_total = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        if isinstance(targets, dict):
            targets = {k: v.to(device) for k, v in targets.items()}
        else:
            targets = {"maturity": targets.to(device), "quality": targets.to(device)}

        optimizer.zero_grad()

        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss_dict["total_loss"].item()
        maturity_loss_total += loss_dict["maturity_loss"].item()
        quality_loss_total += loss_dict["quality_loss"].item()

        if batch_idx % config.logging.log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss'].item():.4f}",
                "mat": f"{loss_dict['maturity_loss'].item():.4f}",
                "qual": f"{loss_dict['quality_loss'].item():.4f}",
            })

    return {
        "loss": total_loss / num_batches,
        "maturity_loss": maturity_loss_total / num_batches,
        "quality_loss": quality_loss_total / num_batches,
    }


@torch.no_grad()
def validate(
    model: BreadfruitClassifier,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation and return loss and accuracy metrics."""
    model.eval()
    total_loss = 0.0
    maturity_loss_total = 0.0
    quality_loss_total = 0.0
    maturity_correct = 0
    quality_correct = 0
    total_samples = 0

    for images, targets in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)

        if isinstance(targets, dict):
            targets = {k: v.to(device) for k, v in targets.items()}
        else:
            targets = {"maturity": targets.to(device), "quality": targets.to(device)}

        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)

        total_loss += loss_dict["total_loss"].item()
        maturity_loss_total += loss_dict["maturity_loss"].item()
        quality_loss_total += loss_dict["quality_loss"].item()

        mat_preds = predictions["maturity_logits"].argmax(dim=1)
        qual_preds = predictions["quality_logits"].argmax(dim=1)

        maturity_correct += (mat_preds == targets["maturity"]).sum().item()
        quality_correct += (qual_preds == targets["quality"]).sum().item()
        total_samples += images.size(0)

    num_batches = len(loader)
    return {
        "loss": total_loss / num_batches,
        "maturity_loss": maturity_loss_total / num_batches,
        "quality_loss": quality_loss_total / num_batches,
        "maturity_accuracy": maturity_correct / total_samples,
        "quality_accuracy": quality_correct / total_samples,
        "avg_accuracy": (maturity_correct + quality_correct) / (2 * total_samples),
    }


def train_model(
    config: Config,
    model: Optional[BreadfruitClassifier] = None,
) -> Tuple[BreadfruitClassifier, Dict[str, List[float]]]:
    """End-to-end training routine.

    Args:
        config: Complete configuration object.
        model: Optional pre-initialized model. Creates new one if None.

    Returns:
        Trained model and training history dictionary.
    """
    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Training config: {config.training}")

    train_loader, val_loader, _ = get_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    if model is None:
        model = create_model(config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if model.has_multi_task:
        criterion = MultiTaskLoss(
            maturity_weight=config.training.loss_weights[0],
            quality_weight=config.training.loss_weights[1],
        )
    else:
        criterion = MultiTaskLoss(maturity_weight=1.0, quality_weight=0.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=config.training.lr_scheduler_patience,
        factor=config.training.lr_scheduler_factor,
        min_lr=config.training.min_lr,
    )

    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode="max",
    )

    checkpoint_dir = Path(config.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_maturity_loss": [],
        "train_quality_loss": [],
        "val_maturity_loss": [],
        "val_quality_loss": [],
        "val_maturity_accuracy": [],
        "val_quality_accuracy": [],
        "val_avg_accuracy": [],
        "learning_rate": [],
    }

    best_val_acc = 0.0
    start_time = time.time()

    # Phase 1: Freeze backbone, train only heads
    if config.model.freeze_backbone_epochs > 0:
        print(f"\nPhase 1: Freezing backbone for {config.model.freeze_backbone_epochs} epochs")
        model.freeze_backbone(True)

        # Only head params are optimized
        head_params = list(model.maturity_head.parameters())
        if model.has_multi_task:
            head_params += list(model.quality_head.parameters())
        optimizer = torch.optim.AdamW(
            head_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=3,
            factor=config.training.lr_scheduler_factor,
            min_lr=config.training.min_lr,
        )

        for epoch in range(1, config.model.freeze_backbone_epochs + 1):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config)
            val_metrics = validate(model, val_loader, criterion, device)

            val_acc = val_metrics["avg_accuracy"] if model.has_multi_task else val_metrics["maturity_accuracy"]
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_maturity_loss"].append(train_metrics["maturity_loss"])
            history["train_quality_loss"].append(train_metrics["quality_loss"])
            history["val_maturity_loss"].append(val_metrics["maturity_loss"])
            history["val_quality_loss"].append(val_metrics["quality_loss"])
            history["val_maturity_accuracy"].append(val_metrics["maturity_accuracy"])
            history["val_quality_accuracy"].append(val_metrics["quality_accuracy"])
            history["val_avg_accuracy"].append(val_metrics.get("avg_accuracy", 0.0))
            history["learning_rate"].append(current_lr)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                _save_checkpoint(model, optimizer, epoch, val_acc, config, is_best=True)

            print(f"Phase1  Epoch {epoch:2d}/{config.model.freeze_backbone_epochs}  "
                  f"Train Loss: {train_metrics['loss']:.4f}  "
                  f"Val Loss: {val_metrics['loss']:.4f}  "
                  f"Val Acc: {val_acc:.4f}  "
                  f"LR: {current_lr:.2e}")

        print("Phase 2: Unfreezing backbone for joint fine-tuning")
        model.freeze_backbone(False)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate * 0.1,
            weight_decay=config.training.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max",
            patience=config.training.lr_scheduler_patience,
            factor=config.training.lr_scheduler_factor,
            min_lr=config.training.min_lr,
        )

    # Phase 2 (or main): Full fine-tuning
    start_epoch = max(1, config.model.freeze_backbone_epochs + 1)
    total_epochs = config.training.epochs

    for epoch in range(start_epoch, total_epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config)
        val_metrics = validate(model, val_loader, criterion, device)

        val_acc = val_metrics["avg_accuracy"] if model.has_multi_task else val_metrics["maturity_accuracy"]
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_maturity_loss"].append(train_metrics["maturity_loss"])
        history["train_quality_loss"].append(train_metrics["quality_loss"])
        history["val_maturity_loss"].append(val_metrics["maturity_loss"])
        history["val_quality_loss"].append(val_metrics["quality_loss"])
        history["val_maturity_accuracy"].append(val_metrics["maturity_accuracy"])
        history["val_quality_accuracy"].append(val_metrics["quality_accuracy"])
        history["val_avg_accuracy"].append(val_metrics.get("avg_accuracy", 0.0))
        history["learning_rate"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(model, optimizer, epoch, val_acc, config, is_best=True)

        print(f"Epoch {epoch:2d}/{total_epochs}  "
              f"Train Loss: {train_metrics['loss']:.4f}  "
              f"Val Loss: {val_metrics['loss']:.4f}  "
              f"Val Maturity Acc: {val_metrics['maturity_accuracy']:.4f}  "
              f"Val Quality Acc: {val_metrics['quality_accuracy']:.4f}  "
              f"LR: {current_lr:.2e}")

        if early_stopping(val_acc, epoch):
            print(f"Early stopping triggered at epoch {epoch}. "
                  f"Best epoch: {early_stopping.best_epoch} (val_acc={best_val_acc:.4f})")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = early_stopping.best_epoch
    history["total_time_min"] = total_time / 60

    # Save training history as JSON
    json_path = checkpoint_dir / "training_history.json"
    serializable = {k: v for k, v in history.items() if k != "best_val_acc" and k != "best_epoch" and k != "total_time_min"}
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"Training history saved to {json_path}")

    return model, history


def _save_checkpoint(
    model: BreadfruitClassifier,
    optimizer: Optimizer,
    epoch: int,
    val_acc: float,
    config: Config,
    is_best: bool = False,
) -> str:
    """Save model checkpoint to disk."""
    checkpoint_dir = Path(config.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "config": {
            "maturity_classes": config.model.maturity_classes,
            "quality_classes": config.model.quality_classes,
            "backbone": config.model.backbone,
            "image_size": config.data.image_size,
            "mean": config.data.mean,
            "std": config.data.std,
        },
    }

    if is_best:
        fname = "best_model.pth"
    else:
        fname = f"checkpoint_epoch_{epoch:03d}.pth"
    path = checkpoint_dir / fname
    torch.save(checkpoint, path)
    print(f"  {'[Best]' if is_best else ''} Checkpoint saved: {path}")

    return str(path)


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[BreadfruitClassifier] = None,
    config: Optional[Config] = None,
    map_location: Optional[str] = None,
) -> Tuple[BreadfruitClassifier, dict]:
    """Load a saved checkpoint and return model + checkpoint dictionary."""
    if map_location is None and config is not None:
        map_location = config.device

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if model is None:
        if config is None:
            raise ValueError("Either model or config must be provided to load checkpoint")
        model = create_model(config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(map_location or "cpu")

    return model, checkpoint

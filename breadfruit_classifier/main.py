"""
Main entry point for breadfruit maturity and quality classification.
Supports training, evaluation, and inference modes.

Usage:
    python main.py train                                    # Train a new model
    python main.py evaluate --checkpoint checkpoints/best_model.pth   # Evaluate on test set
    python main.py predict --image path/to/image.jpg        # Predict single image
    python main.py predict --folder path/to/images/         # Predict folder of images
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import Config
from datasets import get_dataloaders
from model import create_model, count_parameters
from train import train_model, load_checkpoint
from evaluate import (
    evaluate_model,
    plot_confusion_matrices,
    plot_training_history,
    print_evaluation_report,
)
from predict import (
    load_model_for_inference,
    predict_single,
    predict_folder,
    visualize_prediction,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def create_output_dirs(config: Config) -> None:
    """Ensure all output directories exist."""
    for dir_path in [config.logging.checkpoint_dir, config.logging.log_dir, config.logging.plot_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def cmd_train(config: Config, args: argparse.Namespace) -> None:
    """Train a new breadfruit classifier model."""
    print("=" * 70)
    print("  BREADFRUIT MATURITY & QUALITY CLASSIFIER — TRAINING")
    print("=" * 70)

    create_output_dirs(config)
    set_seed(config.training.seed)

    print(f"\nBackbone: {config.model.backbone} (pretrained={config.model.pretrained})")
    print(f"Maturity classes: {config.model.maturity_classes}")
    print(f"Quality classes:  {config.model.quality_classes}")
    print(f"Device: {config.device}")
    print(f"Data root: {config.data.data_root}\n")

    model, history = train_model(config)

    print(f"\nLoading best checkpoint for evaluation...")
    checkpoint_path = Path(config.logging.checkpoint_dir) / "best_model.pth"
    if checkpoint_path.exists():
        model, _ = load_checkpoint(str(checkpoint_path), config=config)
    else:
        print("Warning: best_model.pth not found. Using final model state.")

    plot_training_history(history, save_dir=config.logging.plot_dir, prefix=config.logging.experiment_name)

    _, _, test_loader = get_dataloaders(config)
    if test_loader is not None and len(test_loader.dataset) > 0:
        print(f"\n{'=' * 70}")
        print("  TEST SET EVALUATION")
        print(f"{'=' * 70}")
        results = evaluate_model(model, test_loader, config)
        print_evaluation_report(results)
        plot_confusion_matrices(results, save_dir=config.logging.plot_dir, prefix=config.logging.experiment_name)

        results_path = Path(config.logging.log_dir) / f"{config.logging.experiment_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {results_path}")

    print("\nTraining pipeline completed successfully.")


def cmd_evaluate(config: Config, args: argparse.Namespace) -> None:
    """Evaluate a trained model on the test set."""
    create_output_dirs(config)
    set_seed(config.training.seed)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(config.logging.checkpoint_dir) / "best_model.pth"
        if not checkpoint_path.exists():
            print("Error: No checkpoint specified and best_model.pth not found.")
            sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model_for_inference(str(checkpoint_path), config)

    _, _, test_loader = get_dataloaders(config)
    if test_loader is None or len(test_loader.dataset) == 0:
        print("Error: No test data found.")
        sys.exit(1)

    print(f"Test samples: {len(test_loader.dataset)}")
    results = evaluate_model(model, test_loader, config)
    print_evaluation_report(results)
    plot_confusion_matrices(results, save_dir=config.logging.plot_dir, prefix=f"test_{config.logging.experiment_name}")

    results_path = Path(config.logging.log_dir) / f"test_{config.logging.experiment_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")


def cmd_predict(config: Config, args: argparse.Namespace) -> None:
    """Run inference on images using a trained model."""
    create_output_dirs(config)
    set_seed(config.training.seed)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(config.logging.checkpoint_dir) / "best_model.pth"
        if not checkpoint_path.exists():
            print("Error: No checkpoint specified and best_model.pth not found.")
            sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model_for_inference(str(checkpoint_path), config)

    if args.image:
        print(f"\nPredicting: {args.image}")
        result = predict_single(model, args.image, config, return_probs=True)
        print(f"  Maturity: {result['maturity']['prediction']} "
              f"(confidence: {result['maturity']['confidence']:.4f})")
        print(f"  Quality:  {result['quality']['prediction']} "
              f"(confidence: {result['quality']['confidence']:.4f})")

        if args.probabilities:
            print(f"\n  Maturity probabilities:")
            for cls, prob in result["maturity"]["probabilities"].items():
                print(f"    {cls:20s}: {prob:.4f}")
            print(f"  Quality probabilities:")
            for cls, prob in result["quality"]["probabilities"].items():
                print(f"    {cls:20s}: {prob:.4f}")

        if args.visualize:
            visualize_prediction(model, args.image, config)

    if args.folder:
        print(f"\nPredicting all images in: {args.folder}")
        results = predict_folder(
            model, args.folder, config,
            output_csv=args.output_csv,
            return_probs=args.probabilities,
        )
        print(f"\nProcessed {len(results)} images.")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Breadfruit Maturity & Quality Classifier — Training, Evaluation, and Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train
  python main.py evaluate --checkpoint checkpoints/best_model.pth
  python main.py predict --image path/to/breadfruit.jpg
  python main.py predict --folder path/to/test_images/ --output_csv results.csv
  python main.py predict --image test.jpg --visualize --probabilities
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (overrides default config values)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides config)",
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (overrides config)",
    )
    train_parser.add_argument(
        "--lr", type=float, default=None,
        help="Initial learning rate (overrides config)",
    )
    train_parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment name for logging (overrides config)",
    )
    train_parser.add_argument(
        "--data_root", type=str, default=None,
        help="Path to dataset root (overrides config)",
    )
    train_parser.add_argument(
        "--annotation_file", type=str, default=None,
        help="Path to CSV annotations file",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pth)",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference on images")
    predict_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pth)",
    )
    predict_parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image for prediction",
    )
    predict_parser.add_argument(
        "--folder", type=str, default=None,
        help="Path to folder of images for batch prediction",
    )
    predict_parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Save batch predictions to CSV file",
    )
    predict_parser.add_argument(
        "--visualize", action="store_true",
        help="Generate prediction visualization with Grad-CAM",
    )
    predict_parser.add_argument(
        "--probabilities", action="store_true",
        help="Show class probabilities",
    )

    return parser


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """Override config values with command-line arguments."""
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.experiment is not None:
        config.logging.experiment_name = args.experiment
    if args.data_root is not None:
        config.data.data_root = args.data_root
    if args.annotation_file is not None:
        config.data.annotation_file = args.annotation_file
    return config


def load_config_from_file(config_path: str) -> Config:
    """Load config from JSON file, merging with defaults."""
    config = Config()
    with open(config_path, "r") as f:
        overrides = json.load(f)

    for section, values in overrides.items():
        if hasattr(config, section):
            section_obj = getattr(config, section)
            for key, val in values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, val)
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = Config()

    if args.command in ("train", "evaluate", "predict"):
        config = update_config_from_args(config, args)

    if args.command == "train":
        cmd_train(config, args)
    elif args.command == "evaluate":
        cmd_evaluate(config, args)
    elif args.command == "predict":
        cmd_predict(config, args)


if __name__ == "__main__":
    main()

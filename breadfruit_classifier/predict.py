"""
Inference module — load a trained model and make predictions on new images.
Supports single image, folder of images, and visualization of results.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import Config
from model import BreadfruitClassifier
from datasets import get_inference_transform
from train import load_checkpoint
from gradcam import GradCAM, visualize_gradcam


def load_model_for_inference(
    checkpoint_path: str,
    config: Config,
    map_location: Optional[str] = None,
) -> BreadfruitClassifier:
    """Load a trained model from a checkpoint file for inference."""
    if map_location is None:
        map_location = config.device
    model, checkpoint = load_checkpoint(checkpoint_path, config=config, map_location=map_location)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Trained for {checkpoint.get('epoch', '?')} epochs")
    print(f"  Validation accuracy: {checkpoint.get('val_acc', '?'):.4f}")
    return model


def predict_single(
    model: BreadfruitClassifier,
    image: Union[str, Path, Image.Image],
    config: Config,
    return_probs: bool = False,
    generate_gradcam: bool = False,
) -> Dict:
    """Predict maturity and quality for a single image.

    Args:
        model: Trained BreadfruitClassifier model.
        image: Path to image file or PIL Image.
        config: Configuration object.
        return_probs: If True, include class probabilities.
        generate_gradcam: If True, include Grad-CAM heatmaps.

    Returns:
        Dictionary with predictions, confidence scores, probabilities (optional),
        and optional Grad-CAM heatmaps.
    """
    model.eval()
    device = next(model.parameters()).device

    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    transform = get_inference_transform(config)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Predict without grad for class predictions
    with torch.no_grad():
        predictions = model(input_tensor)

    results = {}

    for task_name, logits_key in [("maturity", "maturity_logits"), ("quality", "quality_logits")]:
        logits = predictions[logits_key]
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

        class_names = (
            config.model.maturity_classes
            if task_name == "maturity"
            else config.model.quality_classes
        )

        task_result = {
            "prediction": class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}",
            "class_id": pred_idx,
            "confidence": round(confidence, 4),
        }

        if return_probs:
            task_result["probabilities"] = {
                class_names[i] if i < len(class_names) else f"class_{i}": round(probs[0, i].item(), 4)
                for i in range(len(class_names))
            }

        if generate_gradcam:
            gradcam = GradCAM(model, target_layer=config.gradcam.target_layer)
            heatmap = gradcam.generate(input_tensor, class_idx=None, task=task_name)
            task_result["grad_cam"] = heatmap.tolist()

        results[task_name] = task_result

    return results


def predict_folder(
    model: BreadfruitClassifier,
    folder_path: str,
    config: Config,
    output_csv: Optional[str] = None,
    return_probs: bool = False,
    generate_gradcam: bool = False,
) -> List[Dict]:
    """Predict for all images in a folder.

    Args:
        model: Trained model.
        folder_path: Path to folder containing images.
        config: Configuration object.
        output_csv: Optional path to save predictions as CSV.
        return_probs: Include class probabilities in output.
        generate_gradcam: If True, include Grad-CAM heatmaps per image.

    Returns:
        List of prediction dictionaries per image.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in image_extensions]
    )

    if not image_files:
        print(f"No image files found in {folder_path}")
        return []

    print(f"Found {len(image_files)} images in {folder_path}")
    all_results = []

    for img_path in image_files:
        result = predict_single(model, img_path, config, return_probs=return_probs, generate_gradcam=generate_gradcam)
        result["filename"] = img_path.name
        all_results.append(result)

        print(f"  {img_path.name:40s}  "
              f"Maturity: {result['maturity']['prediction']:20s} "
              f"(conf={result['maturity']['confidence']:.4f})  "
              f"Quality: {result['quality']['prediction']:15s} "
              f"(conf={result['quality']['confidence']:.4f})")

    # Save to CSV
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "filename",
                "maturity_prediction", "maturity_confidence", "maturity_class_id",
                "quality_prediction", "quality_confidence", "quality_class_id",
            ]
            if return_probs:
                fieldnames.append("probabilities")
            if generate_gradcam:
                fieldnames.append("maturity_grad_cam")
                fieldnames.append("quality_grad_cam")

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in all_results:
                row = {
                    "filename": res["filename"],
                    "maturity_prediction": res["maturity"]["prediction"],
                    "maturity_confidence": res["maturity"]["confidence"],
                    "maturity_class_id": res["maturity"]["class_id"],
                    "quality_prediction": res["quality"]["prediction"],
                    "quality_confidence": res["quality"]["confidence"],
                    "quality_class_id": res["quality"]["class_id"],
                }
                if return_probs:
                    row["probabilities"] = json.dumps({
                        "maturity": res["maturity"].get("probabilities", {}),
                        "quality": res["quality"].get("probabilities", {}),
                    })
                if generate_gradcam:
                    row["maturity_grad_cam"] = json.dumps(res["maturity"].get("grad_cam", []))
                    row["quality_grad_cam"] = json.dumps(res["quality"].get("grad_cam", []))
                writer.writerow(row)
        print(f"Predictions saved to: {csv_path}")

    return all_results


def visualize_prediction(
    model: BreadfruitClassifier,
    image_path: str,
    config: Config,
    save_dir: str = "plots/predictions",
    generate_gradcam: bool = True,
) -> None:
    """Generate a full visualization: original image + predictions + Grad-CAM.

    Saves a composite figure to disk.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fname = Path(image_path).stem

    pil_image = Image.open(image_path).convert("RGB")
    result = predict_single(model, pil_image, config, return_probs=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(np.array(pil_image))
    axes[0].set_title("Input Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    info_text = (
        f"Maturity: {result['maturity']['prediction']}\n"
        f"  Confidence: {result['maturity']['confidence']:.4f}\n\n"
        f"Quality: {result['quality']['prediction']}\n"
        f"  Confidence: {result['quality']['confidence']:.4f}"
    )
    axes[1].imshow(np.array(pil_image))
    axes[1].axis("off")
    axes[1].set_title("Predictions", fontsize=14, fontweight="bold")

    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85)
    axes[1].text(
        0.5, 0.5, info_text,
        transform=axes[1].transAxes,
        fontsize=12, ha="center", va="center",
        bbox=props, fontfamily="monospace",
    )

    plt.tight_layout()
    out_path = save_path / f"{fname}_prediction.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction visualization saved: {out_path}")

    if generate_gradcam:
        for task in ["maturity", "quality"]:
            try:
                visualize_gradcam(model, image_path, config, save_dir=str(save_path), task=task)
            except Exception as e:
                print(f"Grad-CAM failed for task '{task}': {e}")

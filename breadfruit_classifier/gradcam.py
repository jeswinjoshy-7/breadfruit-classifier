"""
Grad-CAM visualization module for model interpretability.
Generates class activation heatmaps to explain model predictions.
"""

from pathlib import Path
from typing import Optional, Tuple, Union, List

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import Config
from model import BreadfruitClassifier
from datasets import get_inference_transform


class GradCAM:
    """Grad-CAM implementation for EfficientNet-B0.

    Produces a coarse localization heatmap highlighting regions
    the model focuses on for a particular class prediction.

    Attributes:
        model: The trained BreadfruitClassifier model.
        target_layer: Name of the convolutional layer to hook into.
        gradient: Hooked gradient values.
        activation: Hooked activation values.
        handles: List of hook handles for cleanup.
    """

    def __init__(
        self,
        model: BreadfruitClassifier,
        target_layer: str = "features.8",
    ):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradient: Optional[torch.Tensor] = None
        self.activation: Optional[torch.Tensor] = None
        self.handles: List = []

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activation = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()

        target_module = self._get_layer(self.target_layer)
        if target_module is None:
            raise ValueError(f"Layer '{self.target_layer}' not found in model")

        self.handles.append(target_module.register_forward_hook(forward_hook))
        self.handles.append(target_module.register_full_backward_hook(backward_hook))

    def _get_layer(self, layer_name: str):
        """Retrieve a layer by dot-separated name."""
        parts = layer_name.split(".")
        module = self.model.backbone
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
            if module is None:
                return None
        return module

    def _find_last_conv(self) -> torch.nn.Module:
        """Find the last convolutional layer in the backbone as fallback."""
        last_conv = None
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No convolutional layer found in backbone")
        return last_conv

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        task: str = "maturity",
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W).
            class_idx: Target class index. If None, uses argmax.
            task: Which head to use ("maturity" or "quality").

        Returns:
            Heatmap as a 2D numpy array (H, W) in range [0, 1].
        """
        if self.activation is not None:
            self.activation = None
        if self.gradient is not None:
            self.gradient = None

        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor.requires_grad_(True)

        predictions = self.model(input_tensor)

        logits_key = f"{task}_logits"
        if logits_key not in predictions:
            raise ValueError(f"Task '{task}' not found. Available: {list(predictions.keys())}")

        logits = predictions[logits_key]
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradient is None or self.activation is None:
            raise RuntimeError("Gradient or activation not captured. Ensure forward/backward passed through target layer.")

        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = np.maximum(cam, 0)

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        image: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay heatmap on the original image."""
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlayed = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
        return overlayed

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def visualize_gradcam(
    model: BreadfruitClassifier,
    image_path: str,
    config: Config,
    save_dir: str = "plots/gradcam",
    task: str = "maturity",
    class_idx: Optional[int] = None,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full Grad-CAM visualization pipeline for a single image.

    Args:
        model: Trained BreadfruitClassifier model.
        image_path: Path to the input image.
        config: Configuration object.
        save_dir: Directory to save visualizations.
        task: Which task head to use ("maturity" or "quality").
        class_idx: Target class index (None = argmax).
        alpha: Heatmap overlay opacity.

    Returns:
        Tuple of (original_image, heatmap, overlayed_image) as numpy arrays.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    pil_image = Image.open(image_path).convert("RGB")
    original_np = np.array(pil_image)
    transform = get_inference_transform(config)
    input_tensor = transform(pil_image).unsqueeze(0)

    gradcam = GradCAM(model, target_layer=config.gradcam.target_layer)
    heatmap = gradcam.generate(input_tensor, class_idx=class_idx, task=task)

    h, w = original_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    rgb_image = original_np
    if original_np.ndim == 2:
        rgb_image = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
    elif original_np.shape[2] == 4:
        rgb_image = cv2.cvtColor(original_np, cv2.COLOR_RGBA2RGB)

    overlayed = gradcam.overlay_heatmap(heatmap_resized, rgb_image, alpha=alpha)
    gradcam.remove_hooks()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap_resized, cmap="jet", interpolation="bilinear")
    axes[1].set_title(f"Grad-CAM Heatmap ({task})", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlayed)
    axes[2].set_title(f"Overlay (α={alpha})", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    fname = f"gradcam_{Path(image_path).stem}_{task}.png"
    fig.savefig(save_path / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM saved to: {save_path / fname}")

    return rgb_image, heatmap_resized, overlayed


def generate_gradcam_batch(
    model: BreadfruitClassifier,
    image_paths: List[str],
    config: Config,
    save_dir: str = "plots/gradcam",
    task: str = "maturity",
) -> None:
    """Generate Grad-CAM visualizations for multiple images."""
    for img_path in image_paths:
        try:
            visualize_gradcam(model, img_path, config, save_dir, task=task)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

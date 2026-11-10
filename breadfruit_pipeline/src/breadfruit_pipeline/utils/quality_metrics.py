"""
Image Quality Metrics
Calculates various quality metrics for image assessment.
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def calculate_image_quality(image: np.ndarray) -> Dict[str, float]:
    """Calculate all quality metrics for an image."""
    metrics = {}
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Blur score (Laplacian variance)
    metrics["blur_score"] = calculate_blur_score(gray)
    
    # Sharpness (Tenengrad)
    metrics["sharpness_score"] = calculate_sharpness(gray)
    
    # Brightness (mean luminance)
    metrics["brightness_score"] = calculate_brightness(image)
    
    # Contrast (RMS contrast)
    metrics["contrast_score"] = calculate_contrast(gray)
    
    # Entropy
    metrics["entropy"] = calculate_entropy(gray)
    
    # Noise estimation
    metrics["noise_estimation"] = estimate_noise(gray)
    
    # Resolution
    metrics["resolution"] = (image.shape[1], image.shape[0])
    
    # Aspect ratio
    metrics["aspect_ratio"] = image.shape[1] / image.shape[0] if image.shape[0] > 0 else 0
    
    return metrics


def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate blur score using Laplacian variance."""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()


def calculate_sharpness(image: np.ndarray) -> float:
    """Calculate sharpness using Tenengrad (Sobel gradient magnitude)."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(gradient_magnitude)


def calculate_brightness(image: np.ndarray) -> float:
    """Calculate average brightness (luminance)."""
    if len(image.shape) == 3:
        # Convert to luminance (Y channel)
        y = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        return np.mean(y)
    return np.mean(image)


def calculate_contrast(image: np.ndarray) -> float:
    """Calculate RMS contrast."""
    return np.std(image)


def calculate_entropy(image: np.ndarray, bins: int = 256) -> float:
    """Calculate Shannon entropy of image."""
    hist, _ = np.histogram(image.flatten(), bins=bins, range=(0, 256))
    hist = hist / hist.sum()
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def estimate_noise(image: np.ndarray) -> float:
    """Estimate noise using Laplacian-based method."""
    # Use median absolute deviation of Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return 1.4826 * np.median(np.abs(laplacian - np.median(laplacian)))


def calculate_compression_quality(image_path: str) -> Optional[int]:
    """Estimate JPEG compression quality."""
    try:
        with Image.open(image_path) as img:
            if hasattr(img, 'quantization'):
                # For JPEG, estimate from quantization tables
                quant = img.quantization
                if quant:
                    # Simple heuristic: average quantization value
                    avg_q = np.mean(list(quant.values()))
                    return max(0, min(100, int(100 - avg_q)))
    except Exception:
        pass
    return None


def evaluate_quality(
    metrics: Dict[str, float],
    thresholds: Dict[str, float]
) -> Dict[str, bool]:
    """Evaluate quality metrics against thresholds."""
    flags = {}
    
    flags["is_blurry"] = metrics.get("blur_score", 0) < thresholds.get("blur_score", 100)
    flags["is_low_sharpness"] = metrics.get("sharpness_score", 0) < thresholds.get("sharpness_score", 50)
    flags["is_dark"] = metrics.get("brightness_score", 128) < thresholds.get("brightness_low", 30)
    flags["is_bright"] = metrics.get("brightness_score", 128) > thresholds.get("brightness_high", 220)
    flags["is_low_contrast"] = metrics.get("contrast_score", 0) < thresholds.get("contrast_low", 20)
    flags["is_low_entropy"] = metrics.get("entropy", 0) < thresholds.get("entropy_low", 3.0)
    flags["is_noisy"] = metrics.get("noise_estimation", 0) > thresholds.get("noise_high", 50)
    
    res = metrics.get("resolution", (0, 0))
    min_res = thresholds.get("min_resolution", (100, 100))
    flags["is_low_resolution"] = res[0] < min_res[0] or res[1] < min_res[1]
    
    max_res = thresholds.get("max_resolution", (4000, 4000))
    flags["is_high_resolution"] = res[0] > max_res[0] or res[1] > max_res[1]
    
    ar = metrics.get("aspect_ratio", 1.0)
    ar_range = thresholds.get("aspect_ratio_range", (0.33, 3.0))
    flags["is_bad_aspect_ratio"] = ar < ar_range[0] or ar > ar_range[1]
    
    flags["has_issues"] = any(flags.values())
    
    return flags


def get_quality_category(flags: Dict[str, bool]) -> str:
    """Get quality category from flags."""
    if not flags.get("has_issues", False):
        return "Good"
    
    issues = []
    if flags.get("is_blurry") or flags.get("is_low_sharpness"):
        issues.append("Blurry")
    if flags.get("is_dark"):
        issues.append("Dark")
    if flags.get("is_bright"):
        issues.append("Bright")
    if flags.get("is_low_contrast"):
        issues.append("Low Contrast")
    if flags.get("is_noisy"):
        issues.append("Noisy")
    if flags.get("is_low_resolution"):
        issues.append("Low Resolution")
    if flags.get("is_bad_aspect_ratio"):
        issues.append("Bad Aspect Ratio")
    
    return ", ".join(issues) if issues else "Review"
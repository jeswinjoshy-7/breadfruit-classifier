"""Image processing utilities for the pipeline."""
import cv2
import numpy as np
from PIL import Image, ImageOps, ExifTags
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


# EXIF orientation tag
ORIENTATION_TAG = 274  # ExifTags.Base.Orientation


def load_image(
    file_path: str,
    color_space: str = "RGB",
    correct_orientation: bool = True
) -> Optional[np.ndarray]:
    """Load image with proper color space and orientation handling."""
    try:
        # Use PIL for EXIF orientation handling
        with Image.open(file_path) as pil_img:
            if correct_orientation:
                pil_img = correct_exif_orientation(pil_img)
            
            # Convert to RGB if needed
            if color_space == "RGB" and pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            elif color_space == "BGR" and pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            
            img_array = np.array(pil_img)
            
            if color_space == "BGR":
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_array
    except Exception as e:
        logger.error(f"Failed to load image {file_path}: {e}")
        return None


def correct_exif_orientation(image: Image.Image) -> Image.Image:
    """Correct image orientation based on EXIF data."""
    try:
        exif = image.getexif()
        if exif is None:
            return image
        
        orientation = exif.get(ORIENTATION_TAG, 1)
        
        if orientation == 1:
            return image
        elif orientation == 2:
            return ImageOps.mirror(image)
        elif orientation == 3:
            return image.rotate(180, expand=True)
        elif orientation == 4:
            return ImageOps.flip(image)
        elif orientation == 5:
            return ImageOps.mirror(image.rotate(90, expand=True))
        elif orientation == 6:
            return image.rotate(270, expand=True)
        elif orientation == 7:
            return ImageOps.mirror(image.rotate(270, expand=True))
        elif orientation == 8:
            return image.rotate(90, expand=True)
    except Exception:
        pass
    return image


def save_image(
    image: np.ndarray,
    file_path: str,
    quality: int = 95,
    color_space: str = "RGB"
) -> bool:
    """Save image with specified quality."""
    try:
        if color_space == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(image)
        pil_img.save(file_path, "JPEG", quality=quality, optimize=True)
        return True
    except Exception as e:
        logger.error(f"Failed to save image {file_path}: {e}")
        return False


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    padding_color: Tuple[int, int, int] = (0, 0, 0),
    interpolation: int = cv2.INTER_LANCZOS4
) -> np.ndarray:
    """Resize image preserving aspect ratio with padding."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Create padded image
    padded = np.full((target_h, target_w, 3), padding_color, dtype=image.dtype)
    
    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


def resize_to_efficientnet(
    image: np.ndarray,
    input_size: int = 224,
    padding_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Resize image to EfficientNet input size with padding."""
    return resize_with_padding(image, (input_size, input_size), padding_color)


def remove_alpha_channel(image: np.ndarray) -> np.ndarray:
    """Remove alpha channel if present."""
    if image.shape[2] == 4:
        # Blend with white background
        alpha = image[:, :, 3:4] / 255.0
        rgb = image[:, :, :3]
        white_bg = np.ones_like(rgb) * 255
        result = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        return result
    return image


def normalize_image(
    image: np.ndarray,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """Normalize image for EfficientNet."""
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (image - mean) / std


def denormalize_image(
    image: np.ndarray,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """Denormalize image for visualization."""
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = image * std + mean
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def convert_color_space(image: np.ndarray, target_space: str) -> np.ndarray:
    """Convert image color space."""
    if target_space == "RGB" and len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif target_space == "BGR" and len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif target_space == "GRAY":
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif target_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif target_space == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return image


def get_image_info(image: np.ndarray) -> dict:
    """Get basic image information."""
    return {
        "height": image.shape[0],
        "width": image.shape[1],
        "channels": image.shape[2] if len(image.shape) == 3 else 1,
        "dtype": str(image.dtype),
        "aspect_ratio": image.shape[1] / image.shape[0] if image.shape[0] > 0 else 0,
        "size": image.shape[0] * image.shape[1]
    }


def is_valid_image(image: np.ndarray) -> bool:
    """Check if image array is valid."""
    if image is None:
        return False
    if len(image.shape) not in [2, 3]:
        return False
    if image.shape[0] == 0 or image.shape[1] == 0:
        return False
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False
    return True


def estimate_compression_quality(image_path: str) -> Optional[float]:
    """Estimate JPEG compression quality."""
    try:
        with Image.open(image_path) as img:
            if img.format == "JPEG":
                # Rough estimation based on quantization tables
                if hasattr(img, 'quantization') and img.quantization:
                    qtables = img.quantization.values()
                    if qtables:
                        avg_q = np.mean([np.mean(q) for q in qtables])
                        # Convert to quality scale (rough approximation)
                        quality = max(1, min(100, int(100 - avg_q * 2)))
                        return quality
    except Exception:
        pass
    return None
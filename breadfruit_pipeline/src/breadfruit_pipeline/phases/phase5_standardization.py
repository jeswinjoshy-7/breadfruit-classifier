"""
Phase 5: Image Standardization
Standardizes all accepted images for EfficientNet training.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import logging
import json
import pandas as pd

from breadfruit_pipeline.core.config import get_config
from breadfruit_pipeline.utils.image_utils import (
    load_image,
    save_image,
    resize_to_efficientnet,
    remove_alpha_channel,
    correct_exif_orientation,
    convert_color_space
)

logger = logging.getLogger(__name__)


@dataclass
class StandardizationResult:
    """Result of standardizing a single image."""
    original_path: str
    standardized_path: str
    original_size: Tuple[int, int]
    standardized_size: Tuple[int, int]
    success: bool
    error: str = ""


@dataclass
class StandardizationReport:
    """Complete standardization report."""
    dataset_name: str
    total_images: int
    successful: int
    failed: int
    results: List[StandardizationResult]


class ImageStandardizer:
    """Phase 5: Image Standardization for EfficientNet."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.img_config = self.config.image_processing
        self.phase_config = self.config.phase5_standardization
        
        self.target_size = self.img_config.efficientnet_input_size
        self.target_format = self.img_config.target_format
        self.jpeg_quality = self.img_config.jpeg_quality
        self.color_space = self.img_config.target_color_space
        self.correct_exif = self.img_config.correct_exif_orientation
        self.remove_alpha = self.img_config.remove_alpha_channel
        self.padding_color = tuple(self.img_config.padding_color)
    
    def standardize_dataset(
        self,
        image_paths: List[str],
        dataset_name: str,
        output_dir: str
    ) -> StandardizationReport:
        """Standardize all images in dataset."""
        logger.info(f"Standardizing {dataset_name} ({len(image_paths)} images)")
        
        output_path = Path(output_dir) / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for path in tqdm(image_paths, desc=f"Standardizing {dataset_name}"):
            result = self._standardize_image(path, output_path)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        
        report = StandardizationReport(
            dataset_name=dataset_name,
            total_images=len(results),
            successful=successful,
            failed=failed,
            results=results
        )
        
        logger.info(f"Standardization complete: {successful} successful, {failed} failed")
        return report
    
    def _standardize_image(
        self,
        image_path: str,
        output_dir: Path
    ) -> StandardizationResult:
        """Standardize a single image."""
        try:
            # Load with EXIF correction
            img = load_image(
                image_path,
                color_space="RGB",
                correct_orientation=self.correct_exif
            )
            
            if img is None:
                raise ValueError("Failed to load image")
            
            original_size = (img.shape[1], img.shape[0])
            
            # Remove alpha channel
            if self.remove_alpha:
                img = remove_alpha_channel(img)
            
            # Ensure RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                pass  # Already RGB
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = remove_alpha_channel(img)
            
            # Resize to EfficientNet input size with padding
            img = resize_to_efficientnet(
                img,
                input_size=self.target_size,
                padding_color=self.padding_color
            )
            
            standardized_size = (img.shape[1], img.shape[0])
            
            # Save as JPEG
            output_filename = Path(image_path).stem + ".jpg"
            output_path = output_dir / output_filename
            
            # Handle name conflicts
            counter = 1
            while output_path.exists():
                output_path = output_dir / f"{Path(image_path).stem}_{counter}.jpg"
                counter += 1
            
            success = save_image(
                img,
                str(output_path),
                quality=self.jpeg_quality,
                color_space="RGB"
            )
            
            if not success:
                raise ValueError("Failed to save standardized image")
            
            return StandardizationResult(
                original_path=image_path,
                standardized_path=str(output_path),
                original_size=original_size,
                standardized_size=standardized_size,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Standardization failed for {image_path}: {e}")
            return StandardizationResult(
                original_path=image_path,
                standardized_path="",
                original_size=(0, 0),
                standardized_size=(0, 0),
                success=False,
                error=str(e)
            )
    
    def save_report(self, report: StandardizationReport, output_path: str):
        """Save standardization report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # CSV
        csv_path = output_path.with_suffix('.csv')
        df = pd.DataFrame([asdict(r) for r in report.results])
        df.to_csv(csv_path, index=False)
        
        # Summary
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write(self._format_summary(report))
        
        logger.info(f"Standardization report saved to {output_path.parent}")
    
    def _format_summary(self, report: StandardizationReport) -> str:
        """Format summary report."""
        lines = [
            "=" * 60,
            f"IMAGE STANDARDIZATION REPORT: {report.dataset_name}",
            "=" * 60,
            "",
            f"Total images:       {report.total_images}",
            f"Successful:         {report.successful}",
            f"Failed:             {report.failed}",
            "",
            "Target specifications:",
            f"  Size:               {self.target_size}x{self.target_size}",
            f"  Format:             {self.target_format}",
            f"  Quality:            {self.jpeg_quality}",
            f"  Color space:        {self.color_space}",
            f"  EXIF correction:    {self.correct_exif}",
            f"  Remove alpha:       {self.remove_alpha}",
            f"  Padding color:      {self.padding_color}",
            "",
            "Failures:",
        ]
        
        for r in report.results:
            if not r.success:
                lines.append(f"  {Path(r.original_path).name}: {r.error}")
        
        return "\n".join(lines)
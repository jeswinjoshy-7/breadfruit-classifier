"""
Phase 4: Object Validation
Validates breadfruit presence, size, occlusion, and position in images.
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
from breadfruit_pipeline.utils.file_utils import ensure_dir, safe_move

logger = logging.getLogger(__name__)


@dataclass
class ObjectValidationResult:
    """Object validation result for a single image."""
    path: str
    filename: str
    fruit_detected: bool
    fruit_area_ratio: float
    fruit_count: int
    occlusion_ratio: float
    border_touch_ratio: float
    background_ratio: float
    warnings: List[str]
    validation_passed: bool


@dataclass
class ObjectValidationReport:
    """Complete object validation report."""
    dataset_name: str
    total_images: int
    fruit_detected_count: int
    warnings_generated: int
    validation_passed: int
    results: List[ObjectValidationResult]


class ObjectValidator:
    """Phase 4: Object Validation for breadfruit images."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.obj_config = self.config.phase4_object_validation
        self.methods = self.obj_config.methods
        
        # HSV ranges for breadfruit (green to yellow-brown)
        self.hsv_ranges = []
        if self.methods.color_segmentation.enabled:
            for r in self.methods.color_segmentation.hsv_ranges:
                self.hsv_ranges.append({
                    'h_min': r.h_min, 'h_max': r.h_max,
                    's_min': r.s_min, 's_max': r.s_max,
                    'v_min': r.v_min, 'v_max': r.v_max
                })
        
        self.min_area_ratio = self.methods.color_segmentation.min_fruit_area_ratio
        self.max_area_ratio = self.methods.color_segmentation.max_fruit_area_ratio
        self.max_fruits = self.obj_config.max_fruits_per_image
        self.max_occlusion = self.obj_config.max_occlusion_ratio
        self.max_border_touch = self.obj_config.max_border_touch_ratio
        self.max_background = self.obj_config.max_background_ratio
    
    def validate_dataset(
        self,
        image_paths: List[str],
        dataset_name: str,
        rejected_dir: str
    ) -> ObjectValidationReport:
        """Validate objects in all images."""
        logger.info(f"Validating objects in {dataset_name} ({len(image_paths)} images)")
        
        results = []
        
        for path in tqdm(image_paths, desc=f"Object validation: {dataset_name}"):
            result = self._validate_image(path)
            results.append(result)
        
        # Statistics
        fruit_detected = sum(1 for r in results if r.fruit_detected)
        warnings = sum(len(r.warnings) for r in results)
        passed = sum(1 for r in results if r.validation_passed)
        
        report = ObjectValidationReport(
            dataset_name=dataset_name,
            total_images=len(results),
            fruit_detected_count=fruit_detected,
            warnings_generated=warnings,
            validation_passed=passed,
            results=results
        )
        
        # Move severely problematic images to review
        if self.obj_config.generate_warnings:
            self._move_for_review(results, rejected_dir, dataset_name)
        
        logger.info(f"Object validation complete: {fruit_detected} with fruit detected, {warnings} warnings")
        return report
    
    def _validate_image(self, image_path: str) -> ObjectValidationResult:
        """Validate a single image."""
        warnings = []
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            total_area = h * w
            
            # Color segmentation for fruit detection
            fruit_mask = self._detect_fruit_color(img)
            
            # Find contours
            contours, _ = cv2.findContours(
                fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter contours by area
            fruit_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area / total_area >= self.min_area_ratio:
                    fruit_contours.append(cnt)
            
            fruit_detected = len(fruit_contours) > 0
            fruit_count = len(fruit_contours)
            
            if fruit_detected:
                # Calculate total fruit area
                total_fruit_area = sum(cv2.contourArea(c) for c in fruit_contours)
                fruit_area_ratio = total_fruit_area / total_area
                
                # Check area constraints
                if fruit_area_ratio < self.min_area_ratio:
                    warnings.append(f"Fruit area too small: {fruit_area_ratio:.2%} < {self.min_area_ratio:.2%}")
                if fruit_area_ratio > self.max_area_ratio:
                    warnings.append(f"Fruit area too large: {fruit_area_ratio:.2%} > {self.max_area_ratio:.2%}")
                
                # Check fruit count
                if fruit_count > self.max_fruits:
                    warnings.append(f"Too many fruits detected: {fruit_count} > {self.max_fruits}")
                
                # Check occlusion (if multiple fruits overlap)
                occlusion_ratio = self._estimate_occlusion(fruit_contours, total_area)
                if occlusion_ratio > self.max_occlusion:
                    warnings.append(f"High occlusion: {occlusion_ratio:.2%} > {self.max_occlusion:.2%}")
                
                # Check border touching
                border_touch_ratio = self._check_border_touch(fruit_contours, w, h)
                if border_touch_ratio > self.max_border_touch:
                    warnings.append(f"Fruit touching border: {border_touch_ratio:.2%} > {self.max_border_touch:.2%}")
                
                # Background ratio
                background_ratio = 1.0 - fruit_area_ratio
                if background_ratio > self.max_background:
                    warnings.append(f"Background dominates: {background_ratio:.2%} > {self.max_background:.2%}")
                
            else:
                fruit_area_ratio = 0.0
                occlusion_ratio = 0.0
                border_touch_ratio = 0.0
                background_ratio = 1.0
                warnings.append("No breadfruit detected in image")
            
            validation_passed = len(warnings) == 0
            
            return ObjectValidationResult(
                path=image_path,
                filename=Path(image_path).name,
                fruit_detected=fruit_detected,
                fruit_area_ratio=fruit_area_ratio,
                fruit_count=fruit_count,
                occlusion_ratio=occlusion_ratio,
                border_touch_ratio=border_touch_ratio,
                background_ratio=background_ratio,
                warnings=warnings,
                validation_passed=validation_passed
            )
            
        except Exception as e:
            logger.error(f"Validation failed for {image_path}: {e}")
            return ObjectValidationResult(
                path=image_path,
                filename=Path(image_path).name,
                fruit_detected=False,
                fruit_area_ratio=0.0,
                fruit_count=0,
                occlusion_ratio=0.0,
                border_touch_ratio=0.0,
                background_ratio=1.0,
                warnings=[f"Validation error: {str(e)}"],
                validation_passed=False
            )
    
    def _detect_fruit_color(self, img: np.ndarray) -> np.ndarray:
        """Detect breadfruit using HSV color segmentation."""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for r in self.hsv_ranges:
            lower = np.array([r['h_min'], r['s_min'], r['v_min']])
            upper = np.array([r['h_max'], r['s_max'], r['v_max']])
            range_mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, range_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _estimate_occlusion(
        self,
        contours: List[np.ndarray],
        total_area: int
    ) -> float:
        """Estimate occlusion ratio between fruits."""
        if len(contours) <= 1:
            return 0.0
        
        # Simple approximation: check bounding box overlaps
        bboxes = [cv2.boundingRect(c) for c in contours]
        
        overlap_area = 0
        for i, (x1, y1, w1, h1) in enumerate(bboxes):
            for x2, y2, w2, h2 in bboxes[i+1:]:
                # Calculate intersection
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area += x_overlap * y_overlap
        
        return overlap_area / total_area
    
    def _check_border_touch(
        self,
        contours: List[np.ndarray],
        width: int,
        height: int
    ) -> float:
        """Check how much fruit touches image border."""
        border_thickness = 5
        touch_pixels = 0
        total_border_pixels = 2 * (width + height) * border_thickness
        
        border_mask = np.zeros((height, width), dtype=np.uint8)
        border_mask[:border_thickness, :] = 1
        border_mask[-border_thickness:, :] = 1
        border_mask[:, :border_thickness] = 1
        border_mask[:, -border_thickness:] = 1
        
        for cnt in contours:
            cnt_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
            touch = cv2.bitwise_and(cnt_mask, border_mask)
            touch_pixels += np.sum(touch)
        
        return touch_pixels / total_border_pixels if total_border_pixels > 0 else 0.0
    
    def _move_for_review(
        self,
        results: List[ObjectValidationResult],
        rejected_dir: str,
        dataset_name: str
    ):
        """Move images with warnings to review folder."""
        review_dir = Path(rejected_dir) / "Needs_Review" / dataset_name
        ensure_dir(review_dir)
        
        for result in results:
            if not result.validation_passed and result.warnings:
                src = Path(result.path)
                dst = review_dir / src.name
                
                counter = 1
                while dst.exists():
                    dst = review_dir / f"{src.stem}_obj{counter}{src.suffix}"
                    counter += 1
                
                safe_move(src, dst)
    
    def save_report(self, report: ObjectValidationReport, output_path: str):
        """Save validation report."""
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
        
        logger.info(f"Object validation report saved to {output_path.parent}")
    
    def _format_summary(self, report: ObjectValidationReport) -> str:
        """Format summary report."""
        lines = [
            "=" * 60,
            f"OBJECT VALIDATION REPORT: {report.dataset_name}",
            "=" * 60,
            "",
            f"Total images:           {report.total_images}",
            f"Fruit detected:         {report.fruit_detected_count}",
            f"Validation passed:      {report.validation_passed}",
            f"Total warnings:         {report.warnings_generated}",
            "",
            "Images with warnings:",
        ]
        
        for r in report.results:
            if r.warnings:
                lines.append(f"  {r.filename}:")
                for w in r.warnings:
                    lines.append(f"    - {w}")
        
        return "\n".join(lines)
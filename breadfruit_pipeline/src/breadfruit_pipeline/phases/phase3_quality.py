"""
Phase 3: Image Quality Analysis
Evaluates every image for quality issues and flags problematic images.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import logging
import json
import pandas as pd

from PIL import Image
from breadfruit_pipeline.core.config import get_config
from breadfruit_pipeline.core.exceptions import DatasetError
from breadfruit_pipeline.utils.quality_metrics import (
    calculate_image_quality,
    evaluate_quality,
    get_quality_category
)
from breadfruit_pipeline.utils.file_utils import safe_move, ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Quality assessment result for a single image."""
    path: str
    filename: str
    blur_score: float
    sharpness_score: float
    brightness_score: float
    contrast_score: float
    entropy: float
    noise_estimation: float
    resolution_width: int
    resolution_height: int
    aspect_ratio: float
    compression_quality: Optional[int]
    flags: Dict[str, bool]
    category: str
    review_reason: str = ""


@dataclass
class QualityReport:
    """Complete quality analysis report."""
    dataset_name: str
    total_images: int
    good_images: int
    flagged_images: int
    blurry_count: int
    dark_count: int
    bright_count: int
    noisy_count: int
    low_res_count: int
    bad_ar_count: int
    low_contrast_count: int
    results: List[QualityResult]


class QualityAnalyzer:
    """Phase 3: Image Quality Analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.quality_config = self.config.phase3_quality
        self.thresholds = self.quality_config.thresholds
        self.review_multiplier = self.quality_config.review_threshold_multiplier
    
    def analyze_dataset(
        self,
        image_paths: List[str],
        dataset_name: str,
        rejected_dir: str
    ) -> QualityReport:
        """Analyze quality of all images in dataset."""
        logger.info(f"Analyzing quality of {dataset_name} ({len(image_paths)} images)")
        
        results = []
        
        for path in tqdm(image_paths, desc=f"Quality analysis: {dataset_name}"):
            result = self._analyze_image(path)
            results.append(result)
        
        # Count statistics
        stats = self._compute_statistics(results)
        
        # Create report
        report = QualityReport(
            dataset_name=dataset_name,
            total_images=len(results),
            **stats,
            results=results
        )
        
        # Move flagged images to review
        if self.quality_config.move_flagged:
            self._move_flagged(results, rejected_dir, dataset_name)
        
        logger.info(f"Quality analysis complete: {stats['good_images']} good, {stats['flagged_images']} flagged")
        return report
    
    def _analyze_image(self, image_path: str) -> QualityResult:
        """Analyze a single image."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Calculate metrics
            metrics = calculate_image_quality(img)
            
            # Estimate compression quality
            metrics["compression_quality"] = self._estimate_compression_quality(image_path)
            
            # Evaluate against thresholds
            flags = evaluate_quality(metrics, vars(self.thresholds))
            
            # Apply review threshold multiplier
            review_flags = self._apply_review_thresholds(metrics)
            flags.update(review_flags)
            
            # Determine category
            category = get_quality_category(flags)
            
            # Build review reason
            review_reason = self._build_review_reason(flags)
            
            return QualityResult(
                path=image_path,
                filename=Path(image_path).name,
                blur_score=metrics["blur_score"],
                sharpness_score=metrics["sharpness_score"],
                brightness_score=metrics["brightness_score"],
                contrast_score=metrics["contrast_score"],
                entropy=metrics["entropy"],
                noise_estimation=metrics["noise_estimation"],
                resolution_width=metrics["resolution"][0],
                resolution_height=metrics["resolution"][1],
                aspect_ratio=metrics["aspect_ratio"],
                compression_quality=metrics.get("compression_quality"),
                flags=flags,
                category=category,
                review_reason=review_reason
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze {image_path}: {e}")
            return QualityResult(
                path=image_path,
                filename=Path(image_path).name,
                blur_score=0, sharpness_score=0, brightness_score=0,
                contrast_score=0, entropy=0, noise_estimation=0,
                resolution_width=0, resolution_height=0, aspect_ratio=0,
                compression_quality=None,
                flags={"error": True, "has_issues": True},
                category="Error",
                review_reason=str(e)
            )
    
    def _estimate_compression_quality(self, image_path: str) -> Optional[int]:
        """Estimate JPEG compression quality."""
        try:
            with Image.open(image_path) as img:
                if img.format == 'JPEG' and hasattr(img, 'quantization'):
                    quant = img.quantization
                    if quant:
                        avg_q = np.mean(list(quant.values()))
                        return max(0, min(100, int(100 - avg_q * 0.5)))
        except Exception:
            pass
        return None
    
    def _apply_review_thresholds(self, metrics: Dict) -> Dict[str, bool]:
        """Apply review thresholds (softer than rejection thresholds)."""
        review = {}
        thresh = vars(self.thresholds)
        
        # Use multiplier to create review zone near thresholds
        blur_thresh = thresh.get("blur_score", 100) * self.review_multiplier
        review["review_blurry"] = metrics.get("blur_score", 0) < blur_thresh
        
        bright_low = thresh.get("brightness_low", 30) / self.review_multiplier
        bright_high = thresh.get("brightness_high", 220) * self.review_multiplier
        review["review_dark"] = metrics.get("brightness_score", 128) < bright_low
        review["review_bright"] = metrics.get("brightness_score", 128) > bright_high
        
        contrast_thresh = thresh.get("contrast_low", 20) * self.review_multiplier
        review["review_low_contrast"] = metrics.get("contrast_score", 0) < contrast_thresh
        
        return review
    
    def _build_review_reason(self, flags: Dict[str, bool]) -> str:
        """Build human-readable review reason."""
        reasons = []
        for flag, value in flags.items():
            if value and flag.startswith(("is_", "review_")):
                reason = flag.replace("is_", "").replace("review_", "").replace("_", " ")
                reasons.append(reason.title())
        return "; ".join(reasons)
    
    def _compute_statistics(self, results: List[QualityResult]) -> Dict[str, int]:
        """Compute summary statistics."""
        good = sum(1 for r in results if not r.flags.get("has_issues", False))
        flagged = sum(1 for r in results if r.flags.get("has_issues", False))
        
        return {
            "good_images": good,
            "flagged_images": flagged,
            "blurry_count": sum(1 for r in results if r.flags.get("is_blurry", False)),
            "dark_count": sum(1 for r in results if r.flags.get("is_dark", False)),
            "bright_count": sum(1 for r in results if r.flags.get("is_bright", False)),
            "noisy_count": sum(1 for r in results if r.flags.get("is_noisy", False)),
            "low_res_count": sum(1 for r in results if r.flags.get("is_low_resolution", False)),
            "bad_ar_count": sum(1 for r in results if r.flags.get("is_bad_aspect_ratio", False)),
            "low_contrast_count": sum(1 for r in results if r.flags.get("is_low_contrast", False)),
        }
    
    def _move_flagged(
        self,
        results: List[QualityResult],
        rejected_dir: str,
        dataset_name: str
    ):
        """Move flagged images to review folders."""
        base_rejected = Path(rejected_dir)
        
        # Category to subdir mapping
        category_map = {
            "is_blurry": "Blurry",
            "is_low_sharpness": "Blurry",
            "is_dark": "Dark",
            "is_bright": "Bright",
            "is_noisy": "Needs_Review",
            "is_low_resolution": "Low_Resolution",
            "is_bad_aspect_ratio": "Needs_Review",
            "is_low_contrast": "Needs_Review",
            "is_low_entropy": "Needs_Review",
            "is_over_compressed": "Needs_Review",
        }
        
        for result in results:
            if not result.flags.get("has_issues", False):
                continue
            
            # Determine primary category
            primary_cat = "Needs_Review"
            for flag, subdir in category_map.items():
                if result.flags.get(flag, False):
                    primary_cat = subdir
                    break
            
            src = Path(result.path)
            dst_dir = base_rejected / primary_cat / dataset_name
            ensure_dir(dst_dir)
            
            dst = dst_dir / src.name
            if dst.exists():
                stem = src.stem
                suffix = src.suffix
                counter = 1
                while dst.exists():
                    dst = dst_dir / f"{stem}_q{counter}{suffix}"
                    counter += 1
            
            safe_move(src, dst)
    
    def save_report(self, report: QualityReport, output_path: str):
        """Save quality report."""
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
        
        logger.info(f"Quality report saved to {output_path.parent}")
    
    def _format_summary(self, report: QualityReport) -> str:
        """Format report summary."""
        lines = [
            f"=" * 60,
            f"IMAGE QUALITY ANALYSIS REPORT: {report.dataset_name}",
            f"=" * 60,
            f"",
            f"Total images analyzed:     {report.total_images}",
            f"Good quality images:       {report.good_images}",
            f"Flagged for review:        {report.flagged_images}",
            f"",
            f"Issue breakdown:",
            f"  Blurry:                  {report.blurry_count}",
            f"  Too dark:                {report.dark_count}",
            f"  Too bright:              {report.bright_count}",
            f"  Too noisy:               {report.noisy_count}",
            f"  Low resolution:          {report.low_res_count}",
            f"  Bad aspect ratio:        {report.bad_ar_count}",
            f"  Low contrast:            {report.low_contrast_count}",
            f"",
            f"Flagged images:",
        ]
        
        for r in report.results:
            if r.flags.get("has_issues", False):
                lines.append(f"  {r.filename}: {r.category} - {r.review_reason}")
        
        return "\n".join(lines)
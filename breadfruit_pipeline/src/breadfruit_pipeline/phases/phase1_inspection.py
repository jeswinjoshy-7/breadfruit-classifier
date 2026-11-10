"""
Phase 1: Dataset Inspection
Scans every image and generates a complete dataset report.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import Counter
from tqdm import tqdm
import logging
import json

from breadfruit_pipeline.core.config import get_config
from breadfruit_pipeline.core.exceptions import CorruptedImageError, UnsupportedFormatError, DatasetError
from breadfruit_pipeline.utils.hash_utils import compute_all_hashes
from breadfruit_pipeline.utils.quality_metrics import calculate_image_quality

logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """Information about a single image."""
    path: str
    filename: str
    rel_path: str
    width: int
    height: int
    channels: int
    format: str
    file_size: int
    aspect_ratio: float
    megapixels: float
    is_corrupted: bool
    is_zero_byte: bool
    error_message: str = ""
    phash: str = ""
    dhash: str = ""
    ahash: str = ""
    whash: str = ""


@dataclass
class DatasetReport:
    """Complete dataset inspection report."""
    dataset_name: str
    total_files: int
    valid_images: int
    corrupted_images: int
    zero_byte_files: int
    unsupported_formats: int
    duplicate_filenames: int
    format_distribution: Dict[str, int]
    resolution_distribution: Dict[str, int]
    aspect_ratio_stats: Dict[str, float]
    avg_width: float
    avg_height: float
    avg_megapixels: float
    avg_file_size: float
    total_size_mb: float
    images: List[ImageInfo]


class DatasetInspector:
    """Phase 1: Dataset Inspection."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.inspection_config = self.config.phase1_inspection
        self.supported_formats = set(self.config.image_processing.supported_formats)
        self.results: List[ImageInfo] = []
    
    def inspect_dataset(self, dataset_path: str, dataset_name: str) -> DatasetReport:
        """Perform complete dataset inspection."""
        logger.info(f"Starting inspection of {dataset_name} at {dataset_path}")
        
        path = Path(dataset_path)
        if not path.exists():
            raise DatasetError(f"Dataset path does not exist: {dataset_path}")
        
        # Collect all image files
        image_files = list(self._get_image_files(path))
        logger.info(f"Found {len(image_files)} image files")
        
        # Inspect each image
        self.results = []
        for file_path in tqdm(image_files, desc=f"Inspecting {dataset_name}"):
            info = self._inspect_image(file_path, path, dataset_name)
            self.results.append(info)
        
        # Generate report
        report = self._generate_report(dataset_name)
        logger.info(f"Inspection complete: {report.valid_images} valid, {report.corrupted_images} corrupted")
        
        return report
    
    def _get_image_files(self, path: Path):
        """Get all image files in directory."""
        for ext in self.supported_formats:
            yield from path.rglob(f"*{ext}")
            yield from path.rglob(f"*{ext.upper()}")
    
    def _inspect_image(self, file_path: Path, base_path: Path, dataset_name: str) -> ImageInfo:
        """Inspect a single image file."""
        rel_path = file_path.relative_to(base_path)
        file_size = file_path.stat().st_size
        
        # Check zero byte
        if file_size == 0:
            return ImageInfo(
                path=str(file_path),
                filename=file_path.name,
                rel_path=str(rel_path),
                width=0, height=0, channels=0,
                format=file_path.suffix.lower(),
                file_size=0,
                aspect_ratio=0.0,
                megapixels=0.0,
                is_corrupted=False,
                is_zero_byte=True,
                error_message="Zero byte file"
            )
        
        # Check format
        if file_path.suffix.lower() not in self.supported_formats:
            return ImageInfo(
                path=str(file_path),
                filename=file_path.name,
                rel_path=str(rel_path),
                width=0, height=0, channels=0,
                format=file_path.suffix.lower(),
                file_size=file_size,
                aspect_ratio=0.0,
                megapixels=0.0,
                is_corrupted=False,
                is_zero_byte=False,
                error_message=f"Unsupported format: {file_path.suffix}"
            )
        
        # Try to read image
        try:
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                raise CorruptedImageError(f"OpenCV could not read: {file_path}")
            
            # Get dimensions
            if len(img.shape) == 3:
                height, width, channels = img.shape
            else:
                height, width = img.shape
                channels = 1
            
            # Compute hashes if enabled
            phash = dhash = ahash = whash = ""
            if getattr(self.inspection_config, 'compute_hashes', True):
                hashes = compute_all_hashes(img)
                phash = str(hashes.get("phash", ""))
                dhash = str(hashes.get("dhash", ""))
                ahash = str(hashes.get("ahash", ""))
                whash = str(hashes.get("whash", ""))
            
            return ImageInfo(
                path=str(file_path),
                filename=file_path.name,
                rel_path=str(rel_path),
                width=width,
                height=height,
                channels=channels,
                format=file_path.suffix.lower(),
                file_size=file_size,
                aspect_ratio=width / height if height > 0 else 0,
                megapixels=(width * height) / 1_000_000,
                is_corrupted=False,
                is_zero_byte=False,
                phash=phash,
                dhash=dhash,
                ahash=ahash,
                whash=whash
            )
            
        except Exception as e:
            logger.warning(f"Failed to inspect {file_path}: {e}")
            return ImageInfo(
                path=str(file_path),
                filename=file_path.name,
                rel_path=str(rel_path),
                width=0, height=0, channels=0,
                format=file_path.suffix.lower(),
                file_size=file_size,
                aspect_ratio=0.0,
                megapixels=0.0,
                is_corrupted=True,
                is_zero_byte=False,
                error_message=str(e)
            )
    
    def _generate_report(self, dataset_name: str) -> DatasetReport:
        """Generate dataset report from inspection results."""
        valid = [r for r in self.results if not r.is_corrupted and not r.is_zero_byte]
        corrupted = [r for r in self.results if r.is_corrupted]
        zero_byte = [r for r in self.results if r.is_zero_byte]
        unsupported = [r for r in self.results if r.error_message and "Unsupported format" in r.error_message]
        
        # Format distribution
        format_dist = Counter(r.format for r in self.results)
        
        # Resolution distribution (binned)
        res_dist = Counter()
        for r in valid:
            res_key = f"{r.width}x{r.height}"
            res_dist[res_key] += 1
        
        # Aspect ratio stats
        aspect_ratios = [r.aspect_ratio for r in valid if r.aspect_ratio > 0]
        
        # Duplicate filenames
        filename_counts = Counter(r.filename for r in self.results)
        duplicate_filenames = sum(1 for count in filename_counts.values() if count > 1)
        
        report = DatasetReport(
            dataset_name=dataset_name,
            total_files=len(self.results),
            valid_images=len(valid),
            corrupted_images=len(corrupted),
            zero_byte_files=len(zero_byte),
            unsupported_formats=len(unsupported),
            duplicate_filenames=duplicate_filenames,
            format_distribution=dict(format_dist),
            resolution_distribution=dict(res_dist.most_common(20)),
            aspect_ratio_stats={
                "min": min(aspect_ratios) if aspect_ratios else 0,
                "max": max(aspect_ratios) if aspect_ratios else 0,
                "mean": np.mean(aspect_ratios) if aspect_ratios else 0,
                "median": np.median(aspect_ratios) if aspect_ratios else 0,
                "std": np.std(aspect_ratios) if aspect_ratios else 0
            },
            avg_width=np.mean([r.width for r in valid]) if valid else 0,
            avg_height=np.mean([r.height for r in valid]) if valid else 0,
            avg_megapixels=np.mean([r.megapixels for r in valid]) if valid else 0,
            avg_file_size=np.mean([r.file_size for r in valid]) if valid else 0,
            total_size_mb=sum(r.file_size for r in self.results) / (1024 * 1024),
            images=self.results
        )
        
        return report
    
    def save_report(self, report: DatasetReport, output_path: str):
        """Save report to JSON and CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save full report as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            # Convert dataclasses to dict
            report_dict = asdict(report)
            json.dump(report_dict, f, indent=2, default=str)
        
        # Save images info as CSV
        csv_path = output_path.with_suffix('.csv')
        df = pd.DataFrame([asdict(img) for img in report.images])
        df.to_csv(csv_path, index=False)
        
        # Save summary as text
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write(self._format_summary(report))
        
        logger.info(f"Report saved to {output_path.parent}")
    
    def _format_summary(self, report: DatasetReport) -> str:
        """Format report as human-readable text."""
        lines = [
            f"=" * 60,
            f"DATASET INSPECTION REPORT: {report.dataset_name}",
            f"=" * 60,
            f"",
            f"OVERVIEW:",
            f"  Total files scanned:     {report.total_files}",
            f"  Valid images:            {report.valid_images}",
            f"  Corrupted images:        {report.corrupted_images}",
            f"  Zero-byte files:         {report.zero_byte_files}",
            f"  Unsupported formats:     {report.unsupported_formats}",
            f"  Duplicate filenames:     {report.duplicate_filenames}",
            f"  Total size:              {report.total_size_mb:.2f} MB",
            f"",
            f"IMAGE STATISTICS:",
            f"  Average resolution:      {report.avg_width:.0f} x {report.avg_height:.0f}",
            f"  Average megapixels:      {report.avg_megapixels:.2f} MP",
            f"  Average file size:       {report.avg_file_size / 1024:.1f} KB",
            f"",
            f"ASPECT RATIO:",
            f"  Min:                     {report.aspect_ratio_stats['min']:.2f}",
            f"  Max:                     {report.aspect_ratio_stats['max']:.2f}",
            f"  Mean:                    {report.aspect_ratio_stats['mean']:.2f}",
            f"  Median:                  {report.aspect_ratio_stats['median']:.2f}",
            f"  Std:                     {report.aspect_ratio_stats['std']:.2f}",
            f"",
            f"FORMAT DISTRIBUTION:",
        ]
        
        for fmt, count in sorted(report.format_distribution.items()):
            lines.append(f"  {fmt}: {count}")
        
        lines.extend([
            f"",
            f"TOP 10 RESOLUTIONS:",
        ])
        
        for res, count in list(report.resolution_distribution.items())[:10]:
            lines.append(f"  {res}: {count}")
        
        return "\n".join(lines)


def run_phase1_inspection(config_path: Optional[str] = None) -> Dict[str, DatasetReport]:
    """Run Phase 1 inspection on all configured datasets."""
    config = get_config(config_path)
    inspector = DatasetInspector(config_path)
    
    reports = {}
    for dataset_key, dataset_config in config.datasets.items():
        report = inspector.inspect_dataset(dataset_config.path, dataset_config.name)
        
        # Save report
        output_dir = Path(config.output.base_dir) / config.output.structure.reports
        report_path = output_dir / f"phase1_inspection_{dataset_key}"
        inspector.save_report(report, str(report_path))
        
        reports[dataset_key] = report
    
    return reports
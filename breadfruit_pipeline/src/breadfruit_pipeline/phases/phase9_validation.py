"""
Phase 9: Final Dataset Validation
Validates the complete dataset meets all requirements for EfficientNet training.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import logging
import json
import pandas as pd

from breadfruit_pipeline.core.config import get_config
from breadfruit_pipeline.utils.image_utils import load_image

logger = logging.getLogger(__name__)


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any]


@dataclass
class ValidationReport:
    """Complete final validation report."""
    dataset_name: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    checks: List[ValidationCheck]
    overall_passed: bool


class FinalValidator:
    """Phase 9: Final Dataset Validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.target_size = self.config.image_processing.efficientnet_input_size
        self.target_channels = 3
        self.target_format = "JPEG"
    
    def validate_dataset(
        self,
        dataset_dir: str,
        dataset_name: str
    ) -> ValidationReport:
        """Run all validation checks on final dataset."""
        logger.info(f"Running final validation on {dataset_name}")
        
        checks = []
        
        # Get all images
        image_paths = list(Path(dataset_dir).rglob("*.jpg"))
        image_paths.extend(Path(dataset_dir).rglob("*.jpeg"))
        
        if not image_paths:
            return ValidationReport(
                dataset_name=dataset_name,
                total_checks=0,
                passed_checks=0,
                failed_checks=1,
                checks=[ValidationCheck(
                    name="dataset_not_empty",
                    passed=False,
                    message="No images found in dataset",
                    details={}
                )],
                overall_passed=False
            )
        
        # Run all checks
        checks.append(self._check_corrupted_images(image_paths))
        checks.append(self._check_duplicate_images(image_paths))
        checks.append(self._check_consistent_sizes(image_paths))
        checks.append(self._check_consistent_channels(image_paths))
        checks.append(self._check_valid_format(image_paths))
        checks.append(self._check_efficientnet_compatibility(image_paths))
        checks.append(self._check_folder_structure(dataset_dir))
        checks.append(self._check_no_corrupted_files(image_paths))
        
        passed = sum(1 for c in checks if c.passed)
        total = len(checks)
        
        report = ValidationReport(
            dataset_name=dataset_name,
            total_checks=total,
            passed_checks=passed,
            failed_checks=total - passed,
            checks=checks,
            overall_passed=passed == total
        )
        
        logger.info(f"Validation: {passed}/{total} checks passed")
        return report
    
    def validate_all_datasets(self, base_dir: str) -> Dict[str, ValidationReport]:
        """Validate all datasets in the Breadfruit_Dataset structure."""
        reports = {}
        
        base_path = Path(base_dir)
        
        # Check Personal_Clean
        personal_dir = base_path / "Personal_Clean"
        if personal_dir.exists():
            reports["personal"] = self.validate_dataset(str(personal_dir), "Personal_Clean")
        
        # Check Public_Clean
        public_dir = base_path / "Public_Clean"
        if public_dir.exists():
            reports["public"] = self.validate_dataset(str(public_dir), "Public_Clean")
        
        # Combined validation
        all_images = list(personal_dir.rglob("*.jpg")) + list(public_dir.rglob("*.jpg"))
        if all_images:
            reports["combined"] = self.validate_dataset(str(base_path), "Combined")
        
        return reports
    
    def _check_corrupted_images(self, image_paths: List[Path]) -> ValidationCheck:
        """Check for corrupted/unreadable images."""
        corrupted = []
        
        for path in tqdm(image_paths, desc="Checking corruption", leave=False):
            try:
                img = cv2.imread(str(path))
                if img is None:
                    corrupted.append(str(path))
            except Exception:
                corrupted.append(str(path))
        
        passed = len(corrupted) == 0
        return ValidationCheck(
            name="no_corrupted_images",
            passed=passed,
            message=f"Found {len(corrupted)} corrupted images" if not passed else "No corrupted images",
            details={"corrupted_count": len(corrupted), "corrupted_files": corrupted[:10]}
        )
    
    def _check_duplicate_images(self, image_paths: List[Path]) -> ValidationCheck:
        """Check for duplicate images using perceptual hash."""
        from breadfruit_pipeline.utils.hash_utils import compute_phash, find_duplicate_groups
        
        hashes = {}
        for path in tqdm(image_paths, desc="Computing hashes", leave=False):
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    hashes[str(path)] = compute_phash(img)
            except Exception:
                pass
        
        duplicate_groups = find_duplicate_groups(hashes, threshold=5)
        total_duplicates = sum(len(g) - 1 for g in duplicate_groups)
        
        passed = total_duplicates == 0
        return ValidationCheck(
            name="no_duplicate_images",
            passed=passed,
            message=f"Found {total_duplicates} duplicate images" if not passed else "No duplicates found",
            details={
                "duplicate_groups": len(duplicate_groups),
                "total_duplicates": total_duplicates
            }
        )
    
    def _check_consistent_sizes(self, image_paths: List[Path]) -> ValidationCheck:
        """Check all images have consistent sizes."""
        sizes = {}
        
        for path in tqdm(image_paths, desc="Checking sizes", leave=False):
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    size = (img.shape[1], img.shape[0])  # (width, height)
                    sizes[size] = sizes.get(size, 0) + 1
            except Exception:
                pass
        
        # Expected size
        expected = (self.target_size, self.target_size)
        
        passed = len(sizes) == 1 and expected in sizes
        return ValidationCheck(
            name="consistent_image_sizes",
            passed=passed,
            message=f"All images are {self.target_size}x{self.target_size}" if passed 
                   else f"Found {len(sizes)} different sizes: {list(sizes.keys())[:5]}",
            details={
                "size_distribution": {f"{w}x{h}": c for (w, h), c in sizes.items()},
                "expected_size": f"{expected[0]}x{expected[1]}"
            }
        )
    
    def _check_consistent_channels(self, image_paths: List[Path]) -> ValidationCheck:
        """Check all images have consistent color channels."""
        channel_counts = {}
        
        for path in tqdm(image_paths, desc="Checking channels", leave=False):
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    channels = img.shape[2] if len(img.shape) == 3 else 1
                    channel_counts[channels] = channel_counts.get(channels, 0) + 1
            except Exception:
                pass
        
        passed = len(channel_counts) == 1 and 3 in channel_counts
        return ValidationCheck(
            name="consistent_color_channels",
            passed=passed,
            message="All images have 3 channels (RGB)" if passed 
                   else f"Found channels: {list(channel_counts.keys())}",
            details={"channel_distribution": channel_counts}
        )
    
    def _check_valid_format(self, image_paths: List[Path]) -> ValidationCheck:
        """Check all images are valid JPEG format."""
        invalid = []
        
        for path in tqdm(image_paths, desc="Checking format", leave=False):
            if path.suffix.lower() not in ['.jpg', '.jpeg']:
                invalid.append(str(path))
        
        passed = len(invalid) == 0
        return ValidationCheck(
            name="valid_jpeg_format",
            passed=passed,
            message="All images are JPEG format" if passed 
                   else f"Found {len(invalid)} non-JPEG files",
            details={"invalid_count": len(invalid), "invalid_files": invalid[:10]}
        )
    
    def _check_efficientnet_compatibility(self, image_paths: List[Path]) -> ValidationCheck:
        """Check images are compatible with EfficientNet input requirements."""
        issues = []
        
        for path in tqdm(image_paths, desc="Checking EfficientNet compatibility", leave=False):
            try:
                img = load_image(str(path), color_space="RGB")
                if img is not None:
                    h, w = img.shape[:2]
                    if h != self.target_size or w != self.target_size:
                        issues.append(f"{path.name}: {w}x{h} != {self.target_size}x{self.target_size}")
                    
                    if img.dtype != np.uint8:
                        issues.append(f"{path.name}: dtype {img.dtype} != uint8")
                    
                    if len(img.shape) != 3 or img.shape[2] != 3:
                        issues.append(f"{path.name}: shape {img.shape} != (H,W,3)")
            except Exception as e:
                issues.append(f"{path.name}: {str(e)}")
        
        passed = len(issues) == 0
        return ValidationCheck(
            name="efficientnet_compatibility",
            passed=passed,
            message="All images compatible with EfficientNet" if passed 
                   else f"Found {len(issues)} compatibility issues",
            details={"issue_count": len(issues), "issues": issues[:20]}
        )
    
    def _check_folder_structure(self, dataset_dir: str) -> ValidationCheck:
        """Check folder structure is valid."""
        path = Path(dataset_dir)
        expected_subdirs = ["Personal_Clean", "Public_Clean", "Rejected", "Reports", "Logs"]
        
        missing = [d for d in expected_subdirs if not (path / d).exists()]
        
        passed = len(missing) == 0
        return ValidationCheck(
            name="valid_folder_structure",
            passed=passed,
            message="Folder structure is valid" if passed 
                   else f"Missing directories: {missing}",
            details={"missing_directories": missing}
        )
    
    def _check_no_corrupted_files(self, image_paths: List[Path]) -> ValidationCheck:
        """Check no zero-byte or truncated files."""
        zero_byte = []
        truncated = []
        
        for path in tqdm(image_paths, desc="Checking file integrity", leave=False):
            try:
                if path.stat().st_size == 0:
                    zero_byte.append(str(path))
                else:
                    # Try to read full image
                    img = cv2.imread(str(path))
                    if img is None:
                        truncated.append(str(path))
            except Exception:
                truncated.append(str(path))
        
        passed = len(zero_byte) == 0 and len(truncated) == 0
        return ValidationCheck(
            name="file_integrity",
            passed=passed,
            message="All files intact" if passed 
                   else f"Found {len(zero_byte)} zero-byte, {len(truncated)} truncated files",
            details={
                "zero_byte_files": zero_byte[:10],
                "truncated_files": truncated[:10]
            }
        )
    
    def save_report(self, report: ValidationReport, output_path: str):
        """Save validation report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Summary
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write(self._format_report(report))
        
        logger.info(f"Validation report saved to {output_path.parent}")
    
    def _format_report(self, report: ValidationReport) -> str:
        """Format validation report."""
        lines = [
            "=" * 60,
            f"FINAL VALIDATION REPORT: {report.dataset_name}",
            "=" * 60,
            f"",
            f"Overall Status: {'PASSED' if report.overall_passed else 'FAILED'}",
            f"Checks Passed:    {report.passed_checks}/{report.total_checks}",
            f"",
            f"Detailed Checks:",
        ]
        
        for check in report.checks:
            status = "✓ PASS" if check.passed else "✗ FAIL"
            lines.append(f"  {status} - {check.name}")
            lines.append(f"      {check.message}")
            if check.details:
                for key, val in check.details.items():
                    if key not in ["corrupted_files", "invalid_files", "issues", 
                                   "zero_byte_files", "truncated_files", 
                                   "duplicate_files", "missing_directories"]:
                        lines.append(f"      {key}: {val}")
        
        return "\n".join(lines)
"""
Phase 8: Report Generation
Generates comprehensive reports for all pipeline phases.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
import json

from breadfruit_pipeline.core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    dataset_name: str
    reports_dir: str
    output_formats: List[str] = None
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["csv", "txt"]


class ReportGenerator:
    """Phase 8: Report Generation for all pipeline phases."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.reports_dir = Path(self.config.output.base_dir) / self.config.output.structure.reports
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_reports(
        self,
        phase_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate all reports from phase results."""
        logger.info("Generating comprehensive reports")
        
        report_paths = {}
        
        # Dataset Statistics
        if "inspection" in phase_results:
            paths = self._generate_dataset_statistics(phase_results["inspection"])
            report_paths.update(paths)
        
        # Image Quality Report
        if "quality" in phase_results:
            paths = self._generate_quality_report(phase_results["quality"])
            report_paths.update(paths)
        
        # Duplicate Report
        if "duplicates" in phase_results:
            paths = self._generate_duplicate_report(phase_results["duplicates"])
            report_paths.update(paths)
        
        # Object Validation Report
        if "object_validation" in phase_results:
            paths = self._generate_object_validation_report(phase_results["object_validation"])
            report_paths.update(paths)
        
        # Standardization Report
        if "standardization" in phase_results:
            paths = self._generate_standardization_report(phase_results["standardization"])
            report_paths.update(paths)
        
        # Preprocessing Log
        log_path = self._generate_preprocessing_log(phase_results)
        report_paths["preprocessing_log"] = log_path
        
        # Dataset Summary
        summary_path = self._generate_dataset_summary(phase_results)
        report_paths["dataset_summary"] = summary_path
        
        # Quality Metrics Summary
        metrics_path = self._generate_quality_metrics(phase_results)
        report_paths["quality_metrics"] = metrics_path
        
        logger.info(f"All reports generated in {self.reports_dir}")
        return report_paths
    
    def _generate_dataset_statistics(
        self,
        inspection_results: Dict
    ) -> Dict[str, str]:
        """Generate dataset_statistics.csv and related reports."""
        paths = {}
        
        # Combine all dataset inspections
        all_images = []
        for dataset_name, report in inspection_results.items():
            for img in report.images:
                img_dict = asdict(img)
                img_dict["dataset"] = dataset_name
                all_images.append(img_dict)
        
        if not all_images:
            return paths
        
        df = pd.DataFrame(all_images)
        
        # Dataset statistics CSV
        csv_path = self.reports_dir / "dataset_statistics.csv"
        df.to_csv(csv_path, index=False)
        paths["dataset_statistics_csv"] = str(csv_path)
        
        # Summary text
        txt_path = self.reports_dir / "dataset_statistics.txt"
        with open(txt_path, 'w') as f:
            f.write(self._format_dataset_summary(df))
        paths["dataset_statistics_txt"] = str(txt_path)
        
        return paths
    
    def _generate_quality_report(
        self,
        quality_results: Dict
    ) -> Dict[str, str]:
        """Generate image_quality_report.csv."""
        paths = {}
        
        all_results = []
        for dataset_name, report in quality_results.items():
            for r in report.results:
                d = asdict(r)
                d["dataset"] = dataset_name
                all_results.append(d)
        
        if not all_results:
            return paths
        
        df = pd.DataFrame(all_results)
        
        csv_path = self.reports_dir / "image_quality_report.csv"
        df.to_csv(csv_path, index=False)
        paths["image_quality_csv"] = str(csv_path)
        
        # Summary with flagged images
        flagged = df[df["flags"].apply(lambda x: x.get("has_issues", False))]
        txt_path = self.reports_dir / "image_quality_report.txt"
        with open(txt_path, 'w') as f:
            f.write(self._format_quality_summary(df, flagged))
        paths["image_quality_txt"] = str(txt_path)
        
        return paths
    
    def _generate_duplicate_report(
        self,
        duplicate_results: Dict
    ) -> Dict[str, str]:
        """Generate duplicate_report.csv."""
        paths = {}
        
        all_pairs = []
        all_groups = []
        
        for dataset_name, report in duplicate_results.items():
            for p in report.pairs:
                d = asdict(p)
                d["dataset"] = dataset_name
                all_pairs.append(d)
            
            for g in report.groups:
                d = asdict(g)
                d["dataset"] = dataset_name
                all_groups.append(d)
        
        if all_pairs:
            df_pairs = pd.DataFrame(all_pairs)
            csv_path = self.reports_dir / "duplicate_report.csv"
            df_pairs.to_csv(csv_path, index=False)
            paths["duplicate_report_csv"] = str(csv_path)
        
        if all_groups:
            df_groups = pd.DataFrame(all_groups)
            csv_path = self.reports_dir / "duplicate_groups.csv"
            df_groups.to_csv(csv_path, index=False)
            paths["duplicate_groups_csv"] = str(csv_path)
        
        return paths
    
    def _generate_object_validation_report(
        self,
        validation_results: Dict
    ) -> Dict[str, str]:
        """Generate object validation report."""
        paths = {}
        
        all_results = []
        for dataset_name, report in validation_results.items():
            for r in report.results:
                d = asdict(r)
                d["dataset"] = dataset_name
                all_results.append(d)
        
        if not all_results:
            return paths
        
        df = pd.DataFrame(all_results)
        
        csv_path = self.reports_dir / "object_validation_report.csv"
        df.to_csv(csv_path, index=False)
        paths["object_validation_csv"] = str(csv_path)
        
        txt_path = self.reports_dir / "object_validation_report.txt"
        with open(txt_path, 'w') as f:
            f.write(self._format_validation_summary(df))
        paths["object_validation_txt"] = str(txt_path)
        
        return paths
    
    def _generate_standardization_report(
        self,
        standardization_results: Dict
    ) -> Dict[str, str]:
        """Generate standardization report."""
        paths = {}
        
        all_results = []
        for dataset_name, report in standardization_results.items():
            for r in report.results:
                d = asdict(r)
                d["dataset"] = dataset_name
                all_results.append(d)
        
        if not all_results:
            return paths
        
        df = pd.DataFrame(all_results)
        
        csv_path = self.reports_dir / "standardization_report.csv"
        df.to_csv(csv_path, index=False)
        paths["standardization_csv"] = str(csv_path)
        
        return paths
    
    def _generate_preprocessing_log(
        self,
        phase_results: Dict
    ) -> str:
        """Generate preprocessing_log.txt with all operations."""
        log_path = self.reports_dir / "preprocessing_log.txt"
        
        with open(log_path, 'w') as f:
            f.write("BREADFRUIT DATASET - PREPROCESSING LOG\n")
            f.write("=" * 60 + "\n\n")
            
            for phase, results in phase_results.items():
                f.write(f"\n{phase.upper()} PHASE\n")
                f.write("-" * 40 + "\n")
                
                if isinstance(results, dict):
                    for key, value in results.items():
                        if hasattr(value, 'total_images'):
                            f.write(f"  {key}: {value.total_images} images\n")
                        elif hasattr(value, 'successful'):
                            f.write(f"  {key}: {value.successful} successful\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {results}\n")
        
        return str(log_path)
    
    def _generate_dataset_summary(
        self,
        phase_results: Dict
    ) -> str:
        """Generate dataset_summary.txt with overall statistics."""
        summary_path = self.reports_dir / "dataset_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("BREADFRUIT DATASET - FINAL SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Count clean images per dataset
            if "standardization" in phase_results:
                for name, report in phase_results["standardization"].items():
                    f.write(f"{name}:\n")
                    f.write(f"  Clean images: {report.successful}\n")
                    f.write(f"  Failed: {report.failed}\n\n")
            
            # Rejected counts
            if "quality" in phase_results:
                for name, report in phase_results["quality"].items():
                    f.write(f"{name} - Quality rejected:\n")
                    f.write(f"  Flagged: {report.flagged_images}\n")
                    f.write(f"  Blurry: {report.blurry_count}\n")
                    f.write(f"  Dark: {report.dark_count}\n")
                    f.write(f"  Bright: {report.bright_count}\n\n")
            
            if "duplicates" in phase_results:
                for name, report in phase_results["duplicates"].items():
                    f.write(f"{name} - Duplicates:\n")
                    f.write(f"  Groups: {len(report.groups)}\n")
                    f.write(f"  Images to remove: {report.images_flagged}\n\n")
        
        return str(summary_path)
    
    def _generate_quality_metrics(
        self,
        phase_results: Dict
    ) -> str:
        """Generate quality_metrics.csv with aggregate statistics."""
        metrics_path = self.reports_dir / "quality_metrics.csv"
        
        if "quality" not in phase_results:
            return str(metrics_path)
        
        rows = []
        for dataset_name, report in phase_results["quality"].items():
            # Calculate aggregate metrics
            blur_scores = [r.blur_score for r in report.results]
            sharpness = [r.sharpness_score for r in report.results]
            brightness = [r.brightness_score for r in report.results]
            contrast = [r.contrast_score for r in report.results]
            entropy = [r.entropy for r in report.results]
            noise = [r.noise_estimation for r in report.results]
            
            rows.append({
                "dataset": dataset_name,
                "images": len(report.results),
                "blur_mean": np.mean(blur_scores),
                "blur_std": np.std(blur_scores),
                "blur_min": np.min(blur_scores),
                "blur_max": np.max(blur_scores),
                "sharpness_mean": np.mean(sharpness),
                "brightness_mean": np.mean(brightness),
                "contrast_mean": np.mean(contrast),
                "entropy_mean": np.mean(entropy),
                "noise_mean": np.mean(noise),
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(metrics_path, index=False)
        
        return str(metrics_path)
    
    def _format_dataset_summary(self, df: pd.DataFrame) -> str:
        """Format dataset summary text."""
        lines = [
            "DATASET STATISTICS SUMMARY",
            "=" * 40,
            f"Total images: {len(df)}",
            f"Datasets: {df['dataset'].nunique()}",
            f"Formats: {df['format'].nunique()}",
            f"Corrupted: {df['is_corrupted'].sum()}",
            f"Zero byte: {df['is_zero_byte'].sum()}",
            "",
            "Format distribution:",
        ]
        
        for fmt, count in df['format'].value_counts().items():
            lines.append(f"  {fmt}: {count}")
        
        lines.extend([
            "",
            "Resolution stats:",
            f"  Mean: {df['width'].mean():.0f} x {df['height'].mean():.0f}",
            f"  Min: {df['width'].min()} x {df['height'].min()}",
            f"  Max: {df['width'].max()} x {df['height'].max()}",
            "",
            "Aspect ratio:",
            f"  Mean: {df['aspect_ratio'].mean():.2f}",
            f"  Std: {df['aspect_ratio'].std():.2f}",
        ])
        
        return "\n".join(lines)
    
    def _format_quality_summary(
        self,
        df: pd.DataFrame,
        flagged: pd.DataFrame
    ) -> str:
        """Format quality summary."""
        lines = [
            "IMAGE QUALITY REPORT",
            "=" * 40,
            f"Total images: {len(df)}",
            f"Good quality: {len(df) - len(flagged)}",
            f"Flagged for review: {len(flagged)}",
            "",
            "Issue breakdown:",
        ]
        
        for col in ["is_blurry", "is_dark", "is_bright", "is_noisy", 
                    "is_low_resolution", "is_low_contrast"]:
            if col in df.columns:
                count = sum(1 for f in df['flags'] if f.get(col, False))
                lines.append(f"  {col}: {count}")
        
        lines.extend([
            "",
            "Quality metrics (all images):",
            f"  Blur score mean: {df['blur_score'].mean():.1f}",
            f"  Sharpness mean: {df['sharpness_score'].mean():.1f}",
            f"  Brightness mean: {df['brightness_score'].mean():.1f}",
            f"  Contrast mean: {df['contrast_score'].mean():.1f}",
            f"  Entropy mean: {df['entropy'].mean():.2f}",
            f"  Noise mean: {df['noise_estimation'].mean():.1f}",
        ])
        
        return "\n".join(lines)
    
    def _format_validation_summary(self, df: pd.DataFrame) -> str:
        """Format object validation summary."""
        lines = [
            "OBJECT VALIDATION REPORT",
            "=" * 40,
            f"Total images: {len(df)}",
            f"Fruit detected: {df['fruit_detected'].sum()}",
            f"Validation passed: {df['validation_passed'].sum()}",
            "",
            "Warnings generated:",
        ]
        
        warning_counts = {}
        for warnings in df['warnings']:
            for w in warnings:
                key = w.split(":")[0]
                warning_counts[key] = warning_counts.get(key, 0) + 1
        
        for key, count in sorted(warning_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {key}: {count}")
        
        return "\n".join(lines)
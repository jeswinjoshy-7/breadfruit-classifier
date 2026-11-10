"""
Phase 7: Dataset Analysis
Generates comprehensive statistics and visualizations.
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
import json

from breadfruit_pipeline.core.config import get_config
from breadfruit_pipeline.utils.file_utils import get_image_files

logger = logging.getLogger(__name__)


@dataclass
class DatasetAnalysis:
    """Complete dataset analysis results."""
    dataset_name: str
    class_distribution: Dict[str, int]
    resolution_stats: Dict
    aspect_ratio_stats: Dict
    file_size_stats: Dict
    quality_stats: Dict
    image_count: int


class DatasetAnalyzer:
    """Phase 7: Dataset Analysis and Visualization."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.analysis_config = self.config.phase7_analysis
        self.dpi = self.analysis_config.figure_dpi
        self.style = self.analysis_config.plot_style
    
    def analyze_dataset(
        self,
        image_paths: List[str],
        dataset_name: str,
        quality_reports: Optional[Dict] = None
    ) -> DatasetAnalysis:
        """Perform comprehensive dataset analysis."""
        logger.info(f"Analyzing dataset: {dataset_name}")
        
        # Load quality data if available
        quality_df = None
        if quality_reports and dataset_name in quality_reports:
            quality_df = pd.DataFrame(quality_reports[dataset_name])
        
        # Collect statistics
        class_dist = self._analyze_classes(image_paths)
        resolution_stats = self._analyze_resolutions(image_paths)
        aspect_stats = self._analyze_aspect_ratios(image_paths)
        size_stats = self._analyze_file_sizes(image_paths)
        quality_stats = self._analyze_quality(quality_df) if quality_df is not None else {}
        
        analysis = DatasetAnalysis(
            dataset_name=dataset_name,
            class_distribution=class_dist,
            resolution_stats=resolution_stats,
            aspect_ratio_stats=aspect_stats,
            file_size_stats=size_stats,
            quality_stats=quality_stats,
            image_count=len(image_paths)
        )
        
        # Generate visualizations
        if self.analysis_config.generate_visualizations:
            self._generate_visualizations(
                image_paths,
                dataset_name,
                analysis,
                quality_df
            )
        
        return analysis
    
    def _analyze_classes(self, image_paths: List[str]) -> Dict[str, int]:
        """Analyze class distribution (from folder structure)."""
        classes = {}
        for path in image_paths:
            p = Path(path)
            # Use parent directory as class
            cls = p.parent.name
            classes[cls] = classes.get(cls, 0) + 1
        return classes
    
    def _analyze_resolutions(self, image_paths: List[str]) -> Dict:
        """Analyze resolution distribution."""
        widths = []
        heights = []
        megapixels = []
        
        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    megapixels.append((w * h) / 1_000_000)
            except Exception:
                pass
        
        if not widths:
            return {}
        
        return {
            "width": {
                "min": min(widths),
                "max": max(widths),
                "mean": np.mean(widths),
                "median": np.median(widths),
                "std": np.std(widths)
            },
            "height": {
                "min": min(heights),
                "max": max(heights),
                "mean": np.mean(heights),
                "median": np.median(heights),
                "std": np.std(heights)
            },
            "megapixels": {
                "min": min(megapixels),
                "max": max(megapixels),
                "mean": np.mean(megapixels),
                "median": np.median(megapixels),
                "std": np.std(megapixels)
            },
            "common_resolutions": pd.Series(
                [f"{w}x{h}" for w, h in zip(widths, heights)]
            ).value_counts().head(10).to_dict()
        }
    
    def _analyze_aspect_ratios(self, image_paths: List[str]) -> Dict:
        """Analyze aspect ratio distribution."""
        ratios = []
        
        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    if h > 0:
                        ratios.append(w / h)
            except Exception:
                pass
        
        if not ratios:
            return {}
        
        return {
            "min": min(ratios),
            "max": max(ratios),
            "mean": np.mean(ratios),
            "median": np.median(ratios),
            "std": np.std(ratios),
            "distribution": np.histogram(ratios, bins=20)[0].tolist(),
            "bin_edges": np.histogram(ratios, bins=20)[1].tolist()
        }
    
    def _analyze_file_sizes(self, image_paths: List[str]) -> Dict:
        """Analyze file size distribution."""
        sizes = []
        
        for path in image_paths:
            try:
                sizes.append(Path(path).stat().st_size)
            except Exception:
                pass
        
        if not sizes:
            return {}
        
        sizes_kb = [s / 1024 for s in sizes]
        
        return {
            "min_kb": min(sizes_kb),
            "max_kb": max(sizes_kb),
            "mean_kb": np.mean(sizes_kb),
            "median_kb": np.median(sizes_kb),
            "std_kb": np.std(sizes_kb),
            "total_mb": sum(sizes) / (1024 * 1024)
        }
    
    def _analyze_quality(self, quality_df: pd.DataFrame) -> Dict:
        """Analyze quality metrics from Phase 3."""
        if quality_df.empty:
            return {}
        
        stats = {}
        
        for col in ['blur_score', 'sharpness_score', 'brightness_score', 
                    'contrast_score', 'entropy', 'noise_estimation']:
            if col in quality_df.columns:
                data = quality_df[col].dropna()
                if len(data) > 0:
                    stats[col] = {
                        "min": float(data.min()),
                        "max": float(data.max()),
                        "mean": float(data.mean()),
                        "median": float(data.median()),
                        "std": float(data.std())
                    }
        
        return stats
    
    def _generate_visualizations(
        self,
        image_paths: List[str],
        dataset_name: str,
        analysis: DatasetAnalysis,
        quality_df: Optional[pd.DataFrame]
    ):
        """Generate visualization plots."""
        plt.style.use(self.style)
        
        output_dir = Path(self.config.output.base_dir) / self.config.output.structure.reports / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Resolution distribution
        self._plot_resolution_distribution(analysis, output_dir / f"{dataset_name}_resolution.png")
        
        # 2. Aspect ratio distribution
        self._plot_aspect_ratio_distribution(analysis, output_dir / f"{dataset_name}_aspect_ratio.png")
        
        # 3. File size distribution
        self._plot_file_size_distribution(analysis, output_dir / f"{dataset_name}_file_size.png")
        
        # 4. Class distribution
        if analysis.class_distribution:
            self._plot_class_distribution(analysis, output_dir / f"{dataset_name}_class_dist.png")
        
        # 5. Quality metrics (if available)
        if quality_df is not None and not quality_df.empty:
            self._plot_quality_metrics(quality_df, output_dir / f"{dataset_name}_quality.png")
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def _plot_resolution_distribution(self, analysis: DatasetAnalysis, save_path: Path):
        """Plot resolution distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract common resolutions
        res_data = analysis.resolution_stats.get("common_resolutions", {})
        if res_data:
            res_names = list(res_data.keys())[:10]
            res_counts = list(res_data.values())[:10]
            
            axes[0].barh(range(len(res_names)), res_counts)
            axes[0].set_yticks(range(len(res_names)))
            axes[0].set_yticklabels(res_names)
            axes[0].set_xlabel('Count')
            axes[0].set_title('Top 10 Resolutions')
            axes[0].invert_yaxis()
        
        # Megapixels histogram
        # Need to re-read or use stored data
        axes[1].text(0.5, 0.5, 'Resolution stats available\nin JSON report', 
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Megapixels')
        
        fig.suptitle(f'{analysis.dataset_name} - Resolution Analysis')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_aspect_ratio_distribution(self, analysis: DatasetAnalysis, save_path: Path):
        """Plot aspect ratio distribution."""
        ar = analysis.aspect_ratio_stats
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if 'distribution' in ar:
            counts = ar['distribution']
            edges = ar['bin_edges']
            centers = [(edges[i] + edges[i+1])/2 for i in range(len(counts))]
            
            ax.bar(centers, counts, width=centers[1]-centers[0] if len(centers)>1 else 0.1, 
                   edgecolor='black', alpha=0.7)
        
        ax.axvline(ar.get('mean', 0), color='red', linestyle='--', 
                   label=f"Mean: {ar.get('mean', 0):.2f}")
        ax.axvline(ar.get('median', 0), color='green', linestyle='--', 
                   label=f"Median: {ar.get('median', 0):.2f}")
        
        ax.set_xlabel('Aspect Ratio (Width/Height)')
        ax.set_ylabel('Count')
        ax.set_title(f'{analysis.dataset_name} - Aspect Ratio Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_file_size_distribution(self, analysis: DatasetAnalysis, save_path: Path):
        """Plot file size distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Text summary since we don't have raw data here
        stats = analysis.file_size_stats
        text = f"""File Size Statistics:
Min: {stats.get('min_kb', 0):.1f} KB
Max: {stats.get('max_kb', 0):.1f} KB
Mean: {stats.get('mean_kb', 0):.1f} KB
Median: {stats.get('median_kb', 0):.1f} KB
Std: {stats.get('std_kb', 0):.1f} KB
Total: {stats.get('total_mb', 0):.2f} MB"""
        
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace')
        ax.set_title(f'{analysis.dataset_name} - File Size Statistics')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_class_distribution(self, analysis: DatasetAnalysis, save_path: Path):
        """Plot class distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        classes = list(analysis.class_distribution.keys())
        counts = list(analysis.class_distribution.values())
        
        ax.bar(classes, counts, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title(f'{analysis.dataset_name} - Class Distribution')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_metrics(self, quality_df: pd.DataFrame, save_path: Path):
        """Plot quality metrics distributions."""
        metrics = ['blur_score', 'sharpness_score', 'brightness_score', 
                   'contrast_score', 'entropy', 'noise_estimation']
        available = [m for m in metrics if m in quality_df.columns]
        
        if not available:
            return
        
        n = len(available)
        cols = 3
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flatten() if n > 1 else [axes]
        
        for i, metric in enumerate(available):
            ax = axes[i]
            data = quality_df[metric].dropna()
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(data.mean(), color='red', linestyle='--', 
                       label=f"Mean: {data.mean():.1f}")
            ax.axvline(data.median(), color='green', linestyle='--', 
                       label=f"Median: {data.median():.1f}")
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Count')
            ax.set_title(metric.replace('_', ' ').title())
            ax.legend()
        
        # Hide unused subplots
        for j in range(n, len(axes)):
            axes[j].axis('off')
        
        fig.suptitle('Image Quality Metrics Distribution')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def save_analysis(self, analysis: DatasetAnalysis, output_path: str):
        """Save analysis results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(asdict(analysis), f, indent=2, default=str)
        
        # CSV summary
        df = pd.DataFrame([{
            "dataset": analysis.dataset_name,
            "image_count": analysis.image_count,
            **{f"class_{k}": v for k, v in analysis.class_distribution.items()},
            "mean_width": analysis.resolution_stats.get("width", {}).get("mean"),
            "mean_height": analysis.resolution_stats.get("height", {}).get("mean"),
            "mean_megapixels": analysis.resolution_stats.get("megapixels", {}).get("mean"),
            "mean_aspect_ratio": analysis.aspect_ratio_stats.get("mean"),
            "mean_file_size_kb": analysis.file_size_stats.get("mean_kb"),
        }])
        df.to_csv(output_path.with_suffix('.csv'), index=False)
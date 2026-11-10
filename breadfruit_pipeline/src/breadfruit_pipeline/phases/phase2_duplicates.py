"""
Phase 2: Duplicate Detection
Detects duplicate images using multiple perceptual hash techniques.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm import tqdm
import logging
import json
import pandas as pd

from breadfruit_pipeline.core.config import get_config
from breadfruit_pipeline.core.exceptions import DatasetError
from breadfruit_pipeline.utils.hash_utils import (
    compute_all_hashes,
    hamming_distance,
    find_duplicate_groups,
    find_duplicate_groups_multi
)
from breadfruit_pipeline.utils.file_utils import safe_move, ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class DuplicatePair:
    """A pair of duplicate images."""
    image1: str
    image2: str
    phash_distance: int
    dhash_distance: int
    ahash_distance: int
    whash_distance: int
    colorhash_distance: int
    consensus_votes: int
    method: str


@dataclass
class DuplicateGroup:
    """A group of duplicate images."""
    images: List[str]
    representative: str
    method: str
    max_distance: int


@dataclass
class DuplicateReport:
    """Complete duplicate detection report."""
    dataset_name: str
    total_images: int
    duplicate_pairs: int
    duplicate_groups: int
    images_flagged: int
    pairs: List[DuplicatePair]
    groups: List[DuplicateGroup]


class DuplicateDetector:
    """Phase 2: Duplicate Detection using multiple hash methods."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.dup_config = self.config.phase2_duplicates
        self.methods = self.dup_config.methods
        self.thresholds = self.dup_config.thresholds
        self.thresholds_dict = vars(self.thresholds)
        self.consensus = self.dup_config.consensus_threshold
        
    def detect_duplicates(
        self,
        image_paths: List[str],
        dataset_name: str,
        rejected_dir: str
    ) -> DuplicateReport:
        """Detect duplicates in a list of images."""
        logger.info(f"Detecting duplicates in {dataset_name} ({len(image_paths)} images)")
        
        # Compute all hashes
        hash_dicts = self._compute_all_hashes(image_paths)
        
        # Find duplicates using consensus
        groups = find_duplicate_groups_multi(
            hash_dicts,
            self.thresholds_dict,
            self.consensus
        )
        
        # Generate pairs for detailed report
        pairs = self._generate_pairs(groups, hash_dicts)
        
        # Create report
        report = DuplicateReport(
            dataset_name=dataset_name,
            total_images=len(image_paths),
            duplicate_pairs=len(pairs),
            duplicate_groups=len(groups),
            images_flagged=sum(len(g) - 1 for g in groups),
            pairs=pairs,
            groups=[
                DuplicateGroup(
                    images=g,
                    representative=g[0],
                    method="consensus",
                    max_distance=self._max_group_distance(g, hash_dicts)
                )
                for g in groups
            ]
        )
        
        # Move duplicates to rejected folder
        if self.dup_config.move_duplicates:
            self._move_duplicates(groups, rejected_dir, dataset_name)
        
        logger.info(f"Found {len(groups)} duplicate groups, {len(pairs)} pairs")
        return report
    
    def _compute_all_hashes(self, image_paths: List[str]) -> Dict[str, Dict[str, str]]:
        """Compute all hash types for all images."""
        hash_dicts = defaultdict(dict)
        
        for path in tqdm(image_paths, desc="Computing hashes"):
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                hashes = compute_all_hashes(img)
                for method, hash_val in hashes.items():
                    hash_dicts[method][path] = hash_val
            except Exception as e:
                logger.warning(f"Failed to compute hashes for {path}: {e}")
        
        return dict(hash_dicts)
    
    def _generate_pairs(
        self,
        groups: List[List[str]],
        hash_dicts: Dict[str, Dict[str, str]]
    ) -> List[DuplicatePair]:
        """Generate detailed pair information from groups."""
        pairs = []
        
        for group in groups:
            rep = group[0]
            for img in group[1:]:
                pair = DuplicatePair(
                    image1=rep,
                    image2=img,
                    phash_distance=hamming_distance(
                        hash_dicts.get("phash", {}).get(rep, ""),
                        hash_dicts.get("phash", {}).get(img, "")
                    ),
                    dhash_distance=hamming_distance(
                        hash_dicts.get("dhash", {}).get(rep, ""),
                        hash_dicts.get("dhash", {}).get(img, "")
                    ),
                    ahash_distance=hamming_distance(
                        hash_dicts.get("ahash", {}).get(rep, ""),
                        hash_dicts.get("ahash", {}).get(img, "")
                    ),
                    whash_distance=hamming_distance(
                        hash_dicts.get("whash", {}).get(rep, ""),
                        hash_dicts.get("whash", {}).get(img, "")
                    ),
                    colorhash_distance=hamming_distance(
                        hash_dicts.get("colorhash", {}).get(rep, ""),
                        hash_dicts.get("colorhash", {}).get(img, "")
                    ),
                    consensus_votes=sum([
                        hamming_distance(
                            hash_dicts.get(m, {}).get(rep, ""),
                            hash_dicts.get(m, {}).get(img, "")
                        ) <= self.thresholds_dict.get(m, 10)
                        for m in ["phash", "dhash", "ahash", "whash", "colorhash"]
                    ]),
                    method="consensus"
                )
                pairs.append(pair)
        
        return pairs
    
    def _max_group_distance(
        self,
        group: List[str],
        hash_dicts: Dict[str, Dict[str, str]]
    ) -> int:
        """Find maximum distance within a group."""
        max_dist = 0
        for i, p1 in enumerate(group):
            for p2 in group[i+1:]:
                for method in ["phash", "dhash", "ahash", "whash"]:
                    if p1 in hash_dicts.get(method, {}) and p2 in hash_dicts.get(method, {}):
                        dist = hamming_distance(
                            hash_dicts[method][p1],
                            hash_dicts[method][p2]
                        )
                        max_dist = max(max_dist, dist)
        return max_dist
    
    def _move_duplicates(
        self,
        groups: List[List[str]],
        rejected_dir: str,
        dataset_name: str
    ):
        """Move duplicate images to rejected folder."""
        dup_dir = Path(rejected_dir) / "Duplicates" / dataset_name
        ensure_dir(dup_dir)
        
        for group in groups:
            # Keep first image, move rest
            for img_path in group[1:]:
                src = Path(img_path)
                dst = dup_dir / src.name
                # Handle name conflicts
                if dst.exists():
                    stem = src.stem
                    suffix = src.suffix
                    counter = 1
                    while dst.exists():
                        dst = dup_dir / f"{stem}_dup{counter}{suffix}"
                        counter += 1
                
                if safe_move(src, dst):
                    logger.debug(f"Moved duplicate: {src} -> {dst}")
    
    def save_report(self, report: DuplicateReport, output_path: str):
        """Save duplicate report to JSON and CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Save pairs as CSV
        csv_path = output_path.with_suffix('.csv')
        df = pd.DataFrame([asdict(p) for p in report.pairs])
        df.to_csv(csv_path, index=False)
        
        # Save summary
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write(self._format_summary(report))
        
        logger.info(f"Duplicate report saved to {output_path.parent}")
    
    def _format_summary(self, report: DuplicateReport) -> str:
        """Format report as text summary."""
        lines = [
            f"=" * 60,
            f"DUPLICATE DETECTION REPORT: {report.dataset_name}",
            f"=" * 60,
            f"",
            f"Total images scanned:    {report.total_images}",
            f"Duplicate groups found:  {report.duplicate_groups}",
            f"Duplicate pairs found:   {report.duplicate_pairs}",
            f"Images flagged as dup:   {report.images_flagged}",
            f"",
            f"Groups:",
        ]
        
        for i, group in enumerate(report.groups):
            lines.append(f"  Group {i+1}: {len(group.images)} images")
            for img in group.images:
                lines.append(f"    - {Path(img).name}")
        
        return "\n".join(lines)
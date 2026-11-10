"""
Phase 6: Dataset Organization
Organizes datasets into the final Breadfruit_Dataset structure.
"""
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
import json

from breadfruit_pipeline.core.config import get_config
from breadfruit_pipeline.utils.file_utils import ensure_dir, safe_move, safe_copy

logger = logging.getLogger(__name__)


@dataclass
class OrganizationResult:
    """Result of dataset organization."""
    dataset_name: str
    clean_images: int
    rejected_images: Dict[str, int]
    structure_created: bool


class DatasetOrganizer:
    """Phase 6: Dataset Organization."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.output_config = self.config.output
        self.base_dir = Path(self.output_config.base_dir)
        self.structure = self.output_config.structure
    
    def organize_datasets(
        self,
        dataset_results: Dict[str, Dict]
    ) -> Dict[str, OrganizationResult]:
        """Organize all datasets into final structure."""
        logger.info("Organizing datasets into final structure")
        
        # Create directory structure
        self._create_directory_structure()
        
        results = {}
        
        for dataset_key, data in dataset_results.items():
            result = self._organize_dataset(dataset_key, data)
            results[dataset_key] = result
        
        return results
    
    def _create_directory_structure(self):
        """Create the complete directory structure."""
        dirs_to_create = [
            self.base_dir / self.structure.personal_clean,
            self.base_dir / self.structure.public_clean,
            self.base_dir / self.structure.reports,
            self.base_dir / self.structure.logs,
        ]
        
        # Rejected subdirectories
        rejected_base = self.base_dir / self.structure.rejected.root
        for subdir in self.structure.rejected.subdirs:
            dirs_to_create.append(rejected_base / subdir)
        
        for dir_path in dirs_to_create:
            ensure_dir(dir_path)
        
        logger.info(f"Created directory structure at {self.base_dir}")
    
    def _organize_dataset(
        self,
        dataset_key: str,
        data: Dict
    ) -> OrganizationResult:
        """Organize a single dataset."""
        dataset_name = data.get("name", dataset_key)
        
        # Determine clean directory
        if dataset_key == "personal":
            clean_dir = self.base_dir / self.structure.personal_clean
        else:
            clean_dir = self.base_dir / self.structure.public_clean
        
        ensure_dir(clean_dir)
        
        # Move standardized images to clean directory
        standardized_dir = Path(self.config.output.base_dir) / "standardized" / dataset_key
        clean_count = 0
        
        if standardized_dir.exists():
            for img_path in standardized_dir.glob("*.jpg"):
                dst = clean_dir / img_path.name
                counter = 1
                while dst.exists():
                    dst = clean_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
                    counter += 1
                if safe_move(img_path, dst):
                    clean_count += 1
        
        # Count rejected images
        rejected_counts = {}
        rejected_base = self.base_dir / self.structure.rejected.root
        
        for subdir in self.structure.rejected.subdirs:
            sub_path = rejected_base / subdir / dataset_key
            if sub_path.exists():
                rejected_counts[subdir] = len(list(sub_path.glob("*")))
            else:
                rejected_counts[subdir] = 0
        
        return OrganizationResult(
            dataset_name=dataset_name,
            clean_images=clean_count,
            rejected_images=rejected_counts,
            structure_created=True
        )
    
    def generate_structure_report(self, output_path: str):
        """Generate report of final directory structure."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        structure = {}
        
        for item in self.base_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(self.base_dir)
                parts = rel.parts
                if parts[0] not in structure:
                    structure[parts[0]] = {}
                ext = item.suffix.lower()
                if ext not in structure[parts[0]]:
                    structure[parts[0]][ext] = 0
                structure[parts[0]][ext] += 1
        
        # JSON
        with open(output_path, 'w') as f:
            json.dump(structure, f, indent=2, default=str)
        
        # Human readable
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write("BREADFRUIT DATASET - FINAL STRUCTURE\n")
            f.write("=" * 50 + "\n\n")
            
            for top_dir, exts in structure.items():
                f.write(f"{top_dir}/\n")
                total = sum(exts.values())
                for ext, count in sorted(exts.items()):
                    f.write(f"  {ext}: {count}\n")
                f.write(f"  Total: {total}\n\n")
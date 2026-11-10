"""
Dataset module — loads images, applies transforms, and creates DataLoaders.
Supports multi-task classification via a CSV annotation file.
"""

import os
import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, tv_tensors
from torchvision.transforms import v2
from PIL import Image

from config import Config


class BreadfruitDataset(Dataset):
    """Multi-task dataset for breadfruit maturity and quality classification.

    Expects directory structure:
        data_root/
            train/breadfruit/  (or val/, test/)
            annotations.csv    (columns: filename, maturity, quality)

    If no annotation file is provided, falls back to single-class mode.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        class_name: str,
        annotation_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        maturity_classes: Optional[List[str]] = None,
        quality_classes: Optional[List[str]] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.class_name = class_name
        self.transform = transform
        self.target_transform = target_transform
        self.maturity_classes = maturity_classes or []
        self.quality_classes = quality_classes or []
        self.maturity_to_idx = {c: i for i, c in enumerate(self.maturity_classes)}
        self.quality_to_idx = {c: i for i, c in enumerate(self.quality_classes)}
        self.has_multi_task = bool(self.maturity_classes and self.quality_classes)
        self.use_annotations = False

        self.image_dir = self.data_root / split / class_name
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")

        self.samples: List[Tuple[str, Optional[int], Optional[int]]] = []
        self._load_samples(annotation_file)

    def _load_samples(self, annotation_file: Optional[str]) -> None:
        if annotation_file and Path(annotation_file).exists():
            lookup: Dict[str, Tuple[int, int]] = {}
            with open(annotation_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("filename", "")
                    maturity = row.get("maturity", "")
                    quality = row.get("quality", "")
                    if fname and maturity in self.maturity_to_idx and quality in self.quality_to_idx:
                        lookup[fname] = (
                            self.maturity_to_idx[maturity],
                            self.quality_to_idx[quality],
                        )

            for fname in sorted(os.listdir(self.image_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    if fname in lookup:
                        mat, qual = lookup[fname]
                        self.samples.append((fname, mat, qual))
                        self.use_annotations = True

            if not self.samples:
                print(f"Warning: no matching annotations found in {annotation_file}")
                print(f"Falling back to single-class mode for {self.split}")
        else:
            print(f"No annotation file provided or file not found at: {annotation_file}")
            print(f"Running in single-class mode for {self.split}")

        if not self.samples:
            for fname in sorted(os.listdir(self.image_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((fname, None, None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, mat_label, qual_label = self.samples[idx]
        img_path = self.image_dir / fname
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.use_annotations and self.has_multi_task:
            mat_tensor = torch.tensor(mat_label, dtype=torch.long)
            qual_tensor = torch.tensor(qual_label, dtype=torch.long)
            if self.target_transform:
                mat_tensor, qual_tensor = self.target_transform(mat_tensor, qual_tensor)
            return image, {"maturity": mat_tensor, "quality": qual_tensor}

        return image, 0

    @property
    def num_maturity_classes(self) -> int:
        return len(self.maturity_classes) if self.maturity_classes else 0

    @property
    def num_quality_classes(self) -> int:
        return len(self.quality_classes) if self.quality_classes else 0


def get_transforms(config: Config, is_train: bool) -> transforms.Compose:
    """Build image transform pipeline for train or validation/test."""
    img_size = config.data.image_size

    if is_train and config.augmentation.train_augmentation:
        aug = config.augmentation
        pipeline = [
            transforms.Resize((int(img_size[0] * 1.1), int(img_size[1] * 1.1))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=aug.random_horizontal_flip_prob),
            transforms.RandomVerticalFlip(p=aug.random_vertical_flip_prob),
            transforms.RandomRotation(degrees=aug.random_rotation_degrees),
            transforms.ColorJitter(
                brightness=aug.color_jitter_brightness,
                contrast=aug.color_jitter_contrast,
                saturation=aug.color_jitter_saturation,
                hue=aug.color_jitter_hue,
            ),
            transforms.RandomAffine(
                degrees=aug.random_affine_degrees,
                translate=aug.random_affine_translate,
                scale=aug.random_affine_scale,
                shear=aug.random_affine_shear,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.mean, std=config.data.std),
            transforms.RandomErasing(
                p=aug.random_erasing_prob,
                scale=aug.random_erasing_scale,
            ),
        ]
    else:
        pipeline = [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.mean, std=config.data.std),
        ]

    return transforms.Compose(pipeline)


def get_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, validation, and optional test DataLoaders."""

    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)

    train_dataset = BreadfruitDataset(
        data_root=config.data.data_root,
        split=config.data.train_dir,
        class_name=config.data.class_name,
        annotation_file=config.data.annotation_file,
        transform=train_transform,
        maturity_classes=config.model.maturity_classes,
        quality_classes=config.model.quality_classes,
    )

    val_dataset = BreadfruitDataset(
        data_root=config.data.data_root,
        split=config.data.val_dir,
        class_name=config.data.class_name,
        annotation_file=config.data.annotation_file,
        transform=val_transform,
        maturity_classes=config.model.maturity_classes,
        quality_classes=config.model.quality_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    test_loader = None
    test_dir = config.data.test_dir
    if test_dir:
        test_dataset = BreadfruitDataset(
            data_root=config.data.data_root,
            split=test_dir,
            class_name=config.data.class_name,
            annotation_file=config.data.annotation_file,
            transform=val_transform,
            maturity_classes=config.model.maturity_classes,
            quality_classes=config.model.quality_classes,
        )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory,
            )

    return train_loader, val_loader, test_loader


def get_inference_transform(config: Config) -> transforms.Compose:
    """Minimal transform for inference on a single image."""
    return transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.CenterCrop(config.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.data.mean, std=config.data.std),
    ])

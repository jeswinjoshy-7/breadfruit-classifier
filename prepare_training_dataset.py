#!/usr/bin/env python3
"""
Prepares a balanced training dataset by:
1. Augmenting the small personal dataset (17 → ~340 images)
2. Combining with the public dataset (501 images)
3. Splitting into train/val/test (70/15/15)
4. Organizing into class-labeled folders for PyTorch/TF/Keras
"""
import os
import cv2
import numpy as np
import random
import json
import shutil
from pathlib import Path
from collections import Counter

random.seed(42)
np.random.seed(42)

PERSONAL_DIR = "/home/jeswin/data_augmentation/Breadfruit_Dataset/Personal_Clean"
PUBLIC_DIR = "/home/jeswin/data_augmentation/Breadfruit_Dataset/Public_Clean"
OUTPUT_DIR = "/home/jeswin/data_augmentation/training_dataset"
AUGS_PER_PERSONAL = 20
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
CLASS_NAME = "breadfruit"


def load_images(directory):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(Path(directory).glob(ext))
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            images.append((str(p.name), img))
    return images


def augment_image(img):
    h, w = img.shape[:2]
    augs = []

    # 1. Horizontal flip
    augs.append(cv2.flip(img, 1))

    # 2. Rotation ±15°
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        augs.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT))

    # 3. Brightness/contrast variations
    for alpha, beta in [(0.8, 10), (0.9, 5), (1.1, -5), (1.2, -10)]:
        augs.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))

    # 4. HSV shifts
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    for h_shift, s_scale in [(0, 0.8), (0, 1.2), (5, 0.9), (-5, 1.1)]:
        hsv_aug = hsv.copy()
        hsv_aug[:,:,0] = (hsv_aug[:,:,0] + h_shift) % 180
        hsv_aug[:,:,1] = np.clip(hsv_aug[:,:,1] * s_scale, 0, 255)
        augs.append(cv2.cvtColor(hsv_aug.astype(np.uint8), cv2.COLOR_HSV2BGR))

    # 5. Gaussian blur
    for k in [3, 5]:
        augs.append(cv2.GaussianBlur(img, (k, k), 0))

    # 6. Gaussian noise
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    augs.append(np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8))
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    augs.append(np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8))

    # 7. Random crop + resize back
    for scale in [0.85, 0.9, 0.95]:
        crop_h, crop_w = int(h * scale), int(w * scale)
        y1 = random.randint(0, h - crop_h)
        x1 = random.randint(0, w - crop_w)
        cropped = img[y1:y1+crop_h, x1:x1+crop_w]
        augs.append(cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR))

    # 8. Gamma correction
    for gamma in [0.7, 0.85, 1.15, 1.3]:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        augs.append(cv2.LUT(img, table))

    # 9. Translation shifts
    for tx_pct, ty_pct in [(0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)]:
        tx, ty = tx_pct * w, ty_pct * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        augs.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT))

    # 10. Shadow simulation (simplified)
    for intensity in [0.2, 0.4]:
        shadow = img.copy().astype(np.float32)
        sx = random.randint(0, w - 1)
        sy = random.randint(0, h - 1)
        sw, sh = w // 3, h // 3
        x1 = max(0, sx - sw//2)
        x2 = min(w, sx + sw//2)
        y1 = max(0, sy - sh//2)
        y2 = min(h, sy + sh//2)
        shadow[y1:y2, x1:x2] *= (1 - intensity)
        augs.append(np.clip(shadow, 0, 255).astype(np.uint8))

    return augs


def save_image(img, path):
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    print("Loading standardized images...")
    personal = load_images(PERSONAL_DIR)
    public = load_images(PUBLIC_DIR)
    print(f"  Personal: {len(personal)} images")
    print(f"  Public:   {len(public)} images")

    # Augment personal images
    print(f"\nAugmenting personal images ({AUGS_PER_PERSONAL} variants each)...")
    augmented_personal = []
    for name, img in personal:
        variants = augment_image(img)
        selected = random.sample(variants, min(AUGS_PER_PERSONAL, len(variants)))
        for i, aug_img in enumerate(selected):
            aug_name = f"personal_{Path(name).stem}_aug{i:03d}.jpg"
            augmented_personal.append((aug_name, aug_img))
    print(f"  → {len(augmented_personal)} augmented personal images")

    # Keep public images mostly clean, add a few basic augs for variety
    print(f"\nAdding public images with light augmentation...")
    all_public = []
    for name, img in public:
        all_public.append((f"public_{Path(name).stem}.jpg", img))
        all_public.append((f"public_{Path(name).stem}_flip.jpg", cv2.flip(img, 1)))
    print(f"  → {len(all_public)} public images (originals + flips)")

    # Combine all images
    all_images = augmented_personal + all_public
    random.shuffle(all_images)
    print(f"\nTotal combined dataset: {len(all_images)} images")

    # Split into train/val/test
    n = len(all_images)
    n_test = int(n * TEST_SPLIT)
    n_val = int(n * VAL_SPLIT)
    n_train = n - n_val - n_test

    test_set = all_images[:n_test]
    val_set = all_images[n_test:n_test + n_val]
    train_set = all_images[n_test + n_val:]

    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")

    # Save to disk
    print(f"\nSaving to {OUTPUT_DIR}/")
    for split_name, split_data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        class_dir = Path(OUTPUT_DIR) / split_name / CLASS_NAME
        class_dir.mkdir(parents=True, exist_ok=True)
        for fname, img in split_data:
            save_image(img, class_dir / fname)

    # Summary
    print(f"\nFinal dataset structure:")
    for split_name in ["train", "val", "test"]:
        class_dir = Path(OUTPUT_DIR) / split_name / CLASS_NAME
        count = len(list(class_dir.glob("*.jpg")))
        print(f"  {split_name}/{CLASS_NAME}/: {count} images")

    # Generate metadata
    metadata = {
        "dataset": "Breadfruit Training Dataset",
        "class": CLASS_NAME,
        "personal_original": len(personal),
        "public_original": len(public),
        "augmented_personal": len(augmented_personal),
        "augmented_public": len(all_public),
        "total": len(all_images),
        "splits": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set)
        },
        "augmentations_per_personal": AUGS_PER_PERSONAL,
        "input_size": "224x224 RGB JPEG"
    }
    with open(Path(OUTPUT_DIR) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved. Ready for model training!")


if __name__ == "__main__":
    main()

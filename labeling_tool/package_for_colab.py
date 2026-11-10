"""
Packages classifier code + training dataset + annotations for Colab.

Steps:
    1. Converts dataset.csv (Label Studio export) → annotations.csv
    2. Propagates labels to augmented image variants
    3. Zips classifier code (with annotations) and training dataset

Usage:
    python labeling_tool/package_for_colab.py

Then upload the two ZIP files to Google Drive and run the Colab notebook.
"""
import csv
import re
import zipfile
from pathlib import Path

BASE = Path(__file__).parent.parent
LABELSTUDIO_CSV = BASE / "dataset.csv"
CLASSIFIER_DIR = BASE / "breadfruit_classifier"
DATASET_DIR = BASE / "training_dataset"
ANNOTATIONS_CSV = CLASSIFIER_DIR / "annotations.csv"


def convert_dataset_csv():
    """Convert Label Studio CSV to simple annotations CSV (filename, maturity, quality)."""
    rows = []
    with open(LABELSTUDIO_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row["image"]
            filename = re.sub(r"^.*/data/upload/\d+/[a-f0-9]+-", "", image_path)
            rows.append({
                "filename": filename,
                "maturity": row["maturity"],
                "quality": row["quality"],
            })
    return rows


def propagate_labels(orig_labels):
    """Propagate labels from original images to all augmented variants in the dataset."""
    aug_pattern = re.compile(r'_(aug\d+|flip|rot|crop|noise|blur|shadow|elastic|gamma|zoom|jitter|trans)+')

    def get_base_name(fname):
        return aug_pattern.sub("", re.sub(r"\.(jpg|jpeg|png)$", ".jpg", fname))

    orig = {r["filename"]: (r["maturity"], r["quality"]) for r in orig_labels}
    all_labels = dict(orig)

    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / split / "breadfruit"
        if not img_dir.exists():
            continue
        for f in sorted(img_dir.iterdir()):
            base = get_base_name(f.name)
            if base in orig and f.name not in all_labels:
                all_labels[f.name] = orig[base]

    return all_labels


def save_annotations(labels_dict, output_path):
    """Save annotations dict to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "maturity", "quality"])
        for fname in sorted(labels_dict.keys()):
            mat, qual = labels_dict[fname]
            writer.writerow([fname, mat, qual])
    print(f"Saved {len(labels_dict)} annotations to {output_path}")


def main():
    print("=" * 60)
    print("  PACKAGING FOR COLAB")
    print("=" * 60)

    # Step 1: Convert Label Studio CSV
    print("\n[1/4] Converting Label Studio CSV...")
    if not LABELSTUDIO_CSV.exists():
        print(f"  ERROR: {LABELSTUDIO_CSV} not found!")
        return
    orig_labels = convert_dataset_csv()
    print(f"  Converted {len(orig_labels)} labels")

    # Step 2: Propagate to augmented variants
    print("\n[2/4] Propagating labels to augmented variants...")
    all_labels = propagate_labels(orig_labels)
    total_images = sum(1 for _ in DATASET_DIR.rglob("*") if _.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"  Total images in dataset: {total_images}")
    print(f"  Labeled after propagation: {len(all_labels)}")
    print(f"  Still unlabeled: {total_images - len(all_labels)}")

    # Step 3: Save annotations into classifier dir
    print("\n[3/4] Saving annotations.csv...")
    save_annotations(all_labels, ANNOTATIONS_CSV)

    # Step 4: Create ZIP files
    print("\n[4/4] Creating ZIP files...")

    code_zip = BASE / "breadfruit_classifier.zip"
    with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(CLASSIFIER_DIR.rglob("*")):
            if "__pycache__" not in f.parts and f.suffix not in (".pyc", ".zip") and not f.is_symlink():
                z.write(f, f.relative_to(BASE))
    print(f"  Created: {code_zip.name} ({code_zip.stat().st_size / 1024:.1f} KB)")

    data_zip = BASE / "training_dataset.zip"
    with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(DATASET_DIR.rglob("*")):
            if f.is_file() and not f.is_symlink():
                z.write(f, f.relative_to(BASE))
    print(f"  Created: {data_zip.name} ({data_zip.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\n{'=' * 60}")
    print("  Done! Upload these two ZIP files to Google Drive.")
    print("  Then run the Colab notebook (Breadfruit_Colab_Training.ipynb).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

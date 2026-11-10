"""Propagates labels from original images to all augmented variants."""
import csv
import re
import shutil
from pathlib import Path

AUG_PATTERN = re.compile(r'_(aug\d+|flip|rot|crop|noise|blur|shadow|elastic|gamma|zoom|jitter|trans)+')
DATA_ROOT = Path(__file__).parent.parent / "training_dataset"
ANNOTATIONS_CSV = Path(__file__).parent / "annotations.csv"
OUTPUT_CSV = Path(__file__).parent.parent / "breadfruit_classifier" / "annotations.csv"

def get_base_name(filename):
    return AUG_PATTERN.sub("", re.sub(r"\.(jpg|jpeg|png)$", ".jpg", filename))

def main():
    if not ANNOTATIONS_CSV.exists():
        print(f"❌ annotations.csv not found at {ANNOTATIONS_CSV}")
        print("   First run: python generate_template.py")
        print("   Then label the images, then run this script.")
        return

    orig_labels = {}
    with open(ANNOTATIONS_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("maturity") and row.get("quality"):
                orig_labels[row["filename"]] = (row["maturity"], row["quality"])

    print(f"Loaded {len(orig_labels)} original labels")

    all_labels = dict(orig_labels)
    total_files = 0

    for split in ["train", "val", "test"]:
        img_dir = DATA_ROOT / split / "breadfruit"
        for f in sorted(img_dir.glob("*")):
            total_files += 1
            base = get_base_name(f.name)
            if base in orig_labels and f.name not in all_labels:
                all_labels[f.name] = orig_labels[base]

    print(f"Total files: {total_files}")
    print(f"Labeled files after propagation: {len(all_labels)}")

    # Save full annotation file
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "maturity", "quality"])
        writer.writeheader()
        for fname in sorted(all_labels.keys()):
            mat, qual = all_labels[fname]
            writer.writerow({"filename": fname, "maturity": mat, "quality": qual})

    unlabeled = total_files - len(all_labels)
    print(f"\n✅ Saved annotations for {len(all_labels)} images to {OUTPUT_CSV}")
    print(f"⚠️  {unlabeled} images still unlabeled (originals you haven't annotated yet)")
    print(f"\nCopy this CSV to Colab along with the classifier code.")

if __name__ == "__main__":
    main()

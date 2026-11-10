"""Scans dataset, extracts original images, generates labeling CSV template."""
import re
import csv
from pathlib import Path

AUG_PATTERN = re.compile(r'(aug|flip|rot|crop|noise|blur|shadow|elastic|gamma|zoom|jitter|trans)')
DATA_ROOT = Path(__file__).parent.parent / "training_dataset"
OUTPUT_CSV = Path(__file__).parent / "annotations_template.csv"

rows = []
for split in ["train", "val", "test"]:
    img_dir = DATA_ROOT / split / "breadfruit"
    for f in sorted(img_dir.glob("*")):
        if not AUG_PATTERN.search(f.name):
            rows.append({"filename": f.name, "split": split, "maturity": "", "quality": ""})

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "split", "maturity", "quality"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {OUTPUT_CSV} with {len(rows)} images to label")
print(f"\nLabel options:")
print(f"  maturity: {['immature', 'partially_mature', 'mature']}")
print(f"  quality:  {['poor', 'good', 'excellent']}")
print(f"\nOpen the CSV in Excel/Google Sheets, fill in the blanks, then save as annotations.csv")

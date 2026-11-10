"""Converts Label Studio CSV export to the format our pipeline expects."""
import csv
import re
from pathlib import Path

LABELSTUDIO_CSV = Path(__file__).parent.parent / "project-1-at-2026-07-11-12-18-ebbfe1d6.csv"
OUTPUT_CSV = Path(__file__).parent / "annotations.csv"

rows = []
with open(LABELSTUDIO_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Extract filename from "/data/upload/1/XXXXXXXX-public_0.jpg" -> "public_0.jpg"
        image_path = row["image"]
        filename = re.sub(r"^.*/data/upload/\d+/[a-f0-9]+-", "", image_path)
        rows.append({
            "filename": filename,
            "maturity": row["maturity"],
            "quality": row["quality"],
        })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "maturity", "quality"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Converted {len(rows)} labels from Label Studio format")
print(f"Saved to {OUTPUT_CSV}")
print("\nSample:")
for r in rows[:3]:
    print(f"  {r['filename']} → maturity={r['maturity']}, quality={r['quality']}")

# DELIVERABLE 6: Database Design, Data Dictionary, and ER Diagram

## 6.1 Database Technology
The project does **not** use a traditional relational database. Instead, it uses **file-based persistence**:

- **Image storage**: Directory structure with train/val/test splits
- **Annotations**: CSV file (`breadfruit_classifier/annotations.csv`) with 540 entries (270 original + 270 propagated)
- **Model checkpoints**: PyTorch `.pth` files (`breadfruit_model/best_model.pth`, checkpoints/)
- **Training history**: JSON file (`breadfruit_model/training_history.json`)
- **Pipeline reports**: CSV and TXT files in `breadfruit_pipeline/output/reports/`
- **Pipeline metadata**: `metadata.json` in output directories

## 6.2 Data Dictionary

| Field Name | Description | Data Type | Source |
|---|---|---|---|
| `filename` | Image filename (e.g., "01.JPG") | String | Annotation CSV / image directory |
| `maturity` | Maturity class label | String: "immature", "partially_mature", "mature" | Annotation CSV (manually labeled) |
| `quality` | Quality grade label | String: "poor", "good", "excellent" | Annotation CSV (manually labeled) |
| `maturity_idx` | Maturity class index | Integer: 0, 1, 2 | Derived from maturity_to_idx mapping |
| `quality_idx` | Quality class index | Integer: 0, 1, 2 | Derived from quality_to_idx mapping |
| `image_path` | Full path to image file | String | Constructed from data_root/split/breadfruit/fname |
| `image_size` | Actual image dimensions | Tuple (W, H) | Read via PIL Image.open() |
| `file_hash` | SHA-256 checksum | String | Pipeline phase 1 inspection |
| `is_corrupted` | File integrity flag | Boolean | Pipeline phase 1 inspection |
| `is_zero_byte` | Empty file flag | Boolean | Pipeline phase 1 inspection |

## 6.3 Entity-Relationship Diagram

Since the project uses file-based persistence without a relational database, a traditional ER diagram with tables, columns, and foreign keys is not applicable. However, the **data relationships** can be described:

### 6.3.1 Relationships

**Annotation CSV ↔ Image files**: 
- One-to-many relationship; one annotation entry per image file (270 original images labeled, labels propagated to 540 total entries)
- Each annotation row contains: filename, maturity class, quality grade
- Filenames must match files in the associated image directory

**Dataset splits ↔ Images**: 
- One-to-many; images in train/val/test directories correspond to samples in respective DataLoader
- Each image appears in exactly one of train/val/test splits
- No image is duplicated across splits

**Config ↔ All modules**: 
- One-to-many relationship; single Config dataclass instance configures multiple modules
- Config.data → BreadfruitDataset, Config.model → BreadfruitClassifier, Config.training → train.py

### 6.3.2 Cardinalities

| Relationship | Cardinality |
|---|---|
| Annotation CSV → Images | 1 annotation : N images (270 original annotations propagated to 540 total entries) |
| Image → Dataset split | 1 image : 1 split assignment (each image in exactly one of train/val/test) |
| Config → Module | 1 Config instance : N configured modules (all modules use Config) |

### 6.4 Representative Schema Definitions

Since no relational database is used, DDL is not applicable. However, the **annotation CSV schema** can be represented:

```
filename,maturity,quality
01.JPG,immature,good
02.JPG,partially_mature,excellent
...,...,...
```

**CSV Column Details**:

| Column | Format | Example | Description |
|---|---|---|---|
| `filename` | String | "01.JPG" | Image file name (must match files in directory) |
| `maturity` | String | "immature", "partially_mature", "mature" | Maturity class label |
| `quality` | String | "poor", "good", "excellent" | Quality grade label |

### 6.5 Data Access Patterns

#### 6.5.1 Training
`BreadfruitDataset(data_root, "train", class_name, annotation_file)` → DataLoader with shuffle=True
- Reads images from `data_root/train/breadfruit/`
- Looks up labels from annotation_file CSV
- Returns (image_tensor, {"maturity": idx, "quality": idx}) per sample

#### 6.5.2 Validation
Same as training but with shuffle=False and validation transforms (no augmentation)

#### 6.5.3 Testing
Same dataset structure; test_loader may be None if test_dir not configured

#### 6.5.4 Inference (predict_single)
Uses `get_inference_transform()` only; no annotation lookup needed; image loaded directly from file path

#### 6.5.5 Pipeline Processing
`BreadfruitAugmentationPipeline.process_dataset()` → processes images through 9 phases → outputs to directory structure; no DB queries

## 6.6 Migrations and Schema Evolution

No database migrations exist since the project uses file-based persistence. Schema evolution is managed through:

- **Config dataclasses**: New fields can be added with default values; backward compatible
- **Annotation CSV**: New columns can be added; existing code ignores unknown columns (via `.get()` calls)
- **Directory structure**: Phase 6 organizes images into new splits; old directories can coexist
- **Checkpoint format**: `_save_checkpoint` and `load_checkpoint` handle epoch/val_acc/config serialization; new fields added to config don't break old checkpoints

## 6.7 How Application Modules Interact with Persistent Storage

### 6.7.1 Classifier Model
- Parameters stored in PyTorch state_dict; saved/loaded via `torch.save()`/`torch.load()`
- No direct database interaction during inference or training

### 6.7.2 Dataset Module
- Reads image files from disk via `Path` and `PIL.Image.open()`
- Reads annotation CSV via `csv.DictReader`
- No write operations during normal dataset loading

### 6.7.3 Training Pipeline
- Saves checkpoints via `_save_checkpoint()` → `torch.save(checkpoint_dict, path)`
- Loads checkpoints via `load_checkpoint()` → `torch.load(checkpoint_path, ...)`
- Saves training history as JSON via `json.dump()`
- Reads config from YAML file for pipeline operations

### 6.7.4 Pipeline Engineering
- Reads raw images from source directories
- Writes processed images to output directories (train/val/test splits)
- Writes metadata.json, phase reports, and logs
- No SQL or document database operations

### 6.7.5 Streamlit Application
- Displays images loaded from file paths
- Saves prediction visualizations to `plots/predictions/` directory
- No database operations
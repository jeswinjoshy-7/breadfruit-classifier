# DELIVERABLE 9: System Workflow Diagrams

## 9.1 Overall Application Workflow

```
┌───────────────────────────────────────────────────────────────┐
│                   BREADFRUIT CLASSIFICATION SYSTEM            │
│   (End-to-End Workflow from Data Collection to Deployment)     │
└───────────────────────────────────────────────────────────────┘

┌─────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  DATA       │      │  MODEL              │      │  WEB INTERFACE      │
│  COLLECTION │      │  TRAINING/EVALUATION│      │  (Streamlit App)    │
└──────▲──────┘      └──────▲──────────────┘      └──────▲──────────────┘
       │                    │                      │
       │                    │                      │
       ▼                    ▼                      ▼
┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│  Raw Images         │  │  Training Pipeline   │  │  Image Upload       │
│  (3dataset/)        │  │  (main.py train)     │  │  → predict_single   │
│  ~17 personal      │  │  • Two-phase TF     │  │  → results display  │
│  + 501 Kaggle      │  │  • 21 epochs max    │  │  → Grad-CAM         │
│  ─────────────────│  │  • AdamW optimizer  │  │  ────────────────── │
│  Dataset Eng.     │  │  • Early stopping  │  │  ────────────────── │
│  Pipeline          │  │  • Checkpoint save  │  │  ────────────────── │
│  (run_pipeline.py) │  │  • History & plots  │  │                       │
│  ─────────────────│  └──────────────────────┘  │                       │
│                   │                       │                       │
│                   │                       │                       │
│                   │                       │                       │
│                   │                       │                       │
│                   │                       │                       │
│                   ▼                       │                       │
│         ┌─────────────────────┐        │                       │
│         │  ORGANIZED DATASET  │        │                       │
│         │  (train/val/test)   │        │                       │
│         │  4,026 images      │        │                       │
│         └─────────────────────┘        │                       │
│                   │                       │                       │
│                   │                       │                       │
│                   └───────────────────────┘───────────────────────┘
```

### 9.1.1 Starting Event
Application initialization or user action

### 9.1.2 Actors and Components Involved
- **Operator**: User interacting with Streamlit UI or CLI commands
- **System**: PyTorch model, dataset modules, pipeline phases
- **Components**: Data directories, annotation CSV, model checkpoints, transform pipelines

### 9.1.3 Sequence of Operations
1. Data collection: Personal smartphone images + Kaggle dataset gathered
2. Dataset engineering pipeline (9 phases) → organized dataset (4,026 images in train/val/test)
3. Model training: Two-phase transfer learning on organized dataset
4. Model evaluation: Test set evaluation with metrics computation
5. Web interface: Streamlit app loaded with best model checkpoint
6. Image classification: Upload or sample → model inference → results display
7. Explainability: Grad-CAM heatmaps generated for interpretability

### 9.1.4 Decision Points
- Pipeline phase enable/disable (config.yaml per-phase flags)
- Early stopping: triggered when val accuracy doesn't improve for 10 epochs
- Backbone freezing: Phase 1 (freeze) → Phase 2 (unfreeze)
- Model selection: best_val_acc checkpoint saved and used for evaluation/prediction

### 9.1.5 Success Path
All 9 pipeline phases complete → model trained to convergence → evaluation metrics met → Streamlit app loads best checkpoint → user can classify images with Grad-CAM explanations.

### 9.1.6 Failure Path
Pipeline error in any phase (halted or continued based on `continue_on_error: true`); checkpoint not found → error message; out of memory → training halts.

### 9.1.7 Final Outcome
Either (a) organized dataset + trained model ready for deployment, or (b) error encountered with diagnostic logs.

---

## 9.2 Critical Backend Processing Workflow

```
┌─────────────────────┐    ┌─────────────────────┐
│   Training Loop     │    │   Validation Loop   │
│   (train.py)        │    │   (train.py)        │
└──────▲──────────────┘    └──────▲───────────────┘
       │                      │
       │  forward pass        │  forward pass
       │  model(images)       │  model(images)
       │  ──► loss            │  ──► loss
       │                      │
       │  backward            │  no backward
       │  loss.backward()     │
       │  ──► optimizer.step() │
       │                      │
       │  ──► gradient clip   │  ──► accuracy
       │  torch.nn.utils.clip │  ──► correctness
       │                       │
       │  ──► scheduler.step() │  ──► loss avg
       │                       │
       │  ──► epoch metrics   │  ──► epoch metrics
       └─────────────────────┘
```

### 9.2.1 Starting Event
Beginning of epoch (or validation phase)

### 9.2.2 Actors and Components
- **Model**: BreadfruitClassifier
- **Data**: Batch of (images, targets) from DataLoader
- **Optimizer**: AdamW
- **Scheduler**: ReduceLROnPlateau
- **EarlyStopping monitor**

### 9.2.3 Sequence of Operations (Training Loop)
1. Model.train() - set training mode
2. Iterate through DataLoader batches
3. Move images to device (CPU/CUDA)
4. Move targets to device; convert dict format if needed
5. Zero gradients: optimizer.zero_grad()
6. Forward pass: predictions = model(images)
7. Compute loss: criterion(predictions, targets) → MultiTaskLoss
8. Backward pass: loss.backward()
9. Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
10. Optimizer step: optimizer.step()
11. Update progress bar with running losses
12. Epoch end: compute average losses across all batches

### 9.2.4 Sequence of Operations (Validation Loop)
1. Model.eval() - set evaluation mode
2. torch.no_grad() context
3. Iterate through DataLoader batches
4. Forward pass: same as training
5. Compute loss same as training
6. Compute accuracy: compare argmax predictions to targets
7. Track per-class correctness
8. Epoch end: compute average losses and accuracies
9. Step scheduler: ReduceLROnPlateau(val_acc)
10. Check early stopping: EarlyStopping(val_acc, epoch)
11. Return metrics dict

### 9.2.5 Decision Points
- Gradient clipping threshold: max_norm=10.0 (prevents explosion)
- Scheduler patience: 5 epochs (halve LR if no improvement)
- Early stopping patience: 10 epochs (stop if no improvement)
- Best model selection: val_avg_accuracy > best_val_acc → save checkpoint

### 9.2.6 Data Exchanged
- Batch of images: tensor (batch_size, 3, 224, 224)
- Targets: dict {"maturity": tensor, "quality": tensor} or tensor (single task)
- Loss values: scalar per batch
- Accuracy: proportion of correct predictions
- Learning rate: updated by scheduler

### 9.2.7 Success Path
Epoch completes → losses and accuracies computed → scheduler updated → early stopping check → continue or stop

### 9.2.8 Failure Path
NaN loss → gradient clipping may not prevent; OOM killer → training halts; NaN in metrics → training appears to stall

### 9.2.9 Final Outcome
One epoch of training/validation completed with metrics logged; model parameters updated (training) or evaluated (validation)

---

## 9.3 Database Operations Workflow

Since the project uses file-based persistence rather than a relational database, the "database operations" workflow describes file I/O operations:

```
┌─────────────────────┐    ┌─────────────────────┐
│   Image Loading     │    │   Checkpoint Save   │
│   (datasets.py)     │    │   (train.py)        │
└──────▲──────────────┘    └──────▲──────────────┘
       │                      │
       │  PIL.Image.open()    │  torch.save(checkpoint)
       │  ──► convert RGB    │  ──► best_model.pth
       │  ──► transform      │  ──► includes model.state_dict
       │  ──► tensor         │  ──► optimizer.state_dict
       │                      │  ──► epoch, val_acc, config
       │  ──► DataLoader      │
       └─────────────────────┘
```

### 9.3.1 Starting Event
Dataset __getitem__() call or training checkpoint save

### 9.3.2 Actors and Components
- **Dataset**: BreadfruitDataset
- **File I/O**: PIL, csv, torch, json
- **Model**: BreadfruitClassifier

### 9.3.3 Sequence of Operations (Image Loading)
1. `__getitem__(idx)` called
2. `self.samples[idx]` → (fname, mat_label, qual_label)
3. `img_path = self.image_dir / fname`
4. `image = Image.open(img_path).convert("RGB")`
5. If self.transform: `image = self.transform(image)`
6. If use_annotations and has_multi_task:
   - `mat_tensor = torch.tensor(mat_label, dtype=torch.long)`
   - `qual_tensor = torch.tensor(qual_label, dtype=torch.long)`
   - Return (image, {"maturity": mat_tensor, "quality": qual_tensor})
7. Else: Return (image, 0) - single class mode

### 9.3.4 Sequence of Operations (Checkpoint Save)
1. `_save_checkpoint(model, optimizer, epoch, val_acc, config, is_best)`
2. Construct checkpoint dict: {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_acc": val_acc, "config": {...}}
3. If is_best: fname = "best_model.pth"; else: fname = "checkpoint_epoch_031.pth"
4. `torch.save(checkpoint, path)` → write to checkpoint_dir
5. Print confirmation: "Best Checkpoint saved: path" or "Checkpoint saved: path"

### 9.3.5 Decision Points
- Is this the best validation accuracy? → determines filename (best_model.pth vs checkpoint_epoch_NNN.pth)
- Annotation file available? → determines if use_annotations=True or single-class mode

### 9.3.6 Data Exchanged
- Image file → tensor (3, 224, 224) after transform
- CSV rows → (maturity_idx, quality_idx) lookup
- Checkpoint dict → serialized .pth file

### 9.3.7 Success Path
Image successfully loaded and transformed for model input; or checkpoint saved to disk with all necessary state for resumption.

### 9.3.8 Failure Path
Image file not found → FileNotFoundError; corrupt image → PIL raises exception; caught by try/except in pipeline or predict; annotation CSV malformed → graceful fallback to single-class mode.

### 9.3.9 Final Outcome
Image successfully loaded and transformed for model input; or checkpoint saved to disk with all necessary state for resumption.

---

## 9.4 External API Integration Workflow

The project does not integrate with external APIs. However, the **annotated data workflow** can be documented:

```
┌─────────────────────┐    ┌─────────────────────┐
│  Label Studio       │    │  Label Propagation  │
│  Annotation Tool    │    │  (programmatic)     │
└──────▲──────────────┘    └──────▲──────────────┘
       │                      │
       │  Manual labeling     │  Programmatic propagation
       │  of 270 original     │  of labels to augmented
       │  images              │  variants (540 total)
       │                      │
       │  Output: annotations.csv│
       │                      │
       ▼                      ▼
┌─────────────────────┐    ┌─────────────────────┐
│  Dataset Engineering │    │  Model Training     │
│  Pipeline           │    │  (breadfruit_classifier)│
│  (run_pipeline.py)  │    │  (main.py train)    │
└─────────────────────┘    └─────────────────────┘
```

### 9.4.1 Starting Event
Domain expert labels 270 original breadfruit images in Label Studio

### 9.4.2 Actors and Components
- **Domain expert**: Assigns maturity and quality labels per image
- **Label Studio**: Web-based annotation tool (external to this codebase)
- **Programmatic propagation**: Labels assigned to augmented image variants

### 9.4.3 Sequence of Operations
1. Domain expert opens Label Studio; views each breadfruit image
2. Assigns maturity class: immature / partially_mature / mature
3. Assigns quality grade: poor / good / excellent
4. Export annotations → CSV file (270 rows: filename, maturity, quality)
5. Programmatic pipeline propagates these labels to all augmented variants
6. 540 total annotation entries (270 original × 2 augmentation variants, approximately)
7. CSV used as annotation_file in model training/dataloader

### 9.4.4 Decision Points
- Number of augmentation variants per original image (determines 540 vs other count)
- Which augmented images receive which propagated labels

### 9.4.5 Data Exchanged
- 270 manual annotation entries (filename, maturity, quality)
- 540 propagated annotation entries
- Label mappings: class names → indices (immature→0, partially_mature→1, mature→2; poor→0, good→1, excellent→2)

### 9.4.6 Success Path
270 original images labeled → labels propagated to augmented dataset → annotation CSV ready for model training.

### 9.4.7 Failure Path
Label Studio not available; manual labels missing for some images; propagation algorithm errors → some samples lack labels → falls back to single-class mode.

### 9.4.8 Final Outcome
Annotation CSV with 540 entries (or fewer if propagation incomplete) → used as annotation_file config for classifier training.
```

---

## 9.5 Workflow Diagram Summary

| Workflow Diagram | Starting Event | Key Decision Points | Success Path | Failure Path |
|---|---|---|---|---|
| Overall Application | Data collection / user action | Pipeline phase flags; early stopping; backbone freeze/unfreeze; model selection | All 9 pipeline phases → trained model → evaluation → UI classification | Pipeline error; missing checkpoint; OOM |
| Backend Processing | Beginning of epoch | Gradient clipping threshold; scheduler patience; early stopping patience; best model selection | Epoch completes with metrics logged | NaN loss; OOM; NaN metrics |
| Database Operations | __getitem__() call / checkpoint save | Best accuracy determination; annotation file availability | Image loaded/checkpoint saved | File not found; corrupt image; malformed CSV |
| External API Integration | Domain expert labeling | Number of augmentation variants; label propagation which images | 270 labeled → 540 propagated → CSV ready | Label Studio unavailable; missing labels; propagation errors |
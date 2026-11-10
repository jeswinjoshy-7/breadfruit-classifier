# DELIVERABLE 8: Methodology and Implementation Details

## 8.1 Development Methodology

Based on code inspection and project structure, the project appears to follow a **custom incremental development workflow** rather than a strict off-the-shelf methodology. Evidence from code analysis:

### 8.1.1 Observed Implementation Workflow

#### 1. Requirement Identification
- Problem defined: breadfruit maturity/quality assessment
- Objectives set: automated classification replacing manual inspection
- Classes determined: 3 maturity stages (immature/partially_mature/mature), 3 quality grades (poor/good/excellent)

#### 2. System Analysis
- Dataset collection from two sources (personal smartphone + Kaggle)
- Image preprocessing pipeline designed (9 phases)
- Model architecture selected (EfficientNet-B0 with multi-task heads)
- Two-phase transfer learning strategy adopted
- Evaluation metrics identified (accuracy, precision, recall, F1)

#### 3. Technology Selection
- Python 3.13 + PyTorch 2.6.0 + TorchVision 0.21.0
- EfficientNet-B0 backbone with ImageNet pretrained weights
- Streamlit for web interface
- OpenCV + NumPy for pipeline augmentation
- CSV for annotation storage

#### 4. Architecture Design
- Multi-task architecture shared backbone with dual heads
- Config dataclasses for centralized hyperparameter management
- Two-phase transfer learning (freeze 5 epochs, fine-tune 16 epochs)
- Grad-CAM for interpretability

#### 5. Module Development
- `breadfruit_classifier/` - Model, datasets, training, evaluation, prediction, Grad-CAM
- `breadfruit_pipeline/` - 9-phase dataset engineering
- `streamlit_app/` - Web interface
- Each module developed independently with config-driven behavior

#### 6. Database Implementation
- File-based: CSV annotations, directory-structured images, PyTorch checkpoints
- No SQL or document database

#### 7. Frontend Development
- Streamlit app with custom CSS styling
- Two-tab layout (Upload/Sample)
- Prediction cards, confidence bars, Grad-CAM visualizations

#### 8. Backend Development
- PyTorch training loop with early stopping
- Mixed precision training (AMP)
- Evaluation with comprehensive metrics

#### 9. API Integration
- No external APIs; internal module imports only
- Label Studio mentioned in technical summary but not integrated in current code

#### 10. Error Handling
- try/except in pipeline phases
- Graceful error messages in Streamlit
- Checkpoint existence validation

#### 11. Testing
- Evaluation module produces comprehensive metrics
- Confusion matrices and classification reports generated
- No explicit unit test framework (pytest) observed

#### 12. Deployment
- Local streamlit run (port 8501)
- CLI commands via main.py subcommands
- No Docker, no CI/CD, no production deployment configuration

### 8.1.2 Evidence For/Against Specific Methodologies

- **Agile**: Not evidenced by sprints, stand-ups, or iterative development cycles. However, incremental module development and config-driven approach enables iterative experimentation.
- **Scrum**: No evidence of Scrum ceremonies (daily stand-ups, sprint planning, reviews).
- **Waterfall**: Not pure waterfall; there is some iteration (e.g., data pipeline → model training → evaluation → Grad-CAM).
- **Custom/Ad-hoc**: Most consistent with the evidence. The project appears to be a research prototype developed with pragmatic, experiment-driven workflow.

### 8.1.3 Proposed Documentation of Methodology for Final Report

Given the evidence, the final report should document this as a **research prototype development workflow** with the following phases:

1. **Problem definition and dataset collection**
2. **Data engineering pipeline implementation** (9 phases)
3. **Model architecture design and transfer learning strategy**
4. **Training pipeline with two-phase transfer learning**
5. **Evaluation and metric computation**
6. **Explainability (Grad-CAM) implementation**
7. **Streamlit web interface development**
8. **Integration and documentation**

Each phase builds on the previous, with the config dataclass system enabling experimentation across phases.

---

## 8.2 Implementation of Important Features (Execution Paths)

### 8.2.1 Path 1: Single Image Classification via Web UI

**User Action → Frontend → Backend → Response → UI Update**:

1. **User Action**: Operator uploads breadfruit image via Streamlit file uploader
2. **Frontend**: `st.file_uploader()` captures file; `Image.open(uploaded).convert("RGB")` creates PIL Image; stores in `st.session_state["image_upload"]`
3. **Backend**: User clicks "Classify" button → `predict_single(model, st.session_state["image_upload"], config, return_probs=True)`
   - `predict_single()`: Sets `model.eval()`; determines device from model parameters
   - Opens image if Path/String: `Image.open(image).convert("RGB")`
   - Applies `get_inference_transform(config)`: Resize(224) → CenterCrop(224) → ToTensor → Normalize(mean, std)
   - Adds batch dimension: `input_tensor = transform(pil_image).unsqueeze(0).to(device)`
   - Forward pass: `predictions = model(input_tensor)` → BreadfruitClassifier.forward()
   - Backbone: EfficientNet-B0 with classifier→Identity → GAP → 1280-D features
   - Maturity head: Dropout(p=0.2) → Linear(1280, 3) → maturity_logits
   - Quality head: Dropout(p=0.2) → Linear(1280, 3) → quality_logits
   - Softmax: `probs = torch.softmax(logits, dim=1)` → class probabilities
   - Argmax: `pred_idx = logits.argmax(dim=1).item()` → predicted class index
   - Confidence: `confidence = probs[0, pred_idx].item()`
   - Class names from config: `config.model.maturity_classes` / `config.model.quality_classes`
   - Returns dict: `{"maturity": {"prediction": cls, "confidence": val, "probabilities": {...}}, "quality": {...}}`
4. **Response Processing**: Streamlit receives result dict; calls `render_prediction_card()` for each task
   - Generates HTML card with prediction label and confidence bar
   - Sorts probabilities descending; renders table with class names and progress bars
   - Updates stat boxes: Confidence %, Maturity class title, Quality class title, Model name
5. **UI Update**: Streamlit re-renders page with prediction results displayed; "Grad-CAM Visualization" expander rendered; if expanded, `render_gradcam_tab()` generates heatmaps

### 8.2.2 Path 2: Training Pipeline

1. **User Action**: Developer runs `python main.py train`
2. **CLI**: `argparse` parses command; `cmd_train(config, args)` called
3. **Setup**: `create_output_dirs()`, `set_seed(config.training.seed)`
4. **Phase 1 (epochs 1-5)**: Freeze backbone → `model.freeze_backbone(True)` → AdamW on head parameters only
   - `train_epoch()`: Forward pass → `model(images)` → loss → backward → `optimizer.step()` → gradient clipping `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)`
   - `validate()`: Forward pass on val loader → compute accuracy, loss
   - `EarlyStopping(val_acc, epoch)` → monitor patience=10
   - `scheduler.step(val_acc)` → ReduceLROnPlateau
   - Save best checkpoint: `_save_checkpoint(model, optimizer, epoch, val_acc, config, is_best=True)` → `torch.save()` → `best_model.pth`
5. **Phase 2 (epochs 6-21)**: Unfreeze backbone → `model.freeze_backbone(False)` → AdamW on all parameters with LR=1e-4
   - Continue training with ReduceLROnPlateau patience=5
   - Early stopping may trigger at epoch 21 (as observed: best val avg accuracy 0.9222 at epoch 21)
6. **Evaluation**: After training, load best checkpoint → `evaluate_model(model, test_loader, config)` → `print_evaluation_report(results)` → confusion matrices → plots
7. **Results**: Training history JSON saved; results JSON saved; plots (loss curves, accuracy, LR schedule) saved to plots/ directory

### 8.2.3 Path 3: Dataset Engineering Pipeline

1. **User Action**: Developer runs `python run_pipeline.py`
2. **Pipeline**: `run_pipeline(config_path)` loaded from `breadfruit_pipeline/config/config.yaml`
3. **Phase 1 (Inspection)**: `DatasetInspector.inspect_dataset()` → file format validation, SHA-256 checksums, EXIF orientation, metadata extraction; writes reports
4. **Phase 2 (Duplicates)**: `DuplicateDetector.detect_duplicates()` → perceptual hash (pHash), difference hash (dHash), average hash (aHash), color histogram comparison, SSIM; consensus threshold 3 of 5 methods; flagged duplicates moved to Rejected/Duplicates/
5. **Phase 3 (Quality)**: `QualityAnalyzer.analyze_dataset()` → 9 quality metrics evaluated per image (blur, sharpness, brightness, contrast, entropy, noise, compression quality, resolution, aspect ratio); flagged images moved to Rejected/ subdirectories
6. **Phase 4 (Object Validation)**: `ObjectValidator.validate_dataset()` → HSV-based color segmentation for breadfruit presence; two HSV ranges for green/yellow detection; minimum fruit area ratio 0.05; maximum occlusion 0.5; maximum border touch 0.1; maximum background 0.8; flagged images moved to appropriate Rejected/ subdirectories
7. **Phase 5 (Standardization)**: `ImageStandardizer.standardize_dataset()` → resize to 224×224, RGB conversion, JPEG quality 95, alpha channel removal, aspect ratio preservation with zero-padding
8. **Phase 6 (Organization)**: `DatasetOrganizer.organize_datasets()` → create Personal_Clean/ and Public_Clean/ directories; move valid images; organize into train/val/test splits
9. **Phase 7 (Analysis)**: `DatasetAnalyzer.analyze_dataset()` → statistical analysis: class distributions, image property distributions, quality metric histograms; generate visualizations (seaborn plots)
10. **Phase 8 (Reports)**: `ReportGenerator.generate_all_reports()` → CSV and TXT reports documenting all pipeline decisions, thresholds, and statistics
11. **Phase 9 (Validation)**: `FinalValidator.validate_all_datasets()` → integrity checks: verify all expected files present and correctly formatted; overall pass/fail report

---

## 8.3 Configuration-Driven Workflow

All modules receive Config dataclass injection at initialization. Key configuration flow:

1. **Config creation**: `Config()` instantiates all nested dataclasses with defaults (config.py:95-96)
   - `DataConfig`: data_root, annotation_file, train/val/test dirs, image_size, mean, std
   - `ModelConfig`: backbone, pretrained, dropout, maturity/quality classes, freeze_backbone_epochs
   - `TrainingConfig`: batch_size, epochs, learning_rate, weight_decay, loss_weights, early_stopping/patience/scheduler params
   - `AugmentationConfig`: train_augmentation flag, all augmentation probabilities and parameters
   - `LoggingConfig`: experiment_name, checkpoint_dir, log_dir, plot_dir, tensorboard, log_interval
   - `GradCAMConfig`: target_layer, use_cuda

2. **Module initialization**: Each module imports Config and reads relevant sections
   - `model.py` reads ModelConfig (backbone type, dropout, class lists, freeze epochs)
   - `datasets.py` reads DataConfig (data_root, annotation_file, image_size, mean, std)
   - `train.py` reads TrainingConfig (batch_size, epochs, learning_rate, weight_decay, scheduler params)
   - `gradcam.py` reads GradCAMConfig (target_layer name)
   - `streamlit_app/app.py` reads config values for model loading and UI setup

3. **Data flow**: 
   - Config.data → BreadfruitDataset: data_root, annotation_file, train_dir, val_dir, test_dir
   - Config.model → BreadfruitClassifier: backbone type, dropout, class lists, freeze epochs
   - Config.training → train.py: batch_size, epochs, learning_rate, weight_decay, scheduler params
   - Config.gradcam → gradcam.py: target_layer name for heatmap generation

4. **Pipeline flow**: Config.yaml enables/disables phases; thresholds control quality checks (config.yaml:phase1-9 settings)

5. **Experiment variation**: Change config values or pass CLI arguments (main.py:update_config_from_args) to override defaults for different experimental runs
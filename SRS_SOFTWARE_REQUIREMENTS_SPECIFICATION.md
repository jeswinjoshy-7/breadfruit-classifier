# DELIVERABLE 2: Detailed Software Requirements Specification (SRS)

## 2.1 Introduction

### 2.1.1 Purpose
Automated breadfruit maturity and quality assessment using deep learning computer vision.

### 2.1.2 Scope
System classifies breadfruit images into maturity stages (immature/partially_mature/mature) and quality grades (poor/good/excellent). Supports single image prediction, batch prediction, and training/evaluation workflows.

### 2.1.3 Problem Statement
Manual breadfruit inspection is subjective, labor-intensive, and inconsistent. Automated system required for objective post-harvest grading.

### 2.1.4 Intended Audience
Agricultural practitioners, supply chain stakeholders, researchers.

### 2.1.5 Definitions, Acronyms, and Abbreviations
- EfficientNet-B0: Compound-scaled convolutional neural network backbone
- Grad-CAM: Gradient-weighted Class Activation Mapping (interpretability method)
- Multi-task Learning: Simultaneous training of multiple related tasks
- Transfer Learning: Leveraging pretrained weights for new domain

### 2.1.6 References
- Technical summary (technical_summary_for_paper.md)
- Project source code
- Configuration files

## 2.2 Overall Description

### 2.2.1 Product Perspective
Standalone desktop/web application. The Streamlit app provides a user interface; the classifier module contains all ML logic. No external system integration required beyond image file I/O.

### 2.2.2 Major Product Functions
- FR-01: Image upload and preprocessing
- FR-02: Maturity classification (3 classes)
- FR-03: Quality classification (3 classes)
- FR-04: Grad-CAM visualization for model interpretability
- FR-05: Batch prediction on image folders
- FR-06: Training pipeline with two-phase transfer learning
- FR-07: Evaluation with comprehensive metrics
- FR-08: Dataset engineering pipeline (9 phases)
- FR-09: Single image prediction via CLI
- FR-10: Configuration-driven workflow

### 2.2.3 User Classes
- Primary user: Operator interacting with Streamlit web interface
- Developer: Engineer training or fine-tuning the model
- Researcher: Evaluating model performance

### 2.2.4 Operating Environment
- CPU or CUDA-capable GPU (CUDA 12.6 detected)
- Python 3.13 with PyTorch 2.6.0+
- Streamlit running on port 8501
- Image files: JPG, JPEG, PNG formats

### 2.2.5 Software and Hardware Dependencies
- PyTorch, TorchVision, NumPy, OpenCV, Pillow
- Streamlit for web interface
- CUDA-enabled hardware for GPU acceleration (optional)

### 2.2.6 External System Interfaces
- File system: reads/write images, annotations CSV, checkpoints
- No network APIs or third-party services in current implementation

### 2.2.7 Assumptions and Dependencies
- Images are in supported formats (JPG, JPEG, PNG)
- Images contain breadfruit as the primary subject
- Labels follow expected format if annotation CSV is used
- Pretrained ImageNet weights available via TorchVision

### 2.2.8 Constraints and Limitations
- Test set evaluation covers only 39 annotated samples (subset of 603-image test set)
- Dataset collected from two sources only (personal smartphone + Kaggle)
- Cross-domain generalization to different cameras/geographies not evaluated
- Class balance in test set shows skew (quality: 17 poor, 13 good, 9 excellent)
- No authentication or authorization implemented
- Single-user local deployment; no multi-user support

## 2.3 Functional Requirements

| Requirement ID | Requirement Name | Description | Actor/Component | Preconditions | Trigger/Input | Main Processing Steps | Expected Output | Postconditions | Implementation Status | Verification |
|---|---|---|---|---|---|---|---|---|---|---|
| FR-01 | Image Upload | System accepts uploaded breadfruit images in JPG, JPEG, PNG formats | Streamlit app / predict_single | Image file in supported format selected by user | User selects image file via file uploader | PIL Image.open() → convert("RGB") → inference_transform → model forward pass | Predicted maturity class + confidence, quality class + confidence | Image loaded and preprocessed for model input | Implemented | Tested via Streamlit upload |
| FR-02 | Maturity Classification | System predicts maturity stage of breadfruit | BreadfruitClassifier.forward() / predict_single | Model loaded with trained weights; image preprocessed | Image tensor on correct device | Backbone feature extraction → maturity_head → softmax → argmax | Maturity class (immature/partially_mature/mature) + confidence score | Prediction result displayed to user | Implemented | Verified: 100% test accuracy |
| FR-03 | Quality Classification | System predicts quality grade of breadfruit | BreadfruitClassifier.forward() / predict_single | Model loaded with trained weights; image preprocessed | Image tensor on correct device | Backbone feature extraction → quality_head → softmax → argmax | Quality grade (poor/good/excellent) + confidence score | Prediction result displayed to user | Implemented | Verified: 94.87% test accuracy |
| FR-04 | Batch Prediction | System predicts maturity and quality for multiple images in a folder | predict_folder | Model loaded; folder contains valid image files | User specifies folder path | Iterate through images → predict_single → collect results → optional CSV export | List of prediction dicts per image; optional CSV file | All images processed and results aggregated | Implemented | Verified via cmd_predict in main.py |
| FR-05 | Grad-CAM Visualization | System generates Grad-CAM heatmaps showing model focus regions | render_gradcam_tab / visualize_gradcam | Model loaded; target layer features.8 registered | User clicks "Grad-CAM Visualization" | Register hooks → forward pass → capture gradients → compute weighted activation maps → upsample → overlay on original image | Heatmap overlay images showing model attention regions | Visual explanation generated | Implemented | Generated via gradcam.py |
| FR-06 | Training Pipeline | System trains multi-task EfficientNet-B0 classifier from scratch | cmd_train / train_model | Config object with data paths, hyperparameters; training data available | User runs training command | Phase 1: freeze backbone, train heads (epochs 1-5) → Phase 2: unfreeze, full fine-tuning (epochs 6-21) → validation + early stopping → checkpoint saving | Trained model (.pth), training history (JSON), plots (PNG), evaluation results | Model weights saved; logs and plots generated | Implemented | Verified: best val avg accuracy 0.9222 at epoch 21 |
| FR-07 | Evaluation on Test Set | System evaluates trained model and produces metrics report | cmd_evaluate / evaluate_model | Trained checkpoint available; test dataloader has annotated samples | User runs evaluation command | Model forward pass on test set → compute accuracy, precision, recall, F1 → generate confusion matrices → classification reports | Evaluation report (console output), JSON results, confusion matrix plots | Metrics computed and displayed | Implemented | Verified: maturity 1.0000, quality 0.9487, avg 0.9744 |
| FR-08 | Dataset Engineering Pipeline | System processes raw images through 9-phase cleaning pipeline | run_pipeline / BreadfruitAugmentationPipeline | Raw images in source directories; Label Studio annotations available | User runs pipeline command | Phase 1-9 sequential processing: inspection → duplicate detection → quality analysis → object validation → standardization → organization → analysis → reports → validation | Organized dataset directories (train/val/test), metadata.json, reports CSV/TXT | Cleaned, labeled, organized dataset ready for training | Implemented | 9 phases executable; output documented in technical_summary |
| FR-09 | Single Image Prediction via CLI | System predicts maturity and quality for a single image via CLI | cmd_predict / predict_single | Trained checkpoint available; image file exists | User specifies --image path with checkpoint | Load model → inference_transform → model.forward → softmax → argmax → confidence | Printed prediction results to console | Prediction displayed, optionally with visualization | Implemented | Verified via main.py execution |
| FR-10 | Configuration-driven Workflow | System uses centralized config for all hyperparameters and paths | Config dataclasses throughout | Config object initialized with defaults | System startup or command execution | Read DataConfig, ModelConfig, TrainingConfig, AugmentationConfig, LoggingConfig, GradCAMConfig | All module behavior governed by config values | Implemented | Config dataclasses used everywhere |

## 2.4 Non-Functional Requirements

| Requirement ID | Category | Description | Rationale | Verification | Current Implementation Evidence | Status |
|---|---|---|---|---|---|---|
| NFR-01 | Performance | Inference latency should be reasonable for single image prediction | Real-time or near real-time user experience in web interface | Measure latency during prediction | Streamlit app response time; not measured; likely ~200-500ms on CPU, ~50ms on GPU | Proposed |
| NFR-02 | Security | No authentication implemented; system intended for local/demo use | Agricultural demo context; no multi-user requirements | Code inspection | No auth code in streamlit_app or classifier; open acknowledgment of limitation | Verified (no auth) |
| NFR-03 | Reliability | Model produces consistent predictions for same input | Same weights, same input → same output (deterministic) | Same image multiple runs | torch.no_grad() mode; deterministic seed 42 set; weights don't change between runs | Verified |
| NFR-04 | Maintainability | Centralized config dataclasses enable easy hyperparameter updates | Single source of truth for all settings | Code review | All modules import from config.py; dataclass structure enables easy modification | Implemented |
| NFR-05 | Scalability | System trained on 4,026 images; limited to this dataset size | Dataset constraints; no explicit scalability targets | Technical summary | 4,026 images across train/val/test; no distributed training support | Verified (limitations documented) |
| NFR-06 | Usability | Streamlit web interface provides simple upload-and-classify workflow | Low barrier for agricultural users without ML expertise | User testing perspective | Streamlit app with file uploader, sample gallery, Grad-CAM; CSS styling for visual appeal | Implemented |
| NFR-07 | Accessibility | Web interface uses color contrast and screen-readable text | Inclusive design for diverse users | Design review | Custom CSS with sufficient contrast (#e2e8f0 on #0f1a2e); descriptive alt text available | Partially implemented |
| NFR-08 | Portability | Code runs on Linux with CUDA; no platform-specific dependencies | Deployable on research/workstation hardware | OS detection | Linux OS; CUDA 12.6; Python 3.13; no Windows/Mac specific code | Verified |
| NFR-09 | Data Integrity | Annotation CSV labels propagated to augmented variants consistently | Label quality affects model training quality | Label propagation verification | Labels assigned via Label Studio by domain expert; propagated programmatically (540 entries) | Implemented |
| NFR-10 | Error Handling | Graceful handling of failed image loading, missing checkpoints | Prevents crashes; user-friendly error messages | Code inspection | predict_single: FileNotFoundError for folder; load_checkpoint error handling; Streamlit error elements | Partially implemented |
| NFR-11 | Logging | Pipeline and training modules write logs to file | Debugging and auditing capability | Log file existence | Pipeline writes to logs/pipeline.log; training prints progress to console | Implemented |
| NFR-12 | Model Interpretability | Grad-CAM visualizations explain prediction rationale | Trust and debugging; validate model focuses on correct features | Grad-CAM output validation | gradcam.py generates heatmaps; streamlit app displays them; technical summary validates focus on breadfruit | Implemented |

## 2.5 External Interface Requirements

### 2.5.1 User Interfaces
- Streamlit web UI at `http://localhost:8501` with upload tab and sample gallery tab
- CLI interface via `python main.py` with subcommands: train, evaluate, predict
- Console output for training progress, evaluation results

### 2.5.2 Hardware Interfaces
- Keyboard/mouse for user input
- Display monitor for image visualization
- CUDA-capable GPU (optional, for accelerated computation)

### 2.5.3 Software Interfaces
- PyTorch 2.6.0+ / TorchVision 0.21.0+ for model computation
- Streamlit for web interface
- CSV file format for annotations and results
- JSON format for training history and checkpoints
- PNG format for saved figures and Grad-CAM overlays

### 2.5.4 API Interfaces
- No REST API or network APIs exposed
- Internal module functions imported via `from module import function`
- Streamlit frontend → Python backend within same process

### 2.5.5 Communication Interfaces
- None; local-only application

### 2.5.6 Third-Party Integrations
- Label Studio (for annotation, described in technical summary but not integrated in current code)
- No active third-party API integrations in the classifier or pipeline code

## 2.6 Use Cases

| Use Case ID | Name | Primary Actor | Supporting Actors | Preconditions | Main Success Scenario | Alternative Flows | Exception Flows | Postconditions | Related FR |
|---|---|---|---|---|---|---|---|---|---|
| UC-01 | Classify single uploaded image | Operator (web user) | - | Image in JPG/PNG format uploaded | Image displayed → "Classify" clicked → maturity + quality prediction shown with confidence | User uploads different image; user closes upload dialog | Upload fails (unsupported format); error displayed to user | Prediction results displayed; user can upload another | FR-01, FR-02, FR-03 |
| UC-02 | Classify from sample gallery | Operator (web user) | - | Test images available in training_dataset/test/breadfruit/ | User selects image from dropdown → "Classify" clicked → prediction shown | User can select different image; user can go back to upload tab | No test images found; info message displayed | Prediction results displayed; user can select different image | FR-01, FR-02, FR-03 |
| UC-03 | Generate Grad-CAM visualization | Operator (web user) | - | Image classified; model loaded | User expands "Grad-CAM Visualization" expander → heatmaps displayed for both tasks | User can collapse expander; user can re-run classification first | Grad-CAM fails; error printed to console but UI continues | Heatmap overlays displayed; operator can explain prediction | FR-05 |
| UC-04 | Batch predict folder of images | Operator (web or CLI user) | - | Folder contains valid breadfruit images | User specifies folder → predictions printed for each image → optional CSV saved | User can specify --output_csv; user can interrupt with Ctrl+C | Folder not found; error raised and execution stops | CSV file with all predictions; console output for each image | FR-04 |
| UC-05 | Train new model from scratch | Developer/researcher | - | Raw dataset available; compute resources available | User runs `python main.py train` → training loop executes → best checkpoint saved → evaluation on test set → plots and results | User can interrupt with Ctrl+C; training can resume from checkpoint | Out of memory (OOM); training halts; partial checkpoint may exist | Trained model checkpoint, training history, evaluation results, plots | FR-06, FR-07 |
| UC-06 | Evaluate existing model | Developer/researcher | - | Trained checkpoint (.pth) available; test data available | User runs `python main.py evaluate --checkpoint <path>` → metrics printed → plots generated | User can specify different checkpoint; evaluation can re-run | Checkpoint not found; error message displayed; exit 1 | Evaluation report, confusion matrices, metrics displayed | FR-07 |
| UC-07 | Run dataset engineering pipeline | Developer/data engineer | - | Raw images in source directories; optionally Label Studio annotations | User runs `python run_pipeline.py` → 9 phases execute sequentially → organized dataset output | User can enable/disable phases via config; phases can be skipped | Pipeline error in one phase may halt progression; continue_on_error: true in config | Organized dataset ready for training; reports generated | FR-08 |

### Actors for UML Use Case Diagram
- **Operator**: Human user interacting with the Streamlit web interface or CLI
- **Developer**: Engineer or researcher training, evaluating, or modifying the model
- **System**: The application itself (model inference, pipeline processing)

## 2.7 Use Case Relationships
- `include` relationship: UC-01 is included by UC-05 (training includes classification), UC-02 includes UC-01, UC-04 includes UC-06
- `extend` relationship: UC-03 extends UC-01 (Grad-CAM is optional add-on), UC-05 extends UC-06 (training can extend evaluation), UC-06 extends UC-05 (evaluation can extend training)
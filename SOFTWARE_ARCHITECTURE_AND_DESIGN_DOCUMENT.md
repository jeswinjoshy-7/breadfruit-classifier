# DELIVERABLE 3: Software Architecture and Design Document

## 3.1 Architectural Design

### 3.1.1 Selected Architecture: Layered Architecture
The project employs a **Layered Architecture** with clear separation between:

1. **Presentation Layer**: Streamlit web application (app.py) with CSS styling, file upload components, result display cards, and Grad-CAM visualizations
2. **Application Logic Layer**: Python CLI entry point (main.py) orchestrating training, evaluation, and prediction commands
3. **ML/Model Layer**: PyTorch-based BreadfruitClassifier with EfficientNet-B0 backbone and dual classification heads
4. **Data Layer**: Dataset module (datasets.py) handling image loading, transforms, and DataLoader creation; CSV annotation file I/O
5. **Pipeline Layer**: 9-phase dataset engineering pipeline (breadfruit_pipeline) for dataset cleaning and organization

### 3.1.2 Key Design Decisions

#### EfficientNet-B0 as Backbone
- Chosen for computational efficiency (5.3M parameters) with ImageNet pretrained weights
- Suitable for resource-constrained deployment in agricultural settings
- Compound scaling uniformly balances depth, width, and resolution

#### Multi-Task Architecture
- Shared backbone with separate maturity and quality heads
- Enables joint learning with task-weighted loss (0.5 each)
- Leverages correlation between maturity and quality assessment

#### Two-Phase Transfer Learning
- **Phase 1 (epochs 1-5)**: Freeze backbone, train only heads with AdamW on head parameters (LR=1e-3)
- **Phase 2 (epochs 6-21)**: Unfreeze backbone, full fine-tuning with reduced LR (1e-4)
- Standard approach for small datasets (4,026 images)

#### AMP Mixed Precision
- Enabled via PyTorch GradScaler for faster training and reduced memory usage
- Training completed in ~2.33 minutes across 21 epochs before early stopping

#### Config Dataclasses
- Centralized configuration via dataclasses for all hyperparameters and paths
- Enables easy experimentation and reproducibility
- Nested structure: DataConfig, ModelConfig, TrainingConfig, AugmentationConfig, LoggingConfig, GradCAMConfig

### 3.1.3 Communication Mechanisms

#### Forward Pass
Input tensor → backbone (EfficientNet-B0, classifier→Identity → GAP → 1280-D features) → dual heads → logits → softmax → probabilities → argmax

#### Backward Pass
Gradients flow from loss through heads to backbone for parameter updates

#### Pipeline Phases
Sequential data processing with configurable enable/disable per phase (config.yaml)

#### Streamlit ↔ Classifier
Image upload → preprocessing → model inference → results display

### 3.1.4 Design Rationale
- Layered architecture separates concerns: UI (Streamlit), logic (CLI/main.py), ML (model/datasets), data processing (pipeline)
- Multi-task learning leverages correlation between maturity and quality for breadfruit
- Two-phase transfer learning is standard for small agricultural datasets
- Config dataclasses follow Python best practices for experiment reproducibility
- Grad-CAM provides model interpretability crucial for agricultural AI trustworthiness

### 3.1.5 Actual Architecture Diagram (text representation)

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT APP (app.py)                   │
│  - CSS styling, page config                                    │
│  - File upload, sample gallery                                 │
│  - Prediction display, Grad-CAM                                │
└──────────────────────▲─────────────────────────────────────────┘
                       │ (imports breadfruit_classifier module)
                       │
                ┌──────▼──────┐
                │ breadfruit   │
                │ classifier    │
                │ (PyTorch)     │
                ├──────────────│
                │ • EfficientNet-B0 │
                │ • Multi-task heads │
                │ • Forward/backward pass │
                └──────────────▲──────────────────────┐
                               │                        │
                               │                        │
                   ┌─────────────┌──────────┴─────────────┐
                   │             │                        │
                   ▼             ▼                        ▼
         ┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
         │ datasets.py     │  │ train.py / eval.py  │  │ predict.py / gradcam │
         │ Dataset,        │  │ Training loop,      │  │ Single/image batch  │
         │ transforms      │  │ validation,         │  │ prediction, Grad-CAM│
         │ get_transforms  │  │ early stopping      │  │                     │
         └─────────────────┘  └─────────────────────┘  └─────────────────────┘
                   ▲             │                        │
                   │             ▼                        ▼
                   └────── Config ──┘  ┌───────────────────────┘
                                        │ Config dataclasses  │
                                        │ (config.py)         │
                                        └─────────────────────┘
```

## 3.2 Module Design

### 3.2.1 Module: breadfruit_classifier/model.py

**Purpose**: Defines the EfficientNet-B0 multi-task neural network architecture with dual classification heads for maturity and quality prediction.

**Responsibilities**:
- Backbone initialization with pretrained ImageNet weights
- Replacing backbone classifier with nn.Identity for feature extraction
- Creating maturity_head (Linear → Dropout → Linear) with 3 classes
- Creating quality_head (same structure) with 3 classes
- Head weight initialization (Xavier uniform, gain=0.1)
- Backbone freeze/unfreeze functionality
- Forward pass producing maturity_logits and quality_logits
- MultiTaskLoss module for weighted cross-entropy combination
- Parameter counting and feature extractor extraction

**Inputs**: Preprocessed image tensor (3 × 224 × 224)

**Outputs**: Dictionary {"maturity_logits": Tensor(3), "quality_logits": Tensor(3)}

**Dependencies**: torch, torch.nn, torchvision.models, config.Config

**Important Functions**:
- `BreadfruitClassifier.__init__(config)` - Model construction
- `BreadfruitClassifier.forward(x)` - Inference forward pass
- `MultiTaskLoss(predictions, targets)` - Combined loss computation
- `create_model(config)` - Factory function with device placement
- `count_parameters(model)` - Trainable parameter count
- `get_feature_extractor(model)` - Backbone without heads

### 3.2.2 Module: breadfruit_classifier/datasets.py

**Purpose**: Loads breadfruit images, applies transforms, and creates DataLoaders for multi-task classification.

**Responsibilities**:
- BreadfruitDataset class supporting directory structure: data_root/split/breadfruit/ + annotations.csv
- _load_samples: Reads annotation CSV; matches filenames to maturity/quality labels
- Falls back to single-class mode if no annotation file
- __getitem__: Loads image, applies transform, returns (image, {"maturity": tensor, "quality": tensor}) or (image, 0)
- get_transforms: Builds train/val/test transform pipelines with augmentation
- get_dataloaders: Creates train, val, and test DataLoaders
- get_inference_transform: Minimal transform for single-image inference

**Inputs**: data_root path, split name (train/val/test), annotation_file path, transform compose

**Outputs**: Dataset objects yielding (image, targets) pairs; DataLoader objects

**Dependencies**: torch, torchvision.transforms, PIL.Image, csv.DictReader, config.Config

**Important Functions**:
- `BreadfruitDataset.__init__()` - Dataset initialization with annotation lookup
- `BreadfruitDataset._load_samples()` - Load samples from images + annotations
- `BreadfruitDataset.__getitem__()` - Single sample loading with transform
- `BreadfruitDataset.num_maturity_classes` / `num_quality_classes` - Class count properties
- `get_transforms()` - Train/val/test transform pipelines
- `get_dataloaders()` - DataLoader creation
- `get_inference_transform()` - Inference-only transform

### 3.2.3 Module: breadfruit_classifier/train.py

**Purpose**: Implements the full training pipeline with early stopping, learning rate scheduling, checkpointing, and progress logging.

**Responsibilities**:
- EarlyStopping class: Monitor validation accuracy; stop after patience epochs without improvement
- train_epoch(): Run one training epoch; forward pass, loss backward, optimizer step; gradient clipping at max_norm=10.0
- validate(): Run validation; compute loss and accuracy metrics; track per-class correctness
- train_model(): End-to-end training routine with two-phase transfer learning
  - Phase 1: Freeze backbone, train only heads (epochs 1-5) with AdamW on head parameters
  - Phase 2: Unfreeze backbone, full fine-tuning (epochs 6-21) with reduced LR (1e-4)
  - ReduceLROnPlateau scheduler; early stopping patience=10
  - Checkpoint saving (best_model.pth + epoch checkpoints)
  - Training history JSON export
- _save_checkpoint(): Serialize model + optimizer state + config + epoch/val_acc
- load_checkpoint(): Deserialize and load checkpoint into model

**Inputs**: Config object, DataLoaders, model (or None to create new)

**Outputs**: Trained model, history dictionary with per-epoch metrics

**Dependencies**: torch, torch.optim, torch.optim.lr_scheduler, tqdm, config, model, datasets

**Important Functions**:
- `EarlyStopping.__call__()` - Check if early stopping triggered
- `train_epoch()` - One epoch training
- `validate()` - Validation metrics computation
- `train_model()` - Main training routine
- `_save_checkpoint()` - Checkpoint serialization
- `load_checkpoint()` - Checkpoint loading

### 3.2.4 Module: breadfruit_classifier/evaluate.py

**Purpose**: Comprehensive evaluation module producing metrics, confusion matrices, classification reports, and visualizations.

**Responsibilities**:
- evaluate_model(): Run full evaluation on DataLoader; collect predictions and targets
- _compute_metrics(): Compute accuracy, precision, recall, F1 (weighted and per-class); confusion matrices; classification reports
- plot_confusion_matrices(): Plot and save confusion matrix PNGs for both tasks
- plot_training_history(): Plot loss curves, accuracy curves, LR schedule, per-task loss
- print_evaluation_report(): Print structured report to console

**Inputs**: Trained model, DataLoader, Config object

**Outputs**: Dictionary of metrics; saved figure files; console report; JSON results

**Dependencies**: torch, numpy, matplotlib.pyplot, sklearn.metrics, tqdm, config, model, datasets

**Important Functions**:
- `evaluate_model()` - Main evaluation routine
- `_compute_metrics()` - Metric computation helper
- `plot_confusion_matrices()` - Confusion matrix visualization
- `plot_training_history()` - Training history plots
- `print_evaluation_report()` - Console report

### 3.2.5 Module: breadfruit_classifier/predict.py

**Purpose**: Inference module for loading trained models and making predictions on new images.

**Responsibilities**:
- load_model_for_inference(): Load model from checkpoint; set to eval mode
- predict_single(): Predict maturity and quality for single image; return dict with predictions, confidence, optional probabilities
- predict_folder(): Predict for all images in a folder; print results; optional CSV export
- visualize_prediction(): Generate figure with original image + predictions + optional Grad-CAM

**Inputs**: Trained model, image (path/PIL Image), Config object; optional output_csv, return_probs

**Outputs**: Prediction dicts; saved CSV; saved figure files

**Dependencies**: torch, numpy, matplotlib, PIL.Image, config, model, datasets, gradcam

**Important Functions**:
- `load_model_for_inference()` - Checkpoint loading
- `predict_single()` - Single image prediction
- `predict_folder()` - Batch prediction from folder
- `visualize_prediction()` - Prediction visualization with Grad-CAM

### 3.2.6 Module: breadfruit_classifier/gradcam.py

**Purpose**: Grad-CAM visualization for model interpretability, producing class activation heatmaps.

**Responsibilities**:
- GradCAM class: Register forward/backward hooks on target layer; generate heatmap; overlay on original image
- _register_hooks(): Register hooks on features.8 (final conv layer of EfficientNet-B0)
- _get_layer(): Resolve layer name to module (supports dot-separated path + numeric indices)
- _find_last_conv(): Find last Conv2d layer as fallback
- generate(): Produce Grad-CAM heatmap for given input tensor and class index
- overlay_heatmap(): Combine heatmap with original image using alpha blending
- visualize_gradcam(): Full pipeline: load image → generate heatmap → create 3-panel figure (original/heatmap/overlay)
- generate_gradcam_batch(): Process multiple images

**Inputs**: Model, image tensor/path, config, task ("maturity" or "quality"), class_idx optional

**Outputs**: Heatmap numpy array; overlayed image; saved figure files

**Dependencies**: torch, torch.nn.functional, numpy, cv2, matplotlib.pyplot, PIL.Image, config, model, datasets

**Important Functions**:
- `GradCAM.__init__()` - Hook registration
- `GradCAM.generate()` - Heatmap generation
- `GradCAM.overlay_heatmap()` - Heatmap overlay
- `visualize_gradcam()` - Full visualization pipeline

### 3.2.7 Module: streamlit_app/app.py

**Purpose**: Streamlit web application for interactive breadfruit classification with Grad-CAM visualizations.

**Responsibilities**:
- Custom CSS styling (dark theme, Inter font, color palette)
- Page configuration (layout, sidebar, page title)
- Model loading via @st.cache_resource
- Image upload and classification workflow
- Sample gallery from test dataset
- Prediction result display cards
- Grad-CAM visualization in expanders
- Confidence bars, probability tables, stat boxes
- Two-tab layout: "Upload Image" and "Sample Gallery"

**Inputs**: User interactions (file upload, button clicks, sample selection)

**Outputs**: Web page with images, text, interactive components

**Dependencies**: streamlit, torch, numpy, PIL, matplotlib, config from breadfruit_classifier, model, datasets, train, predict, gradcam

**Important Functions**:
- `apply_global_style()` - CSS injection
- `load_model()` - Cached model loading with checkpoint
- `get_confidence_level()` - Confidence threshold classification
- `render_prediction_card()` - Prediction display card
- `generate_gradcam_visualization()` - Grad-CAM heatmap generation
- `render_gradcam_tab()` - Streamlit Grad-CAM display
- `main()` - Application entry point

### 3.2.8 Module: breadfruit_pipeline/run_pipeline.py

**Purpose**: Main entry point for the 9-phase dataset engineering pipeline.

**Responsibilities**:
- Load configuration from config.yaml
- Execute phases 1-9 sequentially based on config enable flags
- Phase 1: Dataset Inspection (format validation, hashing, metadata)
- Phase 2: Duplicate Detection (phash, dhash, ahash, color histogram, SSIM consensus)
- Phase 3: Quality Analysis (blur, sharpness, brightness, contrast, entropy, noise, compression, resolution, aspect ratio)
- Phase 4: Object Validation (HSV color segmentation for breadfruit presence)
- Phase 5: Standardization (resize, RGB conversion, JPEG q95, alpha removal)
- Phase 6: Organization (directory structure: Personal_Clean, Public_Clean, Rejected)
- Phase 7: Analysis (statistical characterization, visualizations)
- Phase 8: Report Generation (CSV and TXT formats)
- Phase 9: Final Validation (integrity checks)
- Logger configuration and progress reporting

**Inputs**: Config YAML path; dataset directory paths from config

**Outputs**: Log file (pipeline.log); reports directory with phase reports; organized dataset; metadata.json

**Dependencies**: breadfruit_pipeline.core.config, core.logger, phases 1-9, utils

**Important Functions**:
- `run_pipeline()` - Main pipeline execution routine

### 3.2.9 Module: breadfruit_pipeline.config.config.yaml

**Purpose**: YAML configuration for the 9-phase dataset engineering pipeline.

**Responsibilities**: Define dataset paths, phase enable/disable flags, thresholds for quality checks, output directory structure, processing parameters

**Structure**: datasets (personal, public), output (base_dir, structure), image_processing, phase1-9 settings, processing, reproducibility, logging

**Key Parameters**:
- Phase thresholds (phash: 8, dhash: 8, ahash: 8, color_histogram: 0.95, ssim: 0.95)
- Quality metric thresholds (blur_score: 100, sharpness_score: 50, etc.)
- Object validation HSV ranges and ratios
- Standardization settings (224x224, RGB, JPEG q95)
- Dataset splits and class labels

## 3.3 UML Diagrams

### DELIVERABLE 3A: Use Case Diagram PlantUML Source

```
@startuml
actor Operator
actor Developer
actor "System Database" as DB

skinparam rectangle {}

package "Breadfruit Maturity & Quality System" {
  usecase UC-01 "Classify single uploaded image" <<include>> UC-05
  usecase UC-02 "Classify from sample gallery" <<include>> UC-01
  usecase UC-03 "Generate Grad-CAM visualization" <<extend>> UC-01
  usecase UC-04 "Batch predict folder of images" <<include>> UC-06
  usecase UC-05 "Train new model from scratch" <<extend>> UC-06
  usecase UC-06 "Evaluate existing model" <<extend>> UC-05
  
  Operator --> UC-01
  Operator --> UC-02
  Operator --> UC-03
  Operator --> UC-04
  Developer --> UC-05
  Developer --> UC-06
  DB --> UC-07 "Run dataset engineering pipeline"
}

@enduml
```

**Title and Caption**: "Use Case Diagram - Breadfruit Classification System" / "Shows system boundary, actors (Operator, Developer), and valid use case relationships including include/extend flows."

### DELIVERABLE 3B: Class Diagram PlantUML Source

```
@startuml
class Config {
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    logging: LoggingConfig
    gradcam: GradCAMConfig
}

class DataConfig {
    data_root: str
    annotation_file: str
    train_dir: str
    val_dir: str
    test_dir: str
    class_name: str
    image_size: Tuple
    input_channels: int
    mean: Tuple
    std: Tuple
}

class ModelConfig {
    backbone: str
    pretrained: bool
    dropout: float
    maturity_classes: List
    quality_classes: List
    freeze_backbone_epochs: int
}

class TrainingConfig {
    batch_size: int
    epochs: float
    learning_rate: float
    weight_decay: float
    loss_weights: Tuple
    early_stopping_patience: int
    lr_scheduler_patience: int
    lr_scheduler_factor: float
    min_lr: float
    num_workers: int
    pin_memory: bool
    seed: int
}

class AugmentationConfig {
    train_augmentation: bool
    : [many augmentation params]
}

class LoggingConfig {
    : [logging params]
}

class GradCAMConfig {
    target_layer: str
    use_cuda: bool
}

class BreadfruitClassifier {
    - backbone: EfficientNet-B0
    - maturity_head: MultiTaskHead
    - quality_head: MultiTaskHead (optional)
    - has_multi_task: bool
    + freeze_backbone(freeze: bool)
    + forward(x: Tensor): dict
}

class MultiTaskHead {
    - classifier: Sequential [Dropout, Linear]
    + forward(x: Tensor): Tensor
}

class MultiTaskLoss {
    - maturity_weight: float
    - quality_weight: float
    - ce_loss: CrossEntropyLoss
    + forward(predictions, targets): Tuple[Tensor, dict]
}

class BreadfruitDataset {
    - data_root: Path
    - split: str
    - image_dir: Path
    - samples: List[(fname, mat, qual)]
    - use_annotations: bool
    - maturity_to_idx: dict
    - quality_to_idx: dict
    + __len__(): int
    + __getitem__(idx): (image, targets)
    + num_maturity_classes: int
    + num_quality_classes: int
}

DataConfig --> BreadfruitClassifier
ModelConfig --> BreadfruitClassifier
TrainingConfig --> BreadfruitClassifier
AugmentationConfig --> BreadfruitClassifier
GradCAMConfig --> GradCAM

BreadfruitDataset --> Config : uses
MultiTaskLoss --> BreadfruitClassifier : uses
MultiTaskHead --> BreadfruitClassifier : realizes
BreadfruitDataset --> MultiTaskHead : labels
BreadfruitDataset --> MultiTaskLoss : targets

@enduml
```

**Title and Caption**: "Class Diagram - Breadfruit Classifier Core Components" / "Shows Config dataclass hierarchy, BreadfruitClassifier with dual heads, MultiTaskHead, MultiTaskLoss, and BreadfruitDataset with annotation lookup."

### DELIVERABLE 3C: Sequence Diagram PlantUML Source (Single Image Prediction)

```
@startuml
participant "User" as U
participant "Streamlit App" as SA
participant "Config" as C
participant "Predict Module" as PM
participant "BreadfruitClassifier" as MC
participant "Datasets Module" as DM
participant "PIL Image" as PI
participant "Model Output" as MO

U -> SA: Upload image file
SA -> C: Config() initialization
SA -> PM: predict_single(model, image, config, return_probs=True)
PM -> DM: get_inference_transform(config)
DM -> C: mean, std, image_size from data config
DM -> PI: Image.open(path).convert("RGB")
PM -> PM: transform(pil_image) → input_tensor unsqueeze(0)
PM -> MC: model(input_tensor) forward pass
MC -> MC: backbone(x) → GAP → features (1280-D)
MC -> MC: maturity_head(features) → maturity_logits
MC -> MC: quality_head(features) → quality_logits
MC -> MC: softmax(maturity_logits) → maturity_probs
MC -> MC: softmax(quality_logits) → quality_probs
MC -> MC: argmax → predicted indices
MC -> MC: confidence = max(prob)
MC -> PM: {"maturity": {prediction, confidence, probabilities}, "quality": {prediction, confidence, probabilities}}
PM -> U: Display prediction cards + confidence bars
@enduml
```

**Title and Caption**: "Sequence Diagram - Single Image Prediction Workflow" / "Shows chronological flow: User → Streamlit → Config → Datasets → PIL Image → Transform → Classifier → Predict Module → User display. Key processing steps: preprocessing, forward pass, softmax, argmax, confidence extraction."

### DELIVERABLE 3D: Additional Diagram Notes

1. **Use Case Diagram**: The `include` relationships reflect that certain use cases fundamentally require others (e.g., batch prediction includes single-image prediction logic). The `extend` relationships reflect optional extensions (Grad-CAM is an add-on to classification).

2. **Class Diagram**: The class diagram accurately reflects the dataclass-based configuration system and the PyTorch module hierarchy. Config is a dataclass, not a regular class, but is modeled as a class for UML clarity.

3. **Sequence Diagram**: This represents the actual execution path from code inspection. The key decision point is the model forward pass which splits into two independent heads. The diagram is consistent with the implemented logic in predict.py:59-99.

## 3.4 Module Interaction Flow

All modules receive Config dataclass injection at initialization. The typical flow:

1. **Config creation**: `Config()` instantiates all nested dataclasses with defaults
2. **Module initialization**: Each module imports Config and reads relevant sections
   - `model.py` reads ModelConfig (backbone, dropout, class lists, freeze epochs)
   - `datasets.py` reads DataConfig (data_root, annotation_file, image_size, mean, std)
   - `train.py` reads TrainingConfig (batch_size, epochs, learning_rate, weight_decay)
   - `gradcam.py` reads GradCAMConfig (target_layer)
   - `streamlit_app/app.py` reads config values for model loading and UI setup
3. **Data flow**: Config.data → BreadfruitDataset; Config.model → BreadfruitClassifier; Config.training → train.py optimizer/scheduler
4. **Pipeline flow**: Config.yaml enables/disables phases; thresholds control quality checks
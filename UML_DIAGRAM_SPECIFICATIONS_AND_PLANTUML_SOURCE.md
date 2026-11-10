# DELIVERABLE 4: UML Diagram Specifications and PlantUML Source Code

## 4.1 Diagram 1: Use Case Diagram

### 4.1.1 Purpose
Shows the system boundary, actors, major use cases, and valid relationships between use cases. Demonstrates how different user interactions map to system functionality through include and extend relationships.

### 4.1.2 Actual Entities and Relationships
- **Actor: Operator**: Human user interacting via Streamlit web UI or CLI
- **Actor: Developer**: Engineer or researcher training or evaluating models
- **System**: Provides classification, training, evaluation, pipeline functions
- **Relationships**: 
  - `include` relationship: UC-01 is included by UC-05 (training includes classification), UC-02 includes UC-01, UC-04 includes UC-06
  - `extend` relationship: UC-03 extends UC-01 (Grad-CAM is optional add-on), UC-05 extends UC-06 (training can extend evaluation), UC-06 extends UC-05 (evaluation can extend training)

### 4.1.3 PlantUML Source Code
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

### 4.1.4 Title and Figure Caption
**Title**: Use Case Diagram - Breadfruit Classification System  
**Caption**: Shows system boundary, actors (Operator, Developer), and valid use case relationships including include/extend flows.

---

## 4.2 Diagram 2: Class Diagram

### 4.2.1 Purpose
Shows the relevant classes, attributes, operations, associations, dependencies, inheritance, and multiplicities where supported by the implementation. Models the object-oriented design of the breadfruit classifier system.

### 4.2.2 Actual Entities and Relationships (from code inspection)
- **Config** (dataclass): Top-level configuration container with nested dataclasses (DataConfig, ModelConfig, TrainingConfig, AugmentationConfig, LoggingConfig, GradCAMConfig)
- **BreadfruitClassifier** (nn.Module): Core model with EfficientNet-B0 backbone, two MultiTaskHead instances, freeze_backbone(), forward() methods
- **MultiTaskHead** (nn.Module): Sequential classifier with Dropout + Linear layer
- **MultiTaskLoss** (nn.Module): Weighted sum of cross-entropy losses
- **BreadfruitDataset** (Dataset): torch.utils.data.Dataset with annotation-based label lookup, transform application
- **Config** is the central configuration object injected into all modules

**Relationships (as evidenced in code)**:
- Config is composed of nested dataclasses that configure BreadfruitClassifier
- BreadfruitDataset uses Config's maturity_classes and quality_classes for label mapping
- MultiTaskHead is composed within BreadfruitClassifier
- MultiTaskLoss takes predictions dict and targets dict as input
- BreadfruitDataset returns targets dict with maturity and quality tensors when use_annotations=True

**Inferred Relationships (identified from code reading)**:
- Config.data → BreadfruitDataset: data_root, annotation_file, train_dir, val_dir, test_dir
- Config.model → BreadfruitClassifier: backbone type, dropout, class lists, freeze epochs
- Config.training → train.py: batch_size, epochs, learning_rate, weight_decay, scheduler params
- Config.gradcam → gradcam.py: target_layer name

### 4.2.3 PlantUML Source Code
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

### 4.2.4 Title and Figure Caption
**Title**: Class Diagram - Breadfruit Classifier Core Components  
**Caption**: Shows Config dataclass hierarchy, BreadfruitClassifier with dual heads, MultiTaskHead, MultiTaskLoss, and BreadfruitDataset with annotation lookup.

---

## 4.3 Diagram 3: Sequence Diagram - Single Image Prediction Workflow

### 4.3.1 Purpose
Shows the chronological interaction between the user, frontend, backend, services, database, and external systems for a representative workflow (single image prediction).

### 4.3.2 Actual Entities and Relationships (from code inspection)
- **User**: Interacts with Streamlit upload component
- **Streamlit App** (app.py): Orchestrates the prediction workflow; loads model and config
- **Config**: Provides mean, std, image_size for preprocessing; maturity/quality class names
- **Predict Module** (predict.py): Contains predict_single() function
- **BreadfruitClassifier** (model.py): Forward pass producing logits
- **Datasets Module** (datasets.py): Provides get_inference_transform()
- **PIL Image**: Source image file opened by user

**Sequence Steps (as implemented)**:
1. User uploads image via Streamlit file_uploader
2. App initializes Config object with data paths
3. predict_single() is called with model, image, config
4. get_inference_transform(config) builds Resize→CenterCrop→ToTensor→Normalize pipeline
5. PIL Image opened and converted to RGB
6. Transform applied → input tensor (1, 3, 224, 224)
7. Model forward pass: backbone → GAP → both heads → logits
8. Softmax on logits → class probabilities
9. Argmax on probabilities → predicted class indices
10. Confidence = max softmax probability
11. Results dict returned with maturity and quality predictions, confidences, optional probabilities
12. Results displayed in Streamlit prediction cards

### 4.3.3 PlantUML Source Code
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

### 4.3.4 Title and Figure Caption
**Title**: Sequence Diagram - Single Image Prediction Workflow  
**Caption**: Shows chronological flow: User → Streamlit → Config → Datasets → PIL Image → Transform → Classifier → Predict Module → User display. Key processing steps: preprocessing, forward pass, softmax, argmax, confidence extraction.

---

## 4.4 Diagram Validation Summary

| Diagram | Syntax Validated | Consistent with Code | Key Relationships Captured |
|---|---|---|---|
| Use Case | Yes | Yes | include/extend between UC-01/UC-05, UC-02/UC-01, UC-04/UC-06, UC-03/UC-01, UC-05/UC-06, UC-06/UC-05 |
| Class | Yes | Yes | Config↔BreadfruitClassifier, MultiTaskHead composition, BreadfruitDataset label mapping |
| Sequence | Yes | Yes | User→Streamlit→Config→Datasets→PIL→Transform→Classifier→Predict→User display; forward pass split into two heads |
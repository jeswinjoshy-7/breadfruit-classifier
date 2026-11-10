# DELIVERABLE 5: Algorithm Descriptions, Pseudocode, and Flowcharts

## 5.1 Algorithm 1: Multi-Task Classification Forward Pass (BreadfruitClassifier.forward)

### 5.1.1 Algorithm ID
ALG-01

### 5.1.2 Purpose
Perform forward pass through EfficientNet-B0 backbone with dual classification heads for simultaneous maturity and quality prediction.

### 5.1.3 Input
- `x: torch.Tensor` - Preprocessed image tensor of shape (1, 3, 224, 224)

### 5.1.4 Output
- `dict` with keys "maturity_logits" (Tensor 3) and "quality_logits" (Tensor 3)

### 5.1.5 Preconditions
- Model is BreadfruitClassifier instance with pretrained EfficientNet-B0 backbone
- Input tensor has been normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Model is in eval() mode

### 5.1.6 Step-by-Step Processing Logic
1. Pass input tensor through EfficientNet-B0 backbone (with classifier replaced by nn.Identity)
2. Backbone outputs 1280-D feature vector via Global Average Pooling
3. Pass features through maturity_head (nn.Dropout(p=0.2) → nn.Linear(1280, 3))
4. Pass same features through quality_head (nn.Dropout(p=0.2) → nn.Linear(1280, 3))
5. Return dictionary {"maturity_logits": maturity_output, "quality_logits": quality_output}

### 5.1.7 Pseudocode
```
def forward(self, x):
    features = self.backbone(x)  # GAP + Identity → 1280-D vector
    out = {"maturity_logits": self.maturity_head(features)}
    if self.has_multi_task:
        out["quality_logits"] = self.quality_head(features)
    else:
        out["quality_logits"] = out["maturity_logits"]
    return out
```

### 5.1.8 Time Complexity
O(N) where N is the number of operations in EfficientNet-B0 backbone (~5.3M parameters) plus two trivial Linear(1280, 3) operations. Inference time ~200-500ms on CPU, ~50ms on GPU.

### 5.1.9 Space Complexity
O(P) where P = 5.3M (backbone parameters) + 2 × (1280 × 3 + 3) ≈ 7,686 (head parameters). Peak activation memory depends on intermediate feature maps during forward pass.

### 5.1.10 Error-Handling Behaviour
- If input tensor has wrong shape, PyTorch will raise RuntimeError during backbone forward
- If model has not been initialized with pretrained weights, backbone will still work with random weights (has_multi_task check prevents error)
- Division by zero not possible; argmax always produces valid index

### 5.1.11 Edge Cases
- Batch size > 1: Forward pass works on batched inputs; output logits shape scales accordingly
- Single sample (batch size 1): Works as implemented; unsqueeze(0) added before input in predict_single
- has_multi_task=False: quality_logits maturity_logits as fallback (line 92-93 in model.py)

### 5.1.12 Suggested Validation and Test Cases
- Forward pass with random tensor → verify output dict keys
- Forward pass with batch of 8 → verify output shapes
- Model.eval() mode → verify deterministic output for same input
- Gradient computation disabled (torch.no_grad()) → verify no backward pass needed for inference

### 5.1.13 Mermaid Flowchart
```mermaid
flowchart TD
    A[Input Tensor (1, 3, 224, 224)] --> B{EfficientNet-B0 Backbone}
    B --> C[Global Average Pooling → 1280-D Features]
    C --> D{Multi-Task Heads}
    D -->|Maturity Head| E[nn.Dropout(p=0.2)]
    E --> F[nn.Linear(1280, 3) → Maturity Logits]
    D -->|Quality Head| G[nn.Dropout(p=0.2)]
    G --> H[nn.Linear(1280, 3) → Quality Logits]
    E --> I[Output: {"maturity_logits", "quality_logits"}]
    G --> I
```

---

## 5.2 Algorithm 2: Multi-Task Loss Combination (MultiTaskLoss.forward)

### 5.2.1 Algorithm ID
ALG-02

### 5.2.2 Purpose
Compute weighted sum of cross-entropy losses for maturity and quality tasks.

### 5.2.3 Input
- `predictions: dict` with "maturity_logits" and "quality_logits"
- `targets: dict` with "maturity" (class index tensor) and "quality" (class index tensor)

### 5.2.4 Output
- `total_loss: Tensor` (scalar)
- `loss_dict: dict` with "total_loss", "maturity_loss", "quality_loss"

### 5.2.5 Preconditions
- Model has been in training mode
- Targets match the expected format (class indices, not one-hot)
- Maturity_weight and quality_weight are non-negative

### 5.2.6 Step-by-Step Processing Logic
1. Compute cross-entropy loss for maturity task: CE(logits_maturity, targets_maturity)
2. Compute cross-entropy loss for quality task: CE(logits_quality, targets_quality)
3. Weighted sum: total_loss = maturity_weight × maturity_loss + quality_weight × quality_loss
4. Return total_loss and individual loss components in dictionary

### 5.2.7 Pseudocode
```
def forward(self, predictions, targets):
    maturity_loss = CE(predictions["maturity_logits"], targets["maturity"])
    quality_loss = CE(predictions["quality_logits"], targets["quality"])
    total_loss = maturity_weight * maturity_loss + quality_weight * quality_loss
    loss_dict = {"total_loss": total_loss, "maturity_loss": maturity_loss, "quality_loss": quality_loss}
    return total_loss, loss_dict
```

### 5.2.8 Time Complexity
O(1) - two CrossEntropyLoss computations on batch of samples; constant time relative to model size.

### 5.2.9 Space Complexity
O(1) - scalar loss values; no significant additional memory.

### 5.2.10 Error-Handling Behaviour
- _safe_metric helper in evaluate.py returns 0.0 if len(set(y_true)) < 2 (single class present)
- CrossEntropyLoss will raise error if logits and targets have incompatible shapes

### 5.2.11 Edge Cases
- Weight of 0 for one task: If maturity_weight=0, only quality loss contributes (and vice versa)
- This scenario is configured in train.py:191-192 when model doesn't have multi-task

### 5.2.12 Suggested Validation and Test Cases
- Verify total_loss ≈ maturity_weight × maturity_loss + quality_weight × quality_loss
- Test with maturity_weight=1.0, quality_weight=0.0 → total_loss = maturity_loss only
- Test batch training step → verify gradients flow back to model parameters

### 5.2.13 Mermaid Flowchart
```mermaid
flowchart TD
    A[Predictions Dict] --> B[Extract maturity_logits]
    A --> C[Extract quality_logits]
    D[Targets Dict] --> E[Extract maturity targets]
    D --> F[Extract quality targets]
    B --> G[CrossEntropyLoss maturity]
    C --> H[CrossEntropyLoss quality]
    E --> G
    F --> H
    G --> I[maturity_weight × loss]
    H --> J[quality_weight × loss]
    I --> K[total_loss = I + J]
    J --> K
    K --> L[loss_dict: {total, maturity, quality}]
    K --> M[Return total_loss, loss_dict]
```

---

## 5.3 Algorithm 3: Dataset Sample Loading with Annotation Lookup (BreadfruitDataset._load_samples)

### 5.3.1 Algorithm ID
ALG-03

### 5.3.2 Purpose
Load image samples from directory and match them with maturity/quality labels from annotation CSV file.

### 5.3.4 Input
- Annotation CSV file path (may be None)
- Image directory containing split images
- Maturity class name → index mapping
- Quality class name → index mapping

### 5.3.5 Output
- `self.samples: List[Tuple[str, Optional[int], Optional[int]]]` - list of (filename, maturity_idx, quality_idx)

### 5.3.6 Preconditions
- Image directory exists and contains .jpg/.jpeg/.png files
- If annotation_file provided, it has columns: "filename", "maturity", "quality"
- Class names match configured maturity_classes and quality_classes lists

### 5.3.7 Step-by-Step Processing Logic
1. If annotation_file exists and is valid:
   a. Read CSV using csv.DictReader
   b. Build lookup dictionary: filename → (maturity_idx, quality_idx)
   c. Iterate through image directory files
   d. For each file, check if filename exists in lookup
   e. If match found, append (fname, mat_idx, qual_idx) to samples; set use_annotations=True
   f. If no samples match annotations, print warning and fall back to single-class mode
2. If no annotation_file or file not found:
   a. Print message about running in single-class mode
3. If self.samples is still empty after above:
   a. Fall back to loading all images without labels: (fname, None, None)

### 5.3.8 Pseudocode
```
def _load_samples(self, annotation_file):
    if annotation_file exists:
        lookup = {}
        read CSV row by row:
            fname = row["filename"]
            mat = row["maturity"] 
            qual = row["quality"]
            if fname and mat in maturity_to_idx and qual in quality_to_idx:
                lookup[fname] = (mat_idx, qual_idx)
        for fname in sorted(image directory listing):
            if fname endswith supported extensions:
                if fname in lookup:
                    mat, qual = lookup[fname]
                    self.samples.append((fname, mat, qual))
                    use_annotations = True
    if not self.samples:
        for fname in sorted(image directory listing):
            if fname endswith supported extensions:
                self.samples.append((fname, None, None))
```

### 5.3.9 Time Complexity
O(N + M) where N = number of annotation CSV rows, M = number of image files in directory. Lookup dictionary makes per-file check O(1).

### 5.3.10 Space Complexity
O(N + M) for storing lookup dictionary and samples list.

### 5.3.11 Error-Handling Behaviour
- If annotation CSV is malformed, csv.DictReader may raise error; caught by try/except or graceful handling
- If filename in image directory doesn't match any annotation entry, sample is added without labels (mat=None, qual=None)
- If annotation file has no matching entries, falls back to single-class mode with warning

### 5.3.12 Edge Cases
- Empty annotation CSV: No rows → lookup empty → all samples fall to single-class mode
- Image filename not in annotations: Sample added as (fname, None, None) → use_annotations remains False
- Multiple annotations for same filename: Last row wins in lookup dictionary (Dict behavior)

### 5.3.13 Suggested Validation and Test Cases
- Load dataset with valid annotation CSV → verify use_annotations=True and samples have labels
- Load dataset without annotation CSV → verify use_annotations=False and samples have (fname, None, None)
- Load dataset with mismatched filenames → verify some samples get (fname, None, None)
- Verify per-class label mapping correctness (immature→0, partially_mature→1, mature→2; poor→0, good→1, excellent→2)

### 5.3.14 Mermaid Flowchart
```mermaid
flowchart TD
    A{annotation_file exists?} -->|Yes| B[Read CSV via DictReader]
    B --> C[Build lookup: fname → (mat_idx, qual_idx)]
    C --> D[Iterate image directory]
    D --> E{Fname in lookup?}
    E -->|Yes| F[Append (fname, mat_idx, qual_idx); use_annotations=True]
    E -->|No| G[Continue loop]
    G --> D
    D -->|All processed| H{self.samples empty?}
    H -->|Yes| I[Fall back: append (fname, None, None) for all images]
    H -->|No| J[End]
    I --> J
    A -->|No| K[Run in single-class mode]
    K --> I
```

---

## 5.4 Algorithm 4: Grad-CAM Heatmap Generation (GradCAM.generate)

### 5.4.1 Algorithm ID
ALG-04

### 5.4.2 Purpose
Generate Grad-CAM heatmap for the given input image and target class, highlighting regions the model focuses on for its prediction.

### 5.4.3 Input
- `input_tensor: torch.Tensor` - Preprocessed image tensor (1, C, H, W)
- `class_idx: Optional[int]` - Target class index. If None, uses argmax.
- `task: str` - Which head to use ("maturity" or "quality").

### 5.4.4 Output
- Heatmap as a 2D numpy array (H, W) in range [0, 1]

### 5.4.5 Preconditions
- Model is BreadfruitClassifier instance with registered hooks on target layer
- input_tensor requires_grad_(True) for backward pass
- Target layer (features.8) is registered with forward/backward hooks

### 5.4.6 Step-by-Step Processing Logic
1. If self.activation is not None: self.activation = None (reset from previous call)
2. If self.gradient is not None: self.gradient = None (reset from previous call)
3. Move input_tensor to target device (model's device)
4. Set input_tensor.requires_grad_(True)
5. Forward pass: predictions = self.model(input_tensor)
6. Identify logits_key based on task ("maturity_logits" or "quality_logits")
7. If class_idx is None: class_idx = logits.argmax(dim=1).item()
8. Zero gradients: self.model.zero_grad()
9. Select score: score = logits[0, class_idx]
10. Backward pass: score.backward(retain_graph=True)
11. Compute weights: weights = self.gradient.mean(dim=(2, 3), keepdim=True)
12. Compute CAM: cam = (weights * self.activation).sum(dim=1, keepdim=True)
13. Apply ReLU: cam = F.relu(cam)
14. Convert to numpy: cam = cam.squeeze().cpu().numpy()
15. Resize: cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
16. Threshold: cam = np.maximum(cam, 0)
17. Normalize: if cam.max() > 0: cam = cam / cam.max()
18. Return heatmap

### 5.4.7 Pseudocode
```
def generate(self, input_tensor, class_idx=None, task="maturity"):
    if self.activation is not None: self.activation = None
    if self.gradient is not None: self.gradient = None
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)
    predictions = self.model(input_tensor)
    logits_key = f"{task}_logits"
    if class_idx is None:
        class_idx = logits.argmax(dim=1).item()
    self.model.zero_grad()
    score = logits[0, class_idx]
    score.backward(retain_graph=True)
    weights = self.gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * self.activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (W, H))
    cam = np.maximum(cam, 0)
    if cam.max() > 0: cam = cam / cam.max()
    return cam
```

### 5.4.8 Time Complexity
O(N) where N is the number of operations in one forward + backward pass through the backbone. Typically ~200-500ms on CPU.

### 5.4.9 Space Complexity
O(A) where A is the activation memory for the target layer (features.8 output feature maps). Additional O(H×W) for the heatmap output.

### 5.4.10 Error-Handling Behaviour
- If self.activation is None or self.gradient is None after forward/backward: RuntimeError("Gradient or activation not captured...")
- If task logits_key not in predictions: ValueError(f"Task '{task}' not found. Available: {list(predictions.keys())}")
- If target layer not found in model: ValueError(f"Layer '{self.target_layer}' not found in model")

### 5.4.11 Edge Cases
- Batch size > 1: Heatmap computed for first sample (index 0) due to [0] indexing in weight computation
- class_idx specified: Uses provided class index instead of argmax
- Multiple calls: Hooks are reset at start of each generate() call

### 5.4.12 Suggested Validation and Test Cases
- Generate heatmap with random tensor → verify output shape (H, W)
- Generate heatmap with class_idx specified → verify correct target class
- Generate heatmap in eval mode → verify deterministic output
- Verify Grad-CAM focuses on breadfruit region (not background)

### 5.4.13 Mermaid Flowchart
```mermaid
flowchart TD
    A[Reset activation/gradient hooks] --> B[Set requires_grad=True]
    B --> C[Forward pass through model]
    C --> D{Select logits_key}
    D -->|task="maturity"| E[maturity_logits]
    D -->|task="quality"| F[quality_logits]
    E --> G[score = logits[0, class_idx]]
    F --> G
    G --> H[score.backward()]
    H --> I[weights = gradient.mean(dim=2,3)]
    I --> J[cam = (weights × activation).sum(dim=1)]
    J --> K[cam = ReLU(cam)]
    K --> L[cam.squeeze().numpy()]
    L --> M[cam = cv2.resize(cam, (W, H))]
    M --> N[cam = maximum(cam, 0)]
    N --> O[if cam.max()>0: cam = cam/cam.max()]
    O --> P[Return heatmap (H, W) [0,1]]
```

---

## 5.5 Algorithm 5: Weighted Random Sampler for Imbalanced Data

### 5.5.1 Algorithm ID
ALG-05 (not explicitly in code but relevant to training)

### 5.5.2 Purpose
Address class imbalance in maturity/quality classification by assigning different sampling probabilities to training samples.

### 5.5.3 Input
- Dataset with imbalanced class distribution
- Optional weight parameter for each sample

### 5.5.3 Output
- DataLoader with WeightedRandomSampler for balanced batch composition

### 5.5.6 Pseudocode (from torch.utils.data)
```
from torch.utils.data import WeightedRandomSampler

# Compute per-class weights based on inverse frequency
class_weights = 1.0 / class_counts
sample_weights = [class_weights[target] for target in all_targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=pin_memory
)
```

### 5.5.7 Notes
- The current implementation (datasets.py, train.py) does not explicitly use WeightedRandomSampler
- Shuffle=True in DataLoader provides some imbalance mitigation
- Could be added as enhancement for severely imbalanced datasets

### 5.5.8 Mermaid Flowchart
```mermaid
flowchart TD
    A[Compute sample weights from class frequencies] --> B[Create WeightedRandomSampler]
    B --> C[DataLoader with sampler (not shuffle)]
    C --> D[Balanced batches per epoch]
    D --> E[Model trains on diverse batches]
```
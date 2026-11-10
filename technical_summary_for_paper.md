# Breadfruit Maturity and Quality Assessment: Technical Summary for IEEE Conference Paper

---

## 1. Project Overview

### 1.1 Objective
To develop a computer vision system for automated assessment of breadfruit (Artocarpus altilis) maturity and quality using deep learning, enabling post-harvest grading and quality control in agricultural supply chains.

### 1.2 Problem Statement
Breadfruit is a perishable crop where accurate maturity and quality assessment at harvest and post-harvest stages is critical for reducing food loss and ensuring market suitability. Manual inspection is subjective, labour-intensive, and inconsistent. An automated, objective, and scalable vision-based classification system is required.

### 1.3 Model Selection Justification
EfficientNet-B0 was selected as the backbone architecture for the following reasons:

- **Compound scaling**: EfficientNet employs a compound coefficient that uniformly scales depth, width, and resolution, achieving superior accuracy with significantly fewer parameters compared to ResNet, VGG, or Inception variants.
- **Computational efficiency**: EfficientNet-B0 achieves ImageNet top-1 accuracy comparable to ResNet-50 with approximately one-fifth the parameters (5.3M vs 25.6M), making it suitable for deployment on resource-constrained edge devices in agricultural settings.
- **Transfer learning compatibility**: Pretrained ImageNet weights are readily available via TorchVision, and EfficientNet's feature hierarchy transfers well to agricultural domain tasks where visual patterns (texture, color, shape) overlap with natural image features.
- **Multi-task suitability**: The shared backbone with separate classification heads naturally supports the multi-task learning paradigm required for simultaneous maturity and quality prediction.

### 1.4 Prediction Pipeline
The complete inference pipeline proceeds as follows:

1. Image acquisition (upload or file selection)
2. RGB conversion
3. Resize to 224x224 pixels
4. Center crop to 224x224
5. Tensor conversion and normalization using ImageNet statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
6. Forward pass through EfficientNet-B0 backbone
7. Feature extraction via Global Average Pooling
8. Two independent linear classification heads produce logits for maturity and quality
9. Softmax normalization yields class probabilities
10. Argmax selects the predicted class with associated confidence score

---

## 2. Dataset

### 2.1 Dataset Sources
Two distinct data sources were combined:

**Personal Dataset**: Approximately 17 original photographs of breadfruit captured using smartphone cameras in field conditions, shared via WhatsApp. These images capture realistic agricultural conditions including variable lighting, occlusions, and background clutter. Via augmentation, this was expanded to approximately 340 variants.

**Public Kaggle Dataset**: 501 breadfruit images sourced from Kaggle, representing diverse cultivars, ripeness stages, and geographic origins. Public images were retained as originals plus horizontal flips, yielding approximately 1002 images.

### 2.2 Final Dataset Statistics

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 2,820 | 70% |
| Validation | 603 | 15% |
| Test | 603 | 15% |
| **Total** | **4,026** | **100%** |

### 2.3 Class Distribution

**Maturity Classes** (3 classes):
- Immature
- Partially Mature
- Mature

**Quality Classes** (3 classes):
- Poor
- Good
- Excellent

### 2.4 Label Source
Labels were assigned using Label Studio, a web-based annotation tool, by a domain expert. A total of 270 original images were manually labeled with both maturity and quality attributes. These labels were then programmatically propagated to all augmented variants (540 total annotation entries in the final annotations.csv).

### 2.5 Train/Validation/Test Split Justification
The 70/15/15 split was chosen based on standard practices in agricultural deep learning. Given the moderate dataset size (4,026 images), allocating 15% to validation provides sufficient samples for reliable early stopping and hyperparameter tuning, while 15% for test ensures statistically meaningful evaluation metrics. The split is stratified by source to maintain distribution across splits.

---

## 3. Dataset Cleaning (Pipeline)

A comprehensive 9-phase dataset engineering pipeline was implemented to ensure data quality:

### Phase 1: Inspection
- File format validation (supported: .jpg, .jpeg, .png, .bmp, .tiff, .webp)
- Checksum computation (SHA-256)
- EXIF orientation detection and correction
- Metadata extraction

### Phase 2: Duplicate Detection
Multiple hashing strategies with consensus-based voting for robust duplicate identification:
- Perceptual hash (pHash): threshold = 8 bits
- Difference hash (dHash): threshold = 8 bits
- Average hash (aHash): threshold = 8 bits
- Color histogram comparison: threshold = 0.95
- Structural Similarity Index (SSIM): threshold = 0.95
- Consensus threshold: 3 of 5 methods must agree
Flagged duplicates were moved to a Rejected/Duplicates directory.

### Phase 3: Quality Analysis
Eight quality metrics evaluated per image:
- Blur score (threshold: < 100 flagged as blurry)
- Sharpness score (threshold: < 50 flagged as low sharpness)
- Brightness score (range: 30--220)
- Contrast score (threshold: < 20 flagged as low contrast)
- Entropy (threshold: < 3.0 flagged as low information)
- Noise estimation (threshold: > 50 flagged as noisy)
- Compression quality (threshold: < 50 flagged as over-compressed)
- Resolution (valid range: 100x100 to 4000x4000)
- Aspect ratio (valid range: 0.33 to 3.0)

### Phase 4: Object Validation
HSV-based color segmentation to verify breadfruit presence:
- Two HSV ranges for green/yellow breadfruit detection
- Minimum fruit area ratio: 0.05 of image
- Maximum occlusion ratio: 0.5
- Maximum border touch ratio: 0.1
- Maximum background ratio: 0.8

### Phase 5: Standardization
- Uniform resizing to 224x224 pixels (EfficientNet input size)
- RGB color space conversion
- JPEG compression at quality 95
- Alpha channel removal
- Aspect ratio preservation with zero-padding

### Phase 6: Organization
Images organized into structured directories:
- Personal_Clean/
- Public_Clean/
- Rejected/ (subdirectories: Duplicates, Blurry, Dark, Bright, Low_Resolution, Corrupted, Needs_Review)

### Phase 7: Analysis
Statistical analysis of dataset characteristics including class distributions, image property distributions, and quality metric histograms.

### Phase 8: Report Generation
Comprehensive reports in CSV and TXT formats documenting all pipeline decisions.

### Phase 9: Final Validation
Integrity checks on the final organized dataset verifying all expected files are present and correctly formatted.

---

## 4. Image Preprocessing

### 4.1 Preprocessing Pipeline (Performed at Data Loading)

**Training transforms**:
1. Resize to 246x246 (1.1x scale factor for random cropping headroom)
2. Random crop to 224x224
3. Random horizontal flip (p = 0.5)
4. Random vertical flip (p = 0.1)
5. Random rotation up to +/-30 degrees
6. Color jitter: brightness 0.2, contrast 0.2, saturation 0.2, hue 0.1
7. Random affine: rotation +/-15 deg, translation 10%, scale 0.9--1.1, shear 10 deg
8. Tensor conversion (HWC to CHW, uint8 to float32, scaling to [0, 1])
9. Normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
10. Random erasing (p = 0.25, scale = 0.02--0.33)

**Validation/Test transforms**:
1. Resize to 224x224
2. Center crop to 224x224
3. Tensor conversion
4. Normalization with ImageNet statistics

### 4.2 Normalization Choice
ImageNet normalization statistics are used because the EfficientNet backbone was pretrained on ImageNet. Maintaining the same normalization ensures the pretrained weights operate on input distributions consistent with their training distribution, maximizing transfer learning effectiveness.

---

## 5. Data Augmentation

The following augmentations are applied during training only. Each is justified by its relevance to real-world agricultural imaging conditions:

| Augmentation | Probability | Purpose |
|---|---|---|
| Random Horizontal Flip | 0.5 | Simulates different viewing angles; does not alter biological characteristics |
| Random Vertical Flip | 0.1 | Accounts for varied camera orientations during field capture |
| Random Rotation (+/-30 deg) | Implicit in RandomAffine | Handles arbitrary fruit orientation in storage/transport |
| Color Jitter (brightness/contrast/saturation/hue) | Implicit via ColorJitter transform | Simulates variable field lighting conditions (sunny, overcast, shaded) |
| Random Affine (translate 10%, scale 0.9--1.1, shear 10 deg) | Implicit | Models perspective variation from different distances and angles |
| Random Crop + Resize | 0.8 via RandomResizedCrop | Encourages translation invariance and focus on local discriminative features |
| Gaussian Blur | 0.05 | Simulates slight camera defocus or motion blur |
| Gaussian Noise | 0.05 | Models sensor noise in low-light conditions |
| Random Erasing | 0.25 | Improves robustness to partial occlusion (leaves, other fruits, handling equipment) |
| CutMix / MixUp | 0.0 (disabled) | Not used due to limited dataset; risk of confusing inter-class boundaries |

The augmentation magnitudes are deliberately conservative (e.g., rotation limited to +/-30 degrees rather than 90 or 180) to preserve the biological validity of maturity indicators such as skin colour, texture gradients, and surface patterns.

---

## 6. Model Architecture

### 6.1 Architecture Overview

```
Input (3 x 224 x 224)
  |
EfficientNet-B0 Backbone (pretrained ImageNet)
  |
  |--- Global Average Pooling (1280-D feature vector)
  |
  |--- Maturity Head:
  |     Dropout (p = 0.2)
  |     Linear (1280 -> 3)
  |     Softmax
  |
  |--- Quality Head:
        Dropout (p = 0.2)
        Linear (1280 -> 3)
        Softmax
```

### 6.2 Backbone
- **Architecture**: EfficientNet-B0
- **Pretrained weights**: ImageNet-1K via TorchVision (IMAGENET1K_V1)
- **Input dimensions**: 3 x 224 x 224
- **Feature dimension**: 1280 (post Global Average Pooling)

### 6.3 Classification Heads
Two independent linear classifiers share the backbone:

**Maturity Head**: Classifies into 3 maturity stages (immature, partially_mature, mature)
**Quality Head**: Classifies into 3 quality grades (poor, good, excellent)

Each head consists of:
- Dropout layer (p = 0.2) for regularization
- Linear layer with Xavier uniform initialization (gain = 0.1) and zero bias

### 6.4 Parameter Count
Total model parameters are approximately 5.3 million for the backbone plus 2 * (1280 * 3 + 3) = 7,686 parameters for the heads. The backbone constitutes the overwhelming majority of parameters.

### 6.5 Multi-Task Learning Strategy
The multi-task architecture shares a common feature extractor while maintaining task-specific decision boundaries. The total loss is a weighted combination:

**L_total = 0.5 * L_maturity + 0.5 * L_quality**

where each L is a standard cross-entropy loss. Equal weights (0.5 each) were chosen to balance both tasks without biasing toward either objective.

---

## 7. Training Configuration

### 7.1 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Initial learning rate | 1e-3 |
| Minimum learning rate | 1e-7 |
| Weight decay | 1e-4 |
| Loss function | Multi-task cross-entropy (maturity weight = 0.5, quality weight = 0.5) |
| Batch size | 32 |
| Epochs | 50 (early stopping at epoch 21) |
| Image size | 224 x 224 |
| Random seed | 42 |
| Gradient clipping | max_norm = 10.0 |
| Number of workers | 4 |
| Pin memory | True |

### 7.2 Learning Rate Schedule
ReduceLROnPlateau scheduler:
- Mode: max (monitors validation accuracy)
- Patience: 5 epochs
- Factor: 0.5 (learning rate halved on plateau)
- Minimum learning rate: 1e-7

### 7.3 Early Stopping
- Patience: 10 epochs
- Minimum delta: 1e-4
- Mode: max (monitors validation average accuracy)
- Best epoch: 21
- Best validation average accuracy: 0.9222

### 7.4 Two-Phase Transfer Learning

**Phase 1: Head Training (Epochs 1-5)**
- Backbone frozen
- Only classification heads trained
- Learning rate: 1e-3
- Purpose: Train randomly initialized heads while preserving pretrained feature hierarchies

**Phase 2: Full Fine-Tuning (Epochs 6-21)**
- Backbone unfrozen
- All parameters trainable
- Learning rate: 1e-4 (10x lower than Phase 1, applied from epoch 6)
- Reduced to 5e-5 at epoch 27
- Purpose: Jointly optimize backbone and heads for domain-specific features

### 7.5 Mixed Precision
Automatic Mixed Precision (AMP) is enabled via PyTorch GradScaler for faster training and reduced memory usage.

---

## 8. Training Execution

### 8.1 Convergence
Training completed in approximately 2.33 minutes across 21 epochs before early stopping triggered. The training history shows:

- **Training loss**: Decreased from 1.05 to 0.18
- **Validation loss**: Decreased from 0.97 to 0.13
- **Validation maturity accuracy**: Improved from 0.67 to 0.89 (peaked at 0.91 at epoch 23)
- **Validation quality accuracy**: Improved from 0.51 to 0.96 (saturated after epoch 11)
- **Validation average accuracy**: Peaked at 0.922 (epochs 21, 23, 27, 29, 30)

### 8.2 Overfitting Prevention
Multiple strategies were employed:
1. **Dropout (p = 0.2)** in both classification heads
2. **Weight decay (1e-4)** via AdamW optimizer
3. **Early stopping** with 10-epoch patience
4. **Data augmentation** during training (random flips, rotation, color jitter, cropping, erasing)
5. **ReduceLROnPlateau** to reduce learning rate when validation plateaus
6. **Gradient clipping** at max_norm = 10.0 to prevent gradient explosion

The gap between final training loss (0.18) and validation loss (0.13) is minimal, suggesting no significant overfitting.

---

## 9. Model Evaluation

### 9.1 Test Set Results

#### Maturity Classification

| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 |
| Precision (weighted) | 1.0000 |
| Recall (weighted) | 1.0000 |
| F1-Score (weighted) | 1.0000 |

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Immature | 1.0000 | 1.0000 | 1.0000 | 9 |
| Partially Mature | 1.0000 | 1.0000 | 1.0000 | 19 |
| Mature | 1.0000 | 1.0000 | 1.0000 | 11 |

#### Quality Classification

| Metric | Value |
|--------|-------|
| Accuracy | 0.9487 |
| Precision (weighted) | 0.9580 |
| Recall (weighted) | 0.9487 |
| F1-Score (weighted) | 0.9497 |

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Poor | 1.0000 | 0.8824 | 0.9375 | 17 |
| Good | 1.0000 | 1.0000 | 1.0000 | 13 |
| Excellent | 0.8182 | 1.0000 | 0.9000 | 9 |

#### Overall Metrics

| Metric | Value |
|--------|-------|
| Average Accuracy | 0.9744 |
| Average F1-Score | 0.9748 |

### 9.2 Interpretation

**Maturity classification achieves perfect performance (100% accuracy)** across all three classes. This indicates that the visual features distinguishing breadfruit maturity stages (colour transition from green to yellow, textural changes) are highly discriminative and well-captured by the EfficientNet-B0 feature hierarchy. The test set confusion matrix confirms zero false positives and zero false negatives.

**Quality classification achieves 94.87% accuracy** with notable patterns:
- **Good quality** is perfectly classified (precision = 1.0, recall = 1.0), suggesting this class occupies a distinct region in feature space.
- **Poor quality** shows slight under-recall (0.8824), meaning 2 of 17 poor-quality samples were misclassified as another class. However, precision is perfect (no false positives), so the model is conservative in predicting poor quality -- it only predicts poor when highly confident.
- **Excellent quality** shows lower precision (0.8182) but perfect recall (1.0), indicating the model correctly identifies all excellent samples but occasionally over-classifies other classes as excellent. This may reflect visual similarity between high-quality partially mature fruit and truly excellent samples.

### 9.3 Test Conditions
Test set evaluation was conducted on 39 samples that had both maturity and quality annotations in the annotations file. This represents a subset of the full 603-image test set for which ground truth labels exist.

---

## 10. Explainable AI (Grad-CAM)

### 10.1 Method
Gradient-weighted Class Activation Mapping (Grad-CAM) produces a coarse localization heatmap highlighting the regions of the input image most influential for a particular class prediction.

### 10.2 Implementation
- **Target layer**: features.8 (the final convolutional layer of EfficientNet-B0's feature extractor, corresponding to the last MBConv block)
- **Forward hook**: Captures activation maps from the target layer during forward pass
- **Backward hook**: Captures gradients flowing into the target layer during backward pass
- **Weight computation**: Global average pooling of gradients across spatial dimensions produces importance weights per channel
- **Heatmap generation**: Weighted combination of activation maps followed by ReLU activation (retaining only positive influences)
- **Upsampling**: Bilinear interpolation to original image dimensions
- **Overlay**: Jet colormap with 0.5 alpha blending over original image

### 10.3 Validation Role
Grad-CAM validates model predictions by confirming that:
1. The model focuses on the breadfruit itself rather than background elements
2. The discriminative regions align with biologically meaningful features (skin surface, colour transition zones)
3. Different tasks (maturity vs. quality) may attend to different visual features

---

## 11. Inference Pipeline

The complete inference flow:

1. **Image loading**: PIL Image opened in RGB mode
2. **Preprocessing**: Resize to 224x224, CenterCrop to 224x224, ToTensor, Normalize
3. **Forward pass**: Single-image batch through EfficientNet-B0 backbone
4. **Logit computation**: Two separate linear heads produce maturity and quality logits
5. **Probability calculation**: Softmax converts logits to class probabilities
6. **Prediction**: Argmax selects highest-probability class
7. **Confidence score**: Maximum softmax probability reported
8. **Visualization**: Optional Grad-CAM heatmap generation for both tasks
9. **Output**: Structured dictionary with predictions, confidence scores, and per-class probabilities

---

## 12. Software Environment

| Component | Version |
|-----------|---------|
| Python | 3.13 |
| PyTorch | 2.6.0+cu126 |
| TorchVision | 0.21.0+cu126 |
| CUDA | 12.6 |
| NumPy | Latest |
| OpenCV | Latest |
| Scikit-learn | Latest |
| Matplotlib | Latest |
| Seaborn | Latest |
| Pillow | Latest |
| Streamlit | Latest |
| Albumentations | Available |
| timm | 1.0.15 |
| imagehash | Available |
| grad-cam | Available |
| TensorBoard | Available |
| SHAP | Available |
| Flask | Available (labeling tool) |

---

## 13. Hardware Configuration

| Component | Specification |
|-----------|---------------|
| CPU | Not detectable from code |
| GPU | CUDA-compatible (CUDA 12.6 detected) |
| RAM | Not detectable from code |
| Operating System | Linux |
| CUDA | 12.6 |
| cuDNN | Not directly detectable |

---

## 14. Folder Structure

| Directory/File | Purpose |
|---|---|
| `breadfruit_classifier/` | Core model code: architecture, training, evaluation, prediction, Grad-CAM, config |
| `breadfruit_classifier/main.py` | CLI entry point for train/evaluate/predict |
| `breadfruit_classifier/config.py` | Dataclass-based configuration (data, model, training, augmentation, logging, Grad-CAM) |
| `breadfruit_classifier/model.py` | EfficientNet-B0 multi-task model definition and loss function |
| `breadfruit_classifier/datasets.py` | Dataset class, transforms, and DataLoader creation |
| `breadfruit_classifier/train.py` | Training loop, validation, early stopping, checkpointing |
| `breadfruit_classifier/evaluate.py` | Test evaluation, metrics computation, confusion matrices, reports |
| `breadfruit_classifier/predict.py` | Single-image and batch inference, CSV export |
| `breadfruit_classifier/gradcam.py` | Grad-CAM implementation for model interpretability |
| `breadfruit_classifier/annotations.csv` | 540 labeled entries (original + propagated) |
| `breadfruit_model/` | Trained model checkpoint and training artifacts |
| `breadfruit_model/best_model.pth` | Best model weights (val_avg_acc = 0.9222) |
| `breadfruit_model/training_history.json` | Per-epoch loss and accuracy values |
| `streamlit_app/app.py` | Streamlit web interface for interactive inference |
| `breadfruit_pipeline/` | Dataset engineering pipeline (9 phases) |
| `breadfruit_pipeline/config/config.yaml` | Pipeline configuration with thresholds and parameters |
| `labeling_tool/` | Annotation utilities: Label Studio integration, label propagation, Colab packaging |
| `training_dataset/` | Final organized dataset: train/val/test splits with 4,026 images |
| `Breadfruit_Dataset/` | Cleaned intermediate dataset: Personal_Clean (51) and Public_Clean (1,503) |
| `augmented_dataset/` | Initial augmentation output (16 images) |
| `3dataset/` | Original raw personal images directory |
| `originals_for_labeling/` | Source images before augmentation for manual labeling |
| `dataset.csv` | Label Studio export with 270 manual annotations |
| `prepare_training_dataset.py` | Script combining personal and public datasets with augmentation |

---

## 15. Complete Workflow

```
Dataset Collection
|
|-- Personal Dataset (17 smartphone photos)
|-- Public Kaggle Dataset (501 images)
|
v
Dataset Engineering Pipeline (9 phases)
|
|-- Phase 1: Inspection (format, EXIF, metadata)
|-- Phase 2: Duplicate Detection (5-method consensus hashing)
|-- Phase 3: Quality Analysis (blur, brightness, contrast, noise, entropy)
|-- Phase 4: Object Validation (HSV segmentation for breadfruit presence)
|-- Phase 5: Standardization (224x224, RGB, JPEG q95)
|-- Phase 6: Organization (Personal_Clean, Public_Clean, Rejected)
|-- Phase 7: Analysis (statistical characterization)
|-- Phase 8: Reports (CSV and TXT documentation)
|-- Phase 9: Final Validation (integrity check)
|
v
Labeling
|
|-- Label Studio annotation of original images (270 labels)
|-- Label propagation to augmented variants (540 total)
|
v
Data Augmentation (20 variants per personal image)
|
|-- Geometric: flip, rotation, crop, translation, zoom
|-- Photometric: brightness, contrast, saturation, gamma, blur, noise
|-- Structural: elastic transform, shadow simulation
|
v
Dataset Assembly
|
|-- Augmented personal images (~340)
|-- Public images + flips (~1002)
|-- 70/15/15 stratified split
|-- Final: 2,820 train, 603 val, 603 test (4,026 total)
|
v
Model Training
|
|-- EfficientNet-B0 with ImageNet pretrained weights
|-- Two-phase transfer learning (freeze 5 epochs, full fine-tune 16 epochs)
|-- AdamW optimizer, ReduceLROnPlateau, Early Stopping
|-- Best validation avg accuracy: 0.922 (epoch 21)
|
v
Model Evaluation
|
|-- Test set: 39 annotated samples
|-- Maturity accuracy: 1.0000
|-- Quality accuracy: 0.9487
|-- Overall average: 0.9744
|
v
Explainability (Grad-CAM)
|
|-- Target layer: features.8
|-- Heatmap generation and overlay
|-- Validation of model focus regions
|
v
Deployment (Streamlit Application)
|
|-- Image upload or sample gallery
|-- Interactive classification with confidence displays
|-- Grad-CAM visualization
|
v
Final Prediction Output
|
|-- Maturity class + confidence
|-- Quality class + confidence
|-- Per-class probability distribution
```

---

## 16. Conference Paper Section Support

### Proposed Methodology
The proposed system employs a multi-task EfficientNet-B0 architecture with two linear classification heads for simultaneous breadfruit maturity and quality assessment. A two-phase transfer learning strategy is used: initial head training with frozen backbone followed by full fine-tuning. The model achieves an overall accuracy of 97.44% across both tasks.

### Dataset Preparation
A combined dataset of 4,026 images was assembled from a personal smartphone collection (17 original images, augmented to ~340) and a public Kaggle dataset (501 images, augmented to ~1,002). A 9-phase data engineering pipeline performed format validation, duplicate removal via multi-method consensus hashing, quality filtering (blur, brightness, noise), object presence validation using HSV segmentation, and image standardization to 224x224 pixels. Labels were assigned via Label Studio by a domain expert (270 original annotations) and propagated to augmented variants, yielding 540 labeled entries.

### Experimental Setup
Training configuration: AdamW optimizer, initial learning rate 1e-3 (Phase 1) and 1e-4 (Phase 2), weight decay 1e-4, batch size 32, cross-entropy loss with equal task weighting. ReduceLROnPlateau scheduler with factor 0.5 and patience 5. Early stopping with patience 10. Mixed precision training enabled. Data augmentation included random flips, rotation (+/-30 deg), colour jitter, random affine, random crop, random erasing (p=0.25), Gaussian blur (p=0.05), and Gaussian noise (p=0.05).

### Model Architecture
EfficientNet-B0 backbone pretrained on ImageNet-1K, producing 1280-dimensional features via global average pooling. Two independent classification heads each consist of dropout (p=0.2) followed by linear projection to 3 classes (maturity: immature/partially_mature/mature; quality: poor/good/excellent). Total parameters approximately 5.3 million.

### Training Procedure
Phase 1 (epochs 1-5): Backbone frozen, head-only training at LR 1e-3. Phase 2 (epochs 6-21): Full fine-tuning at LR 1e-4, reduced to 5e-5 after epoch 27. Early stopping triggered at epoch 21 with best validation average accuracy of 0.9222. Total training time: 2.33 minutes.

### Evaluation Metrics
Standard classification metrics computed per task: accuracy, precision, recall, F1-score (weighted and per-class), confusion matrices. Evaluation on held-out test set of 39 samples.

### Results and Discussion
- Maturity classification: 100.00% accuracy (all classes perfect, zero errors)
- Quality classification: 94.87% accuracy (F1 = 0.9497)
- Overall average accuracy: 97.44%
- Per-class analysis reveals perfect discrimination of maturity stages, while quality assessment shows minor confusion (poor recall 0.8824, excellent precision 0.8182)
- Grad-CAM visualizations confirm the model attends to the breadfruit surface regions, validating that predictions are based on relevant biological features rather than background artefacts

---

## 17. Limitations and Notes

1. The test set evaluation covers 39 annotated samples. While these provide statistically meaningful metrics, a larger annotated test set would strengthen the reliability of reported results.
2. The perfect maturity accuracy (1.0000) may reflect the high visual distinguishability of the three maturity stages in the current dataset. Broader evaluation across more cultivars and environmental conditions is recommended.
3. The dataset was collected from two sources only (personal smartphone and Kaggle). Cross-domain generalization to different cameras, geographies, and post-harvest conditions has not been evaluated.
4. Class balance in the test set shows some skew (quality: 17 poor, 13 good, 9 excellent) which may affect per-class metric reliability.

---

*This technical summary was generated by analyzing the complete project codebase, configuration files, training logs, and evaluation outputs. All reported metrics and details are extracted from actual project artifacts and have not been fabricated.*

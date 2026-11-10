# DELIVERABLE 11: API and External Integration Documentation

## 10.1 Internal Module Interfaces (import-based)

| Module | Exported Functions/Classes | Usage |
|---|---|---|
| `breadfruit_classifier/config.py` | Config dataclass, DataConfig, ModelConfig, TrainingConfig, AugmentationConfig, LoggingConfig, GradCAMConfig | Imported by all other modules; central configuration |
| `breadfruit_classifier/model.py` | BreadfruitClassifier, MultiTaskHead, MultiTaskLoss, create_model(), count_parameters(), get_feature_extractor() | Imported by train.py, evaluate.py, predict.py, gradcam.py, main.py |
| `breadfruit_classifier/datasets.py` | BreadfruitDataset, get_transforms(), get_dataloaders(), get_inference_transform() | Imported by train.py, evaluate.py, predict.py, main.py, streamlit_app/app.py |
| `breadfruit_classifier/train.py` | train_epoch(), validate(), train_model(), EarlyStopping, _save_checkpoint(), load_checkpoint() | Imported by main.py |
| `breadfruit_classifier/evaluate.py` | evaluate_model(), _compute_metrics(), plot_confusion_matrices(), plot_training_history(), print_evaluation_report() | Imported by main.py |
| `breadfruit_classifier/predict.py` | load_model_for_inference(), predict_single(), predict_folder(), visualize_prediction() | Imported by streamlit_app/app.py |
| `breadfruit_classifier/gradcam.py` | GradCAM class, visualize_gradcam(), generate_gradcam_batch() | Imported by streamlit_app/app.py |
| `breadfruit_pipeline/run_pipeline.py` | run_pipeline() | Standalone entry point |
| `breadfruit_pipeline/config/config.yaml` | Pipeline configuration (phase flags, thresholds, paths) | Read by run_pipeline.py |

---

## 10.2 API Endpoints

### 10.2.1 HTTP REST Endpoints
**The project does NOT expose any HTTP REST endpoints, FastAPI routes, or Flask APIs.**

All communication is internal via Python imports within the same process. There is no web server, no Flask app, no FastAPI instance, no uvicorn configuration.

### 10.2.2 Streamlit Internal "Endpoints"

The Streamlit app (`streamlit_app/app.py`) provides the following internal endpoints via function calls (note: these are not HTTP endpoints but in-process function calls):

| Endpoint | Method/Type | Description |
|---|---|---|
| `load_model()` | GET (cached, @st.cache_resource) | Load model + config from checkpoint; returns (model, config) |
| `predict_single()` | Callable | Predict maturity + quality for single image; returns dict |
| `predict_folder()` | Callable | Predict for all images in folder; optionally CSV export |
| `render_prediction_card()` | Streamlit component | Display prediction result card with confidence bar |
| `render_gradcam_tab()` | Streamlit component | Display Grad-CAM heatmaps for both tasks |

### 10.2.3 CLI Subcommands (main.py)

| Subcommand | Arguments | Description |
|---|---|---|
| `train` | --epochs, --batch_size, --lr, --experiment, --data_root, --annotation_file | Train a new model |
| `evaluate` | --checkpoint | Evaluate a trained model on test set |
| `predict` | --image, --folder, --output_csv, --visualize, --probabilities | Run inference on images |

---

## 10.3 External Integrations

| Integration | Purpose | Status | Details |
|---|---|---|---|
| **Label Studio** | Manual annotation of 270 original breadfruit images | Mentioned in technical_summary_for_paper.md; **not integrated** in current codebase; annotations stored in `breadfruit_classifier/annotations.csv` | Domain expert assigns maturity/quality labels via web UI; export → CSV (270 rows) |
| **Kaggle Dataset** | 501 breadfruit images for public dataset | **Downloaded and stored** in `3dataset/` and `training_dataset/` directories | Personal collection + Kaggle combined to create 4,026-image dataset |
| **PyTorch/TorchVision** | EfficientNet-B0 pretrained on ImageNet | **Integrated** via `models.EfficientNet_B0_Weights.IMAGENET1K_V1` | Backbone initialization; feature extractor with GAP → 1280-D features |
| **CSV Annotation File** | 540 entries (270 original + propagated) | **Integrated** as `annotation_file` in DataConfig | Read by `BreadfruitDataset._load_samples()`; columns: filename, maturity, quality |
| **Albumentations** | Image augmentation library | **Available** (listed in technical_summary) | Not explicitly imported in classifier code; pipeline uses OpenCV+NumPy custom augmentations |

---

## 10.4 Request/Response Formats (Inference)

Since no REST API exists, request/response formats are internal to the Python process.

### 10.4.1 predict_single() Input/Output

**Input**:
- Model: BreadfruitClassifier in eval mode
- Image: PIL Image (RGB) or str/Path to image file
- Config: Config dataclass with data, model, training parameters

**Output** (when return_probs=True):
```json
{
  "maturity": {
    "prediction": "mature",          // class name string
    "class_id": 2,                   // integer index (0=immature, 1=partially_mature, 2=mature)
    "confidence": 0.9876,            // float 0-1
    "probabilities": {               // optional, included when return_probs=True
      "immature": 0.0012,
      "partially_mature": 0.0034,
      "mature": 0.9876
    }
  },
  "quality": {
    "prediction": "good",            // class name string
    "class_id": 1,                   // integer index (0=poor, 1=good, 2=excellent)
    "confidence": 0.9543,            // float 0-1
    "probabilities": {
      "poor": 0.0234,
      "good": 0.9543,
      "excellent": 0.0223
    }
  }
}
```

### 10.4.2 predict_folder() Output

**Output**: List of dicts, each containing:
- `filename`: image filename (string)
- `maturity_prediction`: predicted maturity class (string)
- `maturity_confidence`: confidence score (float)
- `maturity_class_id`: class index (int)
- `quality_prediction`: predicted quality grade (string)
- `quality_confidence`: confidence score (float)
- `quality_class_id`: class index (int)
- `probabilities`: optional JSON string with per-class probabilities for both tasks

**CSV Export format** (when output_csv specified):
```
filename,maturity_prediction,maturity_confidence,maturity_class_id,quality_prediction,quality_confidence,quality_class_id,probabilities
01.JPG,mature,0.9876,2,good,0.9543,1,"{""maturity"": {""immature"": 0.0012, ""partially_mature"": 0.0034, ""mature"": 0.9876}, ""quality"": {""poor"": 0.0234, ""good"": 0.9543, ""excellent"": 0.0223}}"
```

---

## 10.5 Rate Limits

**No external APIs** → **no rate limits** apply. The project is a local-only application with no network communication.

### 10.5.1 Internal Rate Considerations
- Streamlit app runs on `localhost:8501` only
- No concurrent user handling; single-user deployment
- No API throttling or request quotas

---

## 10.6 Timeout and Retry Behaviour

### 10.6.1 Model Inference
- **No explicit timeout**: PyTorch forward pass completes in ~200ms-500ms on CPU, ~50ms on GPU
- **No retry mechanism**: If error occurs, user must re-upload image or re-click "Classify"
- **Typical latency**: 200-500ms for single image prediction on CPU; ~50ms on GPU

### 10.6.2 Training
- **No retry mechanism**: If error occurs (OOM, etc.), training halts; no automatic restart
- **Gradient clipping**: max_norm=10.0 prevents gradient explosion
- **Early stopping**: patience=10 epochs; training stops if val accuracy doesn't improve
- **Checkpoint saving**: Best model saved when improvement detected; allows resumption

### 10.6.3 Pipeline Phases
- **continue_on_error: true** in config.yaml allows skipping failed images
- **Individual phase errors**: Pipeline may continue past single image errors
- **Re-run capability**: Pipeline can be re-run to retry failed phases

### 10.6.4 Streamlit
- **No automatic retry**: Button click re-executes prediction
- **spinner** indicates processing time (typically < 1s)

---

## 10.7 Fallback Behaviour

### 10.7.1 Present Fallback Mechanisms

| Scenario | Fallback | Code Location |
|---|---|---|
| No annotation CSV | Single-class mode (maturity only) | `BreadfruitDataset._load_samples()` - falls back with print warning |
| No test dataloader | test_loader = None | `get_dataloaders()` - sets test_loader=None if test_dir not configured or empty |
| Single-head mode | quality_logits falls back to maturity_logits | `model.py:92-93` - `if self.has_multi_task: ... else: out["quality_logits"] = out["maturity_logits"]` |
| Pipeline phase skipping | Phase disabled via config.yaml | `run_pipeline.py` - each phase has `if config.phaseX.enabled` guard |
| Best vs last checkpoint | Try best_model.pth first; fall back to final model | `main.py:cmd_train`, `cmd_evaluate`, `cmd_predict` - checkpoint existence check |

### 10.7.2 Missing Fallback Mechanisms (Not Implemented)

- No fallback for OOM during inference
- No fallback for corrupted image in upload (currently error message, then halt)
- No fallback for missing dependencies (if PyTorch layer fails)
- No automatic training resume from interrupted session (checkpoint exists but may not be at exact epoch)

---

## 10.8 Error-Handling Strategy

### 10.8.1 Error Handling Patterns (from code inspection)

| Module | Pattern | Quality |
|---|---|---|
| `predict_single()` | Caller (Streamlit) handles exceptions; `Image.open()` may raise | Basic |
| `predict_folder()` | Has try/except around each image: `except Exception as e: logger.error(...); continue` | Good - continues past errors |
| `BreadfruitDataset._load_samples()` | No try/except around CSV reading; falls back with print() warnings | Minimal |
| `run_pipeline.py` | `continue_on_error: true` in config; each phase has `if config.phaseX.enabled` guards | Good - configurable |
| `train_model()` | No try/except around entire training loop; individual errors may halt training | Limited |
| `main.py` cmd functions | Check for checkpoint existence; `sys.exit(1)` on missing checkpoint; `argparse` errors | Good for CLI |
| `streamlit_app/app.py` | `try/except` in `render_gradcam_tab()`: `except Exception as e: print(...);` UI continues | Good - graceful degradation |

### 10.8.2 Common Patterns
- `logger.error()` for pipeline errors
- `print()` for CLI/user feedback
- `try/except` in predict_folder() per-image (allows batch to continue)
- `sys.exit(1)` for critical failures (missing checkpoint)
- Streamlit `st.error()`, `st.spinner()` for user-facing feedback

### 10.8.3 Uncaught Exception Risks
- `KeyboardInterrupt` (Ctrl+C) during training: may leave partial checkpoint; `best_model.pth` may or may not be saved depending on timing
- `RuntimeError: CUDA out of memory`: may crash training/prediction; no graceful retry
- `KeyError` in annotation lookup: _load_samples falls back via `.get()` but may still have edge cases
- `FileNotFoundError`: handled in some places (load_checkpoint, predict_folder), not in all

---

## 10.9 Data Validation

### 10.9.1 Present Validations

| Validation Type | Present | Location | Details |
|---|---|---|---|
| File format validation | Yes | Streamlit file_uploader accepts `type=["jpg", "jpeg", "png"]`; predict_single checks file extension | Limited to supported formats; other formats may cause PIL errors |
| Image load validation | Yes | `predict_single`: `Image.open(image).convert("RGB")` may raise exception for corrupt files; caught in try/except | Raises exception if file is not a valid image |
| Annotation CSV format | Partially | `BreadfruitDataset._load_samples`: uses `row.get("filename", "")` with default; safe access | May silently skip malformed rows via `.get()` |
| Transform input | Yes | `get_inference_transform()` always produces same compose; `predict_single` applies transform then model | Assumes input is valid PIL Image |
| CLI argument validation | Yes | `argparse` in main.py; `--checkpoint`, `--image`, `--folder` validated; missing required args cause error | `required=True` on subparsers; individual arg defaults handle optional params |

### 10.9.2 Missing Validations

- No validation that image content actually contains breadfruit
- No validation of image resolution/minimum size beyond pipeline quality checks
- No validation that annotation CSV class values match expected list (beyond dict key lookup)
- No batch size validation beyond config.training.batch_size = 32

### 10.9.3 Validation Error Consequences
- Wrong format → error message displayed; user must correct input
- Corrupt image → exception raised; pipeline/logger captures error; user must upload different image
- Malformed annotation → falls back to single-class mode with warning print; some training samples lose labels
- Missing CLI arg → argparse error message; command usage displayed
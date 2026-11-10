# DELIVERABLE 13: Deployment and Execution Guide

## 12.1 System Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Operating System | Linux (tested) | Linux (any modern distro) |
| Python | 3.13 | 3.13 (with venv or conda) |
| PyTorch | 2.6.0+cu126 | 2.6.0+cu126 (with CUDA 12.6) |
| CUDA | 12.6 (GPU) | 12.6 (if using GPU) |
| CPU | Any modern CPU | Multi-core (4+ cores recommended for training) |
| RAM | Not specified from code | 16GB+ (for 4,026 images + training) |
| Disk Space | Not specified | 5GB+ (for dataset, checkpoints, logs) |
| Python Packages | See requirements.txt | All packages from requirements.txt |

## 12.2 Runtime Versions (from code inspection)

| Component | Version | Source |
|---|---|---|
| Python | 3.13 | `technical_summary_for_paper.md:411` |
| PyTorch | 2.6.0+cu126 | `technical_summary_for_paper.md:412` |
| TorchVision | 0.21.0+cu126 | `technical_summary_for_paper.md:413` |
| CUDA | 12.6 | `technical_summary_for_paper.md:414` |
| NumPy | Latest | `technical_summary_for_paper.md:415` |
| OpenCV | Latest | `technical_summary_for_paper.md:416` |
| Scikit-learn | Latest | `technical_summary_for_paper.md:417` |
| Matplotlib | Latest | `technical_summary_for_paper.md:418` |
| Seaborn | Latest | `technical_summary_for_paper.md:419` |
| Pillow | Latest | `technical_summary_for_paper.md:420` |
| Streamlit | Latest | `technical_summary_for_paper.md:421` |
| Albumentations | Available | `technical_summary_for_paper.md:422` |
| timm | 1.0.15 | `technical_summary_for_paper.md:423` |
| imagehash | Available | `technical_summary_for_paper.md:424` |
| grad-cam | Available | `technical_summary_for_paper.md:425` |
| TensorBoard | Available | `technical_summary_for_paper.md:426` |
| SHAP | Available | `technical_summary_for_paper.md:427` |
| Flask | Available (labeling tool) | `technical_summary_for_paper.md:428` |

## 12.3 Installation Instructions

### 12.3.1 Step-by-Step Installation

**From Code Inspection**:

1. **Clone or obtain the project**: Ensure all files from `/home/jeswin/data_augmentation/` are present

2. **Create Python environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

3. **Install dependencies**:
   ```bash
   pip install -r breadfruit_classifier/requirements.txt
   pip install -r streamlit_app/requirements.txt
   # Pipeline dependencies covered by main requirements
   ```

4. **Verify package versions** (from technical_summary):
   - PyTorch 2.6.0+cu126
   - TorchVision 0.21.0+cu126
   - CUDA 12.6
   - Streamlit (latest at time of development)

5. **Place dataset files** in expected locations:
   - Raw images in `3dataset/` (personal) and appropriate paths in config.yaml
   - Annotation CSV at `breadfruit_classifier/annotations.csv`
   - Test images in `training_dataset/test/breadfruit/`

6. **Verify directory structure** matches config expectations:
   - `config.py:data.data_root` → should point to dataset root
   - `config.py:data.train_dir` → "train"
   - `config.py:data.val_dir` → "val"
   - `config.py:data.test_dir` → "test" (or empty for no test set)
   - `config.py:data.annotation_file` → path to annotations.csv

### 12.4 Environment Variables

#### 12.4.1 Currently Used Variables
**None explicitly used** - all paths hardcoded in config.py and config.yaml.

#### 12.4.2 Path Configuration (hardcoded, not via environment variables)

| Path | Default Value | Module |
|---|---|---|
| `data_root` | `/home/jeswin/data_augmentation` | config.py:11 |
| `annotation_file` | `""` (empty, must be set) | config.py:12 |
| `checkpoint_dir` | `checkpoints/` | config.py:73 |
| `log_dir` | `logs/` | config.py:74 |
| `plot_dir` | `plots/` | config.py:75 |
| `output.base_dir` | `/home/jeswin/data_augmentation/Breadfruit_Dataset` | config.yaml:15 |
| `model.best_path` | `breadfruit_model/best_model.pth` | streamlit_app/app.py:210 |

#### 12.4.3 Suggested Environment Variables (not implemented)
```bash
export BREADFRUIT_DATA_ROOT="/path/to/datasets"
export BREADFRUIT_CHECKPOINT_DIR="/path/to/checkpoints"
export BREADFRUIT_MODEL_DIR="/path/to/breadfruit_model"
export BREADFRUIT_CONFIG_DATA_ROOT="/path/to/datasets"
export BREADFRUIT_SEED=42

# For Streamlit deployment
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
```

#### 12.4.4 How to Make Paths Configurable via Environment (recommended enhancement)
```python
# In config.py, modify DataConfig initialization:
import os
class DataConfig:
    data_root: str = os.getenv("BREADFRUIT_DATA_ROOT", "/home/jeswin/data_augmentation")
    annotation_file: str = os.getenv("BREADFRUIT_ANNOTATION_CSV", "")
    # ... rest of fields
```

---

## 12.5 Database Initialization
**Not applicable** - project uses file-based persistence, not a relational database.

### 12.5.1 Data Directory Setup
Ensure the following directory structure exists (or is created by the pipeline):

```
<data_root>/
  train/
    breadfruit/    (training images)
  val/
    breadfruit/    (validation images)
  test/
    breadfruit/    (test images, optionally)
  annotations.csv  (maturity, quality labels)
```

### 12.5.2 Pipeline Output Directory Setup
The 9-phase pipeline creates:
```
<output_base_dir>/
  standardized/      (phase 5 output)
  Reports/           (phase 8 reports)
  Logs/              (phase 9 validation logs, pipeline.log)
  metadata.json      (pipeline phase 9 output)
  metadata.json      (pipeline metadata)
```

---

## 12.6 Build Commands

### 12.6.1 Training
```bash
source .venv/bin/activate
python main.py train
# Or with overrides:
python main.py train --epochs 50 --batch_size 32 --lr 1e-3
```

### 12.6.2 Evaluation
```bash
source .venv/bin/activate
python main.py evaluate --checkpoint breadfruit_model/best_model.pth
# Or specify custom checkpoint:
python main.py evaluate --checkpoint checkpoints/epoch_025.pth
```

### 12.6.3 Single Image Prediction
```bash
source .venv/bin/activate
python main.py predict --image path/to/breadfruit.jpg
# With Grad-CAM visualization:
python main.py predict --image path/to/breadfruit.jpg --visualize
# With class probabilities:
python main.py predict --image path/to/breadfruit.jpg --probabilities
# Batch prediction:
python main.py predict --folder path/to/images/ --output_csv results.csv
```

### 12.6.4 Batch Prediction
```bash
source .venv/bin/activate
python main.py predict --folder path/to/test_images/ --output_csv results.csv
# With probabilities in CSV:
python main.py predict --folder path/to/test_images/ --output_csv results.csv --probabilities
```

### 12.6.5 Dataset Engineering Pipeline
```bash
source .venv/bin/activate
python run_pipeline.py
# With custom config:
python run_pipeline.py /path/to/custom_config.yaml
# Phase-specific control via config.yaml enable flags
```

---

## 12.7 Application Startup Commands

### 12.7.1 Streamlit Web Interface
```bash
source .venv/bin/activate
cd /home/jeswin/data_augmentation
streamlit run streamlit_app/app.py --server.port 8501 --server.headless false --server.address localhost
# start_app.sh equivalent:
bash start_app.sh
```

### 12.7.2 CLI Entry Points
All accessible via `python main.py --help`:
```
commands:
  train       Train a new model
  evaluate    Evaluate a trained model
  predict     Run inference on images
```

---

## 12.8 Deployment Architecture

### 12.8.1 Current Deployment Model
- **Local only**: Streamlit app runs on `localhost:8501`
- **Single-user**: No multi-user support or authentication
- **No containerization**: No Dockerfile, no Kubernetes configuration
- **No production hardening**: No security middleware, no load balancing

### 12.8.2 Recommended Deployment Enhancements
1. **Dockerize the application**: Create Dockerfile with Python base image
2. **Add authentication**: Streamlit OAuth or token-based auth
3. **GPU support in production**: Ensure CUDA drivers and runtime match development environment
4. **Monitoring**: Add logging infrastructure, health check endpoints
5. **Scale considerations**: Current design limited to single-image batch size 32 training, ~1 image/second inference on CPU

---

## 12.9 Docker Configuration
**Not present** in the project. No Dockerfile, docker-compose.yml, or container configuration found.

### 12.9.1 Suggested Dockerfile (not implemented)
```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.headless=true"]
```

---

## 12.10 CI/CD Pipeline
**Not present** in the project. No CI/CD configuration, no GitHub Actions, no automated test pipelines found.

### 12.10.1 Suggested CI/CD Pipeline (not implemented)
1. **Unit tests**: Run pytest suite on each commit
2. **Integration tests**: Verify end-to-end workflows (train → evaluate → predict)
3. **Linting**: Check code style ( Ruff, flake8)
4. **Security scan**: Detect vulnerable dependencies
5. **Build and push Docker image**: On merge to main branch
6. **Deploy**: Deploy to staging/production environments

---

## 12.11 Logging and Monitoring

### 12.10.1 Existing Logging
- **Pipeline**: `logs/pipeline.log` with phase progression and image counts
- **CLI**: Console output via print statements
- **Streamlit**: Runtime messages via `st.spinner()`, `st.error()`, `st.info()`

### 12.10.2 Monitoring Gaps
- No structured JSON logging
- No correlation IDs for request tracing
- No log rotation configuration
- No health check endpoints
- No performance metrics collection
- No alerting on failures

### 12.10.3 Suggested Monitoring (not implemented)
- Add JSON-structured logs with timestamps and component identifiers
- Implement health check endpoint for Streamlit
- Add metrics collection (inference latency, error rates)
- Set up log rotation and archival

---

## 12.12 Common Runtime Failures

| Failure Mode | Symptoms | Root Cause | Resolution |
|---|---|---|---|
| Out of memory (OOM) | Training crashes; CUDA out of memory error | Batch too large; GPU insufficient for model + data | Reduce batch size; use CPU mode; enable AMP |
| Missing checkpoint | "Error: No checkpoint specified and best_model.pth not found." | Training not run; interrupted before save | Run training; or train from scratch |
| File not found | FileNotFoundError for images/annotations | Wrong paths in config; files moved/deleted | Verify directory structure; check config.yaml paths |
| Corrupt image | PIL.UnidentifiedImageError; "Failed to load image" | Damaged image file; incomplete download | Re-obtain image; check integrity |
| Annotation mismatch | Some samples lose labels; single-class mode activated | CSV filenames don't match image directory filenames | Ensure annotation CSV filenames match image filenames |
| Grad-CAM failure | "Grad-CAM failed for task 'maturity'" | Model not in eval mode; hooks not properly registered | Ensure model.eval(); check target layer exists |
| Device mismatch | Tensor on CPU but model on CUDA (or vice versa) | Model loaded on different device than data | Consistent device placement; use config.device |
# DELIVERABLE 1: Project Overview and Technology Stack

## 1.1 Project Name
Breadfruit Maturity and Quality Assessment System

## 1.2 Project Domain
Agricultural computer vision / Deep learning-based crop classification

## 1.3 Problem Statement
Breadfruit is a perishable crop where accurate maturity and quality assessment at harvest and post-harvest stages is critical for reducing food loss and ensuring market suitability. Manual inspection is subjective, labour-intensive, and inconsistent. An automated, objective, and scalable vision-based classification system is required.

## 1.4 Main Objectives
- Develop a computer vision system for automated breadfruit maturity assessment
- Develop a system for automated breadfruit quality grading
- Enable post-harvest grading and quality control in agricultural supply chains
- Provide objective, consistent classification replacing manual inspection

## 1.5 Intended Users and Stakeholders
- Agricultural workers and farm managers for field grading
- Post-harvest processing facilities
- Agricultural supply chain stakeholders
- Researchers in agricultural computer vision

## 1.6 Existing Features and Modules
- Data augmentation pipeline (9-phase dataset engineering)
- Multi-task deep learning classifier (EfficientNet-B0 with dual heads)
- Streamlit web application for interactive inference
- Grad-CAM visualizations for model interpretability
- Dataset organization and cleaning pipeline
- Training, evaluation, and prediction workflows

## 1.7 Programming Languages
- Python 3.13

## 1.8 Frameworks and Libraries
- PyTorch 2.6.0+cu126
- TorchVision 0.21.0+cu126
- Streamlit (for web interface)
- OpenCV, NumPy, Pillow
- Scikit-learn, Matplotlib, Seaborn
- Albumentations, timm, imagehash, grad-cam, SHAP, Flask (labeling tool)

## 1.9 Backend Architecture
PyTorch-based deep learning classifier with:
- EfficientNet-B0 backbone (pretrained ImageNet)
- Two independent classification heads (maturity + quality)
- Multi-task loss combination (weighted cross-entropy)
- Two-phase transfer learning (freeze backbone then fine-tune)
- AMP mixed precision training

## 1.10 Frontend Architecture
Streamlit web application with:
- Image upload component
- Prediction result display cards
- Grad-CAM visualization panels
- Sample gallery from test dataset
- Custom CSS styling (dark theme, Inter font)

## 1.11 Database Technology
File-based persistence using CSV annotations; no relational database. Images stored in directory structure (train/val/test splits).

## 1.12 External APIs and Third-Party Integrations
- Label Studio (for annotation, mentioned in technical summary)
- No active external API integrations in the current classifier

## 1.13 Authentication and Authorization Mechanisms
None present. The Streamlit app has no auth; intended for local/demo use.

## 1.14 Deployment Infrastructure and Configuration
- Streamlit app runs locally via `streamlit run streamlit_app/app.py`
- Training via `python main.py train`
- Checkpoint saving to `breadfruit_model/` directory
- No Dockerfile or container configuration present
- No CI/CD pipeline configuration

## 1.15 Major Dependencies
- torch, torchvision, streamlit, numpy, PIL, cv2, matplotlib, seaborn, scikit-learn, pandas, tqdm
- grad-cam, albumentations, timm, imagehash, shap
- flask (labeling tool)

## 1.16 Entry Points and Application Startup
- `python main.py train` - Training pipeline
- `python main.py evaluate --checkpoint <path>` - Evaluation
- `python main.py predict --image <path>` - Single image prediction
- `python main.py predict --folder <path>` - Batch prediction
- `streamlit run streamlit_app/app.py` - Web interface (port 8501)

## 1.17 Project Structure Summary
```
breadfruit/             # Core model code: architecture, training, evaluation
breadfruit_classifier/  # PyTorch classifier module
  main.py               # CLI entry point for train/evaluate/predict
  config.py             # Dataclass-based configuration
  model.py              # EfficientNet-B0 multi-task model definition
  datasets.py           # Dataset class, transforms, DataLoader creation
  train.py              # Training loop, validation, early stopping, checkpointing
  evaluate.py           # Test evaluation, metrics, confusion matrices
  predict.py            # Single-image and batch inference, CSV export
  gradcam.py            # Grad-CAM implementation
  annotations.csv       # 540 labeled entries (original + propagated)
breadfruit_model/       # Trained model checkpoint and training artifacts
  best_model.pth        # Best model weights (val_avg_acc = 0.9222)
  training_history.json # Per-epoch loss and accuracy values
breadfruit_pipeline/    # Dataset engineering pipeline (9 phases)
  config/               # Pipeline configuration with thresholds and parameters
  run_pipeline.py       # Main pipeline entry point
  src/                  # Phase implementations (phase1-9)
streamlit_app/          # Web interface for interactive inference
  app.py                # Streamlit web application
  requirements.txt      # Frontend dependencies
data_augmentation_pipeline.py  # Alternative augmentation script
technical_summary_for_paper.md  # Comprehensive technical summary
```
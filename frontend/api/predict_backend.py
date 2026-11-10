"""
Separate FastAPI backend for breadfruit prediction with Grad-CAM.
Uses the existing breadfruit_classifier predict module.
"""

import sys
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, "/home/jeswin/data_augmentation/breadfruit_classifier")

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_single
from config import Config
from model import BreadfruitClassifier
from PIL import Image

# Initialize config and model at startup
config = Config()
model = BreadfruitClassifier(config)
model.eval()

app = FastAPI(
    title="Breadfruit Grad-CAM API",
    description="Breadfruit maturity and quality classification with Grad-CAM interpretability",
    version="2.0.0",
)

# Add CORS middleware for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Breadfruit Grad-CAM API",
        "endpoints": [
            "/predict/",
            "/health",
            "/gradcam/info",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "BreadfruitClassifier", "gradcam": "enabled"}


@app.get("/gradcam/info")
async def gradcam_info():
    """Grad-CAM information endpoint."""
    return {
        "feature": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
        "target_layer": "features.8",
        "heatmap_size": 224,
        "tasks": ["maturity", "quality"],
        "description": "Generates class activation heatmaps explaining model predictions",
    }


@app.post("/predict/", response_model=Dict)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict maturity and quality for a single image with Grad-CAM heatmaps.
    
    Parameters:
        - file: Image file (jpg, png, jpeg)
    
    Returns:
        - maturity prediction and confidence
        - quality prediction and confidence
        - Grad-CAM heatmaps (224-element lists for both tasks)
    """
    # Read and prepare image
    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file: {str(e)}", "status": "error"}
    
    # Run prediction with Grad-CAM
    result = predict_single(model, pil_image, config, generate_gradcam=True)
    
    # Format response
    response = {
        "status": "success",
        "maturity": {
            "prediction": result["maturity"]["prediction"],
            "class_id": result["maturity"]["class_id"],
            "confidence": result["maturity"]["confidence"],
            "grad_cam": result["maturity"]["grad_cam"],  # 224-element list
        },
        "quality": {
            "prediction": result["quality"]["prediction"],
            "class_id": result["quality"]["class_id"],
            "confidence": result["quality"]["confidence"],
            "grad_cam": result["quality"]["grad_cam"],  # 224-element list
        },
        "probabilities": {
            "maturity": result["maturity"].get("probabilities", {}),
            "quality": result["quality"].get("probabilities", {}),
        },
        "image_filename": file.filename,
    }
    
    return response


@app.post("/predict/batch/", response_model=Dict)
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict for multiple images in a batch.
    
    Parameters:
        - files: List of image files
    
    Returns:
        - Results for all images with Grad-CAM heatmaps
    """
    results = []
    
    for file in files:
        contents = await file.read()
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            results.append({
                "status": "error",
                "filename": file.filename,
                "error": "Invalid image file",
            })
            continue
        
        result = predict_single(model, pil_image, config, generate_gradcam=True)
        results.append({
            "status": "success",
            "filename": file.filename,
            "maturity": {
                "prediction": result["maturity"]["prediction"],
                "confidence": result["maturity"]["confidence"],
                "grad_cam": result["maturity"]["grad_cam"],
            },
            "quality": {
                "prediction": result["quality"]["prediction"],
                "confidence": result["quality"]["confidence"],
                "grad_cam": result["quality"]["grad_cam"],
            },
        })
    
    return {
        "status": "batch_complete",
        "total_images": len(files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import sys
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "breadfruit_classifier"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Breadfruit Classification API",
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

    # Import prediction modules inside the app factory to avoid
    # module-level execution issues in serverless environments
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "streamlit_app"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "breadfruit_classifier"))

    from config import Config
    from model import BreadfruitClassifier
    from predict import predict_single
    from PIL import Image

    # Initialize config and model lazily
    config = Config()
    config.data.annotation_file = str(Path(__file__).resolve().parent.parent.parent / "breadfruit_classifier" / "annotations.csv")
    config.logging.checkpoint_dir = str(Path(__file__).resolve().parent.parent.parent / "breadfruit_model")

    model = BreadfruitClassifier(config)
    model.eval()

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Breadfruit Classification API",
            "endpoints": ["/predict/", "/health", "/gradcam/info"],
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

    @app.post("/predict/", response_model=dict)
    async def predict_image(file: UploadFile = File(...)):
        """Predict maturity and quality for a single image with Grad-CAM heatmaps."""
        contents = await file.read()
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            return {"error": f"Invalid image file: {str(e)}", "status": "error"}

        result = predict_single(model, pil_image, config, generate_gradcam=True)

        response = {
            "status": "success",
            "maturity": {
                "prediction": result["maturity"]["prediction"],
                "class_id": result["maturity"]["class_id"],
                "confidence": result["maturity"]["confidence"],
                "grad_cam": result["maturity"]["grad_cam"],
            },
            "quality": {
                "prediction": result["quality"]["prediction"],
                "class_id": result["quality"]["class_id"],
                "confidence": result["quality"]["confidence"],
                "grad_cam": result["quality"]["grad_cam"],
            },
            "probabilities": {
                "maturity": result["maturity"].get("probabilities", {}),
                "quality": result["quality"].get("probabilities", {}),
            },
        }

        return response

    @app.post("/predict/batch/", response_model=dict)
    async def predict_batch(files: list[UploadFile] = File(...)):
        """Predict for multiple images in a batch."""
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

    return app


app = create_app()
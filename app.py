import sys
import os
from pathlib import Path
from io import BytesIO

from flask import Flask

application = Flask(__name__)

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "breadfruit_classifier"))

from config import Config
from model import BreadfruitClassifier
from train import load_checkpoint
from predict import predict_single
from datasets import get_inference_transform
from gradcam import GradCAM

st.set_page_config(
    page_title="Breadfruit Classification System",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded",
    menuItems={'Get help': None, 'Report a bug': None},
)

PALETTE = {
    "bg": "#0f1a2e",
    "surface": "#162240",
    "card": "#1b2a4a",
    "accent": "#3b82f6",
    "accent_light": "#60a5fa",
    "accent_dark": "#1d4ed8",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "border": "#1e3a5f",
}

CONFIDENCE_COLORS = {
    "high": "#10b981",
    "medium": "#f59e0b",
    "low": "#ef4444",
}


def apply_global_style():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .stApp {{
            background-color: {PALETTE['bg']};
        }}
        .main-header {{
            background: linear-gradient(135deg, {PALETTE['accent_dark']} 0%, {PALETTE['accent']} 100%);
            padding: 1.25rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            box-shadow: 0 3px 16px rgba(59,130,246,0.15);
        }}
        .main-header h1 {{
            color: #ffffff;
            font-weight: 700;
            font-size: 1.5rem;
            margin: 0;
            letter-spacing: -0.02em;
        }}
        .main-header p {{
            color: rgba(255,255,255,0.8);
            font-size: 0.8rem;
            margin: 0.25rem 0 0 0;
            font-weight: 300;
        }}
        .prediction-label {{
            color: {PALETTE['text']};
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }}
        .confidence-value {{
            font-size: 1.1rem;
            font-weight: 600;
        }}
        .confidence-bar-bg {{
            background-color: {PALETTE['border']};
            border-radius: 6px;
            height: 8px;
            width: 100%;
            margin-top: 0.4rem;
        }}
        .confidence-bar-fill {{
            height: 8px;
            border-radius: 6px;
            transition: width 0.6s ease;
        }}
        .stat-box {{
            text-align: center;
            padding: 1rem;
            background: {PALETTE['surface']};
            border-radius: 8px;
            border: 1px solid {PALETTE['border']};
        }}
        .stat-value {{
            color: {PALETTE['accent_light']};
            font-size: 1.8rem;
            font-weight: 700;
        }}
        .stat-label {{
            color: {PALETTE['text_muted']};
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.2rem;
        }}
        .footer {{
            text-align: center;
            color: {PALETTE['text_muted']};
            font-size: 0.75rem;
            padding: 2rem 0 0.5rem 0;
            border-top: 1px solid {PALETTE['border']};
            margin-top: 3rem;
        }}
        div[data-testid="stFileUploader"] {{
            background: transparent;
            border: none;
        }}
        div[data-testid="stFileUploader"] > section {{
            border: 2px dashed {PALETTE['border']};
            border-radius: 10px;
            background: {PALETTE['surface']};
        }}
        div[data-testid="stFileUploader"] > section:hover {{
            border-color: {PALETTE['accent']};
        }}
        h2, h3 {{
            color: {PALETTE['text']} !important;
            font-weight: 600 !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            background-color: {PALETTE['surface']};
            border-radius: 8px;
            padding: 0.25rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 6px;
            color: {PALETTE['text_muted']};
            font-weight: 500;
            font-size: 0.85rem;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {PALETTE['accent']};
            color: #ffffff;
        }}
        .stSpinner > div {{
            border-color: {PALETTE['accent']} !important;
        }}
        .stButton > button {{
            background-color: {PALETTE['accent']};
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 0.5rem 1.5rem;
            transition: background-color 0.2s ease;
        }}
        .stButton > button:hover {{
            background-color: {PALETTE['accent_light']};
            color: #ffffff;
        }}
        @media (max-width: 768px) {{
            .main-header {{
                padding: 1rem 0.75rem;
                margin-bottom: 1rem;
            }}
            .main-header h1 {{
                font-size: 1.25rem;
            }}
            .main-header p {{
                font-size: 0.75rem;
            }}
            .prediction-label {{
                font-size: 1rem;
            }}
            .confidence-value {{
                font-size: 0.8rem;
            }}
            .confidence-bar-bg,
            .confidence-bar-fill {{
                height: 4px;
            }}
            .stat-box {{
                padding: 0.4rem;
            }}
            .stat-value {{
                font-size: 1.2rem;
            }}
            .stat-label {{
                font-size: 0.55rem;
            }}
            .footer {{
                font-size: 0.6rem;
                padding: 1rem 0 0.25rem 0;
            }}
            div[data-testid="stFileUploader"] > section {{
                padding: 0.5rem;
            }}
            .stButton > button {{
                padding: 0.4rem 1rem;
                font-size: 0.75rem;
            }}
            .stTabs [data-baseweb="tab"] {{
                font-size: 0.7rem;
                padding: 0.1rem 0.2rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    config = Config()
    config.data.annotation_file = str(ROOT / "breadfruit_classifier" / "annotations.csv")
    config.logging.checkpoint_dir = str(ROOT / "breadfruit_model")
    checkpoint_path = str(ROOT / "breadfruit_model" / "best_model.pth")
    model, _ = load_checkpoint(checkpoint_path, config=config)
    model.eval()
    return model, config


def get_confidence_level(confidence):
    if confidence >= 0.90:
        return "high", CONFIDENCE_COLORS["high"]
    elif confidence >= 0.70:
        return "medium", CONFIDENCE_COLORS["medium"]
    return "low", CONFIDENCE_COLORS["low"]


def render_prediction_card(task_name, prediction, confidence, probabilities, class_names):
    level, color = get_confidence_level(confidence)
    pct = confidence * 100
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{task_name} Classification</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="prediction-label">{prediction}</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: {color};">{pct:.1f}%</span>
            </div>
            <div class="confidence-bar-bg" style="margin-top: 0.6rem;">
                <div class="confidence-bar-fill" style="width: {pct}%; background: {color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probabilities:
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        rows = "".join(
            f"""
            <tr>
                <td style="color: {PALETTE['text']};">{name}</td>
                <td style="color: {PALETTE['text_muted']};">{prob:.4f}</td>
                <td>
                    <div style="background: {PALETTE['border']}; border-radius: 4px; height: 6px; width: 100%;">
                        <div style="width: {prob*100}%; background: {PALETTE['accent']}; height: 6px; border-radius: 4px;"></div>
                    </div>
                </td>
            </tr>
            """
            for name, prob in sorted_probs
        )
        st.markdown(
            f"""
            <div style="margin-top: 0.75rem;">
                <table class="prediction-table" style="width: 100%; border-collapse: collapse;">
                    <thead><tr><th>Class</th><th>Probability</th><th></th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


def generate_gradcam_visualization(model, config, pil_image, task):
    transform = get_inference_transform(config)
    input_tensor = transform(pil_image).unsqueeze(0)

    gradcam = GradCAM(model, target_layer=config.gradcam.target_layer)
    heatmap = gradcam.generate(input_tensor, task=task)
    gradcam.remove_hooks()

    original_np = np.array(pil_image)
    h, w = original_np.shape[:2]
    heatmap_resized = np.array(Image.fromarray(heatmap).resize((w, h), Image.BILINEAR))
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    overlayed = (0.5 * heatmap_colored + 0.5 * original_np / 255.0)
    overlayed = (overlayed / overlayed.max() * 255).astype(np.uint8)
    return original_np, heatmap_resized, overlayed


def render_gradcam_tab(model, config, pil_image):
    col1, col2 = st.columns(2)
    with col1:
        with st.spinner("Generating Grad-CAM for maturity..."):
            orig, hm_mat, overlay_mat = generate_gradcam_visualization(
                model, config, pil_image, "maturity"
            )
        st.markdown(f'<p style="color: {PALETTE["text_muted"]}; font-size: 0.8rem; font-weight: 500; margin-bottom: 0.25rem;">Maturity Activation Map</p>', unsafe_allow_html=True)
        st.image(overlay_mat, use_container_width=True)
    with col2:
        with st.spinner("Generating Grad-CAM for quality..."):
            _, hm_qual, overlay_qual = generate_gradcam_visualization(
                model, config, pil_image, "quality"
            )
        st.markdown(f'<p style="color: {PALETTE["text_muted"]}; font-size: 0.8rem; font-weight: 500; margin-bottom: 0.25rem;">Quality Activation Map</p>', unsafe_allow_html=True)
        st.image(overlay_qual, use_container_width=True)


def main():
    apply_global_style()

    if "image_upload" not in st.session_state:
        st.session_state["image_upload"] = None
    if "image_sample" not in st.session_state:
        st.session_state["image_sample"] = None
    if "results_upload" not in st.session_state:
        st.session_state["results_upload"] = None
    if "results_sample" not in st.session_state:
        st.session_state["results_sample"] = None

    st.markdown(
        '<div class="main-header">'
        '<h1>Breadfruit Classification System</h1>'
        '<p>Maturity and Quality Assessment using EfficientNet-B0</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            f'<p style="color: {PALETTE["text"]}; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.25rem;">Classification System</p>',
            unsafe_allow_html=True,
        )

        model, config = load_model()

    tab_upload, tab_sample = st.tabs(["Upload Image", "Sample Gallery"])

    with tab_upload:
        st.markdown(
            f'<p style="color: {PALETTE["text_muted"]}; font-size: 0.85rem; margin-bottom: 1rem;">Upload a breadfruit image in JPEG or PNG format.</p>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="uploader"
        )
        if uploaded is not None:
            st.session_state["image_upload"] = Image.open(uploaded).convert("RGB")
            st.session_state["results_upload"] = None

        if st.session_state["image_upload"] is not None:
            st.image(st.session_state["image_upload"], width=300)
            if st.button("Classify", type="primary", key="btn_upload", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    st.session_state["results_upload"] = predict_single(
                        model, st.session_state["image_upload"], config, return_probs=True
                    )

            if st.session_state["results_upload"] is not None:
                result = st.session_state["results_upload"]
                mat = result["maturity"]
                qual = result["quality"]

                render_prediction_card(
                    "Maturity",
                    mat["prediction"],
                    mat["confidence"],
                    mat.get("probabilities"),
                    config.model.maturity_classes,
                )

                render_prediction_card(
                    "Quality",
                    qual["prediction"],
                    qual["confidence"],
                    qual.get("probabilities"),
                    config.model.quality_classes,
                )

                with st.expander("Grad-CAM Visualization", expanded=False):
                    render_gradcam_tab(model, config, st.session_state["image_upload"])

                row = st.columns(4)
                metrics_data = [
                    ("Confidence", f"{max(mat['confidence'], qual['confidence'])*100:.1f}%"),
                    ("Maturity", mat["prediction"].replace("_", " ").title()),
                    ("Quality", qual["prediction"].replace("_", " ").title()),
                    ("Model", "EfficientNet-B0"),
                ]
                for col, (label, value) in zip(row, metrics_data):
                    col.markdown(
                        f'<div class="stat-box">'
                        f'<div class="stat-value">{value}</div>'
                        f'<div class="stat-label">{label}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    with tab_sample:
        test_dir = ROOT / "training_dataset" / "test" / "breadfruit"
        test_images = sorted(test_dir.glob("*.[jJ][pP][gG]")) + sorted(test_dir.glob("*.[pP][nN][gG]"))
        if test_images:
            filenames = ["Select an image..."] + [p.name for p in test_images]
            selected = st.selectbox(
                "Choose a test image:",
                filenames,
                index=0,
                label_visibility="collapsed",
                key="sample_selector",
            )
            if selected and selected != "Select an image...":
                st.session_state["image_sample"] = Image.open(test_dir / selected).convert("RGB")
                st.session_state["results_sample"] = None

            if st.session_state["image_sample"] is not None:
                st.image(st.session_state["image_sample"], width=300)
                if st.button("Classify", type="primary", key="btn_sample", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        st.session_state["results_sample"] = predict_single(
                            model, st.session_state["image_sample"], config, return_probs=True
                        )

                if st.session_state["results_sample"] is not None:
                    result = st.session_state["results_sample"]
                    mat = result["maturity"]
                    qual = result["quality"]

                    render_prediction_card(
                        "Maturity",
                        mat["prediction"],
                        mat["confidence"],
                        mat.get("probabilities"),
                        config.model.maturity_classes,
                    )

                    render_prediction_card(
                        "Quality",
                        qual["prediction"],
                        qual["confidence"],
                        qual.get("probabilities"),
                        config.model.quality_classes,
                    )

                    with st.expander("Grad-CAM Visualization", expanded=False):
                        render_gradcam_tab(model, config, st.session_state["image_sample"])

                    row = st.columns(4)
                    metrics_data = [
                        ("Confidence", f"{max(mat['confidence'], qual['confidence'])*100:.1f}%"),
                        ("Maturity", mat["prediction"].replace("_", " ").title()),
                        ("Quality", qual["prediction"].replace("_", " ").title()),
                        ("Model", "EfficientNet-B0"),
                    ]
                    for col, (label, value) in zip(row, metrics_data):
                        col.markdown(
                            f'<div class="stat-box">'
                            f'<div class="stat-value">{value}</div>'
                            f'<div class="stat-label">{label}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.info("No test images found in the dataset.")

    st.markdown(
        '<div class="footer">Breadfruit Classification System -- EfficientNet-B0 Multi-Task Learning</div>',
        unsafe_allow_html=True,
    )


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Breadfruit Classification System</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            .header { color: #1e3a5f; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Breadfruit Classification System</h1>
            <p>Streamlit app - use <code>streamlit run app.py</code> locally</p>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    main()
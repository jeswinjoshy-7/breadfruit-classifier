# DELIVERABLE 7: GUI and User Interface Design Documentation

## 7.1 Main Screens and Pages

| Screen ID | Title | Purpose | Intended User | Main UI Components | User Actions | Navigation Behaviour |
|---|---|---|---|---|---|---|
| SC-01 | Upload Image | Upload a breadfruit image for classification | Operator (web user) | File uploader (JPG/PNG), Classify button, image preview | Upload image → Click "Classify" → View results | Returns to same screen after classification; can upload new image |
| SC-02 | Sample Gallery | Select from test dataset images for classification | Operator (web user) | Image thumbnail grid, dropdown selector, Classify button | Select image → Click "Classify" → View results | Returns to gallery after classification; can select different image |
| SC-03 | Prediction Results | Display maturity and quality classification results | Operator (web user) | Two prediction cards (maturity + quality), confidence bars, probability table, Grad-CAM expander, stat boxes (Confidence, Maturity, Quality, Model) | View results → Expand Grad-CAM → Review probabilities | Can navigate back to upload/gallery; can upload new image |
| SC-04 | Application Footer | Display app credits and version | All users | Text footer link | None (display only) | Always visible at bottom of page |

## 7.2 Navigation Structure

```
┌──────────────────────────────────────────────────────────────┐
│  Streamlit Sidebar                                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Classification System (text, color: #e2e8f0)         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌─────────────────────┐  ┌─────────────────────────┐ │
│  │   Upload Image      │  │   Sample Gallery        │ │
│  │   (tab)             │  │   (tab)                 │ │
│  └─────────────────────┘  └─────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Breadfruit Classification System -- EfficientNet-B0 │   │
│  │  -- EfficientNet-B0 Multi-Term Learning              │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

**Tab layout**: Two tabs (`Upload Image` and `Sample Gallery`) at the top level; persistent sidebar with "Classification System" label; footer at bottom.

## 7.3 Layout and Component Hierarchy

### 7.3.1 Upload Image Tab Layout

```
Page Layout:
│ ──────────────────────────────────────────────│
│  MAIN HEADER: "Breadfruit Classification System"│
│  (accent blue gradient background)             │
│  ──────────────────────────────────────────────│
│  Upload Section:                               │
│  • File uploader (hidden label, dashed border) │
│  • Display uploaded image (width=300)          │
│  • "Classify" primary button                   │
│  ──────────────────────────────────────────────│
│  Results Section (conditionally rendered):     │
│  • Maturity prediction card                    │
│  • Quality prediction card                     │
│  • Confidence metrics row (4 stat boxes)       │
│  • "Grad-CAM Visualization" expander           │
│  ──────────────────────────────────────────────│
│  FOOTER: "Breadfruit Classification System --   │
│  EfficientNet-B0 Multi-Term Learning"          │
│  ──────────────────────────────────────────────│
```

### 7.3.2 Sample Gallery Tab Layout

```
Page Layout:
│ ──────────────────────────────────────────────│
│  MAIN HEADER: "Breadfruit Classification System"│
│  ──────────────────────────────────────────────│
│  Instruction: "Choose a test image:"           │
│  • Dropdown selector with test images          │
│  • "Select an image..." default option         │
│  ──────────────────────────────────────────────│
│  Image Display (width=300, conditionally)      │
│  • Selected image shown if                     │
│  ──────────────────────────────────────────────│
│  Results Section (conditionally rendered, same  │
│  structure as Upload tab results)              │
│  ──────────────────────────────────────────────│
│  FOOTER (same as Upload tab)                   │
│  ──────────────────────────────────────────────│
```

## 7.4 Forms and Input Fields

| Form ID | Field Name | Type | Description | Validation | Related FR |
|---|---|---|---|---|---|
| F-01 | Image file | File upload | JPG, JPEG, or PNG format breadfruit image | Accepted types: ['jpg', 'jpeg', 'png']; error shown if other type selected | FR-01 |
| F-02 | None (CLI) | N/A | Command-line arguments (--image, --folder, --checkpoint, --visualize, --probabilities) | Validated in argparse; missing required args cause error | FR-09 |

## 7.5 Buttons and Interactive Controls

| Button ID | Label | Type | Purpose | Enabled State | On Click |
|---|---|---|---|---|---|
| B-01 | "Classify" | primary | Trigger image classification | Enabled after image upload | Calls predict_single() → displays results |
| B-02 | "Select an image..." | default | (dropdown default) | Always shown in gallery dropdown | Navigates to selection (no action) |
| B-03 | "Classify" (gallery) | primary | Trigger classification of selected test image | Enabled after dropdown selection != default | Calls predict_single() → displays results |
| B-04 | "Grad-CAM Visualization" | default | Expand/collapse Grad-CAM heatmaps | Always enabled | Toggles Grad-CAM overlay display |
| B-05 | (footer) | link | Credits display | Always visible | No interaction |

## 7.6 Tables, Charts, and Visual Elements

| Element ID | Type | Description | Data Shown | Related Screen |
|---|---|---|---|---|
| C-01 | Prediction Card | Styled card component | Maturity/quality prediction + confidence percentage | SC-03 (both tabs) |
| C-02 | Confidence Bar | Horizontal progress bar | Confidence percentage width; color-coded (high/medium/low) | SC-03 |
| C-03 | Probability Table | HTML table | Per-class probabilities for each task (3 classes × 2 tasks) | SC-03 (if --probabilities or shown) |
| C-04 | Grad-CAM Overlay | Image display | Heatmap overlaid on original breadfruit image | SC-03 (expander) |
| C-05 | Stat Box | Colored box | Confidence %, Maturity class, Quality class, Model name | SC-01 / SC-02 (metrics row) |
| C-06 | Confidence Color Coding | Color rules | high ≥ 0.90 (green), 0.70-0.90 (amber), < 0.70 (red) | SC-03 |
| C-07 | Test Image Thumbnails | Image grid | Miniature versions of test dataset images | SC-02 (dropdown options) |

## 7.7 Validation and Error Messages

| Scenario | Error Message | Location |
|---|---|---|
| Unsupported file format | "Unsupported file format. Please upload JPG, JPEG, or PNG." | Streamlit file uploader |
| Failed image load (corrupt file) | "Failed to load image. The file may be corrupt." | predict_single() → logged; Streamlit error element |
| No checkpoint found | "Error: No checkpoint specified and best_model.pth not found." | cmd_predict / cmd_train / cmd_evaluate |
| No test images | "No test images found in the dataset." | Streamlit app tab "Sample Gallery" |
| Dropdown no selection | "Select an image..." shown; no prediction until image selected | Sample Gallery dropdown |

## 7.8 Loading States

- `st.spinner("Analyzing image...")` displayed during prediction single image classification ( Streamlit app:352-355, 416-419)
- Predictions are computed synchronously within the spinner context; no async delays beyond model forward pass

## 7.9 Empty States

- **No image uploaded**: File uploader displays with placeholder; "Classify" button disabled until image selected
- **No test images**: Streamlit info message: "No test images found in the dataset." (app.py:461)
- **Dropdown default**: "Select an image..." option in gallery; no image displayed until user makes selection

## 7.10 Success States

- **Successful classification**: Prediction cards displayed with maturity class, quality grade, confidence percentage, and confidence bar color-coded (green for high, amber for medium, red for low)
- **Successful Grad-CAM**: Three-panel figure displayed (original image / heatmap / overlay) in expander
- **Successful batch prediction**: CSV file saved with all results; console output for each image processed

## 7.11 Responsive Design Behaviour

The Streamlit app uses `layout="wide"` in `st.set_page_config()`. The layout adapts as follows:

- **Wide mode (≥ 600px)**: Two-column layout possible; Grad-CAM displayed alongside prediction cards in render_gradcam_tab()
- **Narrow mode (< 600px)**: Single-column layout; elements stacked vertically; Grad-CAM expander may take significant screen space
- The CSS is primarily fixed-width; true responsive behavior is limited by Streamlit's column grid system

## 7.12 Accessibility Considerations

- Custom CSS uses color palette with sufficient contrast: `#e2e8f0` text on `#0f1a2e` background (contrast ratio ≥ 4.5:1 for large text)
- `st.markdown` with `unsafe_allow_html=True` for full CSS control (including focus states, font sizes)
- Descriptive alt text not explicitly set on images in the code; visual content relies on color and position
- Keyboard navigation available through Streamlit's tab and button tab ordering
- Color-dependent information (confidence bar colors) not conveyed via alternative text; users relying on screen readers may miss confidence level meaning

## 7.13 Frontend-to-Backend Communication

- **Streamlit's reactive model**: User interactions trigger re-runs; state managed via `st.session_state`
- **No REST API**: Frontend and backend run in same Python process; communication via function calls within same process
- **Data flow**: 
  - User uploads image → `st.session_state["image_upload"]` stores PIL Image
  - User clicks "Classify" → `predict_single(model, image, config, return_probs=True)` called
  - Result stored in `st.session_state["results_upload"]` 
  - Results displayed via `render_prediction_card()` with HTML/CSS
- **Model loading**: `@st.cache_resource` decorated `load_model()` function loads model + config once; reused across interactions
- **Gradient CAM**: Hooks registered on model backbone; temporary; removed after generation (`gradcam.remove_hooks()`)
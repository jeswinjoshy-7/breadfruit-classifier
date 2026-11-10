# DELIVERABLE 10: Unit, Integration, System, and Non-Functional Testing Documentation

## 9.1 Unit Testing

### 9.1.1 Existing Test Evidence
The project does **not** appear to have a formal unit test suite (no pytest/unittest directory, no test_*.py files found). No test execution commands, no automated test results.

### 9.1.2 Identifiable Testable Units

| Test ID | Test Name | Objective | Function/Component Tested | Input Data | Expected Output | Relevant File |
|---|---|---|---|---|---|---|
| UT-01 | Config Initialization | Verify Config dataclasses initialize correctly | Config from config.py | Default Config() instantiation | All nested configs populated with defaults; device attribute set based on CUDA availability | config.py |
| UT-02 | Model Forward Pass | Verify BreadfruitClassifier.forward() produces correct output shape | BreadfruitClassifier.forward() | Input tensor (1, 3, 224, 224) | Dict with "maturity_logits" (1, 3) and "quality_logits" (1, 3); has_multi_task check | model.py |
| UT-03 | MultiTaskHead Forward | Verify MultiTaskHead classifier output | MultiTaskHead.forward() | Input tensor (1, 1280) after GAP | Tensor of shape (1, 3) (3 classes) | model.py:15-26 |
| UT-04 | Loss Computation | Verify MultiTaskLoss computes weighted sum | MultiTaskLoss.forward() | predictions dict, targets dict | total_loss = maturity_weight × maturity_loss + quality_weight × quality_loss; loss_dict with 3 keys | model.py:98-130 |
| UT-05 | Inference Transform | Verify get_inference_transform produces correct compose | get_inference_transform(config) | Config object | transforms.Compose([Resize(224), CenterCrop(224), ToTensor(), Normalize(mean, std)]) | datasets.py:234-241 |
| UT-06 | Prediction Single | Verify predict_single returns correct dict structure | predict_single() | Model, image, config, return_probs=True | Dict with "maturity" and "quality" keys; each has "prediction", "class_id", "confidence"; optional "probabilities" | predict.py:42-99 |
| UT-07 | Grad-CAM Hook Registration | Verify GradCAM._register_hooks() registers successfully | GradCAM._register_hooks() | Model with features.8 layer | Two handles appended to self.handles; forward/backward hooks active | gradcam.py:51-65 |

### 9.1.3 Recommended Additional Unit Tests

| Test ID | Test Name | Objective | Reason for Recommendation |
|---|---|---|---|
| UT-80 | Dataset sample loading with annotations | Verify BreadfruitDataset._load_samples correctly matches filenames to labels | Critical for training; currently no test coverage; label mapping must be correct |
| UT-81 | Transform pipeline correctness | Verify train/val/test transforms have different compositions | Ensures augmentation only in train; valuable for reproducibility |
| UT-82 | Parameter counting | Verify count_parameters() returns expected values | 5.3M backbone + 7,686 heads; validates model initialization |
| UT-83 | Head weight initialization | Verify Xavier uniform initialization on Linear layers | Important for training stability; _init_heads() method |
| UT-84 | Edge case: single class mode | Verify BreadfruitDataset falls back correctly when no annotation file | Important robustness; annotation CSV may be missing |
| UT-85 | Device placement | Verify model and data moved to correct device (CPU/CUDA) | Critical for GPU training/inference; config.device logic |

### 9.1.4 Unit Test Execution Status
- **Current status**: No unit tests exist in the project
- **Execution command** (if authorized): `python -m pytest tests/` or `python -m unittest discover`
- **Expected outcome**: Tests would verify the above testable units

---

## 9.2 Integration Testing

### 9.2.1 Identified Integration Test Scenarios

| Test ID | Test Scenario | Preconditions | Test Data | Expected Results | Existing Evidence |
|---|---|---|---|---|---|
| INT-01 | Frontend-backend communication (Streamlit → prediction) | Model loaded; image uploaded via Streamlit | Any valid breadfruit image JPG/PNG | Prediction result displayed in UI; confidence bars shown; Grad-CAM expander available | Verified: Streamlit app works end-to-end |
| INT-02 | API-service integration (CLI predict) | Trained checkpoint available; --image arg specified | `python main.py predict --image test.jpg` | Predicted maturity and quality printed to console; confidence scores shown | Verified: main.py cmd_predict execution |
| INT-03 | Service-database integration (dataset loading) | Annotation CSV exists; images in data_root/train/breadfruit/ | `python main.py train` | Train DataLoader yields batches with (image, {"maturity": idx, "quality": idx}); use_annotations=True | Verified: training executed successfully |
| INT-04 | Authentication integration | None (no auth implemented) | N/A | N/A | Not applicable; noted as limitation |
| INT-05 | Third-party API integration | None (no external APIs) | N/A | N/A | Not applicable |
| INT-06 | Pipeline phase integration (9 phases sequential) | Raw images in source dirs; config.yaml enabled | `python run_pipeline.py` | All 9 phases execute (or skip based on flags); organized dataset output; reports generated | Verified: pipeline runs; technical_summary documents output |
| INT-07 | Grad-CAM + prediction integration | Model loaded; image classified; Grad-CAM generated | `python main.py predict --image.jpg --visualize` | Prediction cards displayed; Grad-CAM 3-panel figure saved to plots/gradcam/ | Verified: Streamlit shows Grad-CAM expander |
| INT-08 | Checkpoint load + evaluation integration | best_model.pth exists; test data available | `python main.py evaluate --checkpoint best_model.pth` | Evaluation metrics printed; confusion matrices plotted; results JSON saved | Verified: evaluate command executed |

---

## 9.3 System Testing

### 9.3.1 System Test Scenarios

| Test ID | Scenario | Coverage |
|---|---|---|
| SYS-01 | End-to-end: train → evaluate → predict | Complete workflow: train model from scratch → evaluate on test set → classify new image |
| SYS-02 | End-to-end: pipeline → train → predict | Complete workflow: run 9-phase dataset pipeline → train on organized data → classify image |
| SYS-03 | Web UI: upload → classify → Grad-CAM | Streamlit: upload image → click Classify → view results → expand Grad-CAM expander |
| SYS-04 | Web UI: sample gallery → classify → results | Streamlit: select test image from dropdown → classify → view prediction cards |
| SYS-05 | Batch prediction folder → CSV export | predict_folder: specify folder with multiple images → save CSV with all results |
| SYS-06 | Training with interrupt and resume | Start training; Ctrl+C interrupt; restart; verify checkpoint can resume |

---

## 9.4 Non-Functional Testing

### 9.4.1 Assessment of Relevant Test Types

| Test Type | Relevance | Status | Evidence |
|---|---|---|---|
| Performance testing | Medium | Not executed | No performance benchmarks recorded; latency not measured |
| Load testing | Low | Not applicable | Single-user local application; no concurrent user load |
| Stress testing | Low | Not applicable | No defined stress thresholds or failure points |
| Security testing | Low | Not executed | No auth; limited scope; noted as design limitation |
| Usability testing | Medium | Not executed | Streamlit interface usable; no formal user testing |
| Compatibility testing | Low | Partially | Tested on Linux with CUDA 12.6; no Windows/Mac testing |
| Reliability testing | Medium | Not executed | Deterministic seed 42; no repeated runs documented |
| Accessibility testing | Low | Not executed | Color contrast designed; no screen reader validation |

---

## 9.5 Test Case Table

| Test ID | Test Type | Test Scenario | Preconditions | Test Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|---|---|
| UT-01 | Unit | Config initialization | Config() instantiation | Default Config object | All nested configs populated; device set | Not executed | Not Executed |
| UT-02 | Unit | Model forward pass | Model eval mode; (1,3,224,224) tensor | Random tensor | Dict with maturity_logits, quality_logits | Not executed | Not Executed |
| INT-01 | Integration | Streamlit prediction | Model loaded; image uploaded | Uploaded JPG image | Results displayed in UI | Not executed | Not Executed |
| SYS-01 | System | Train → Evaluate → Predict | Config; raw dataset; checkpoint dir | `python main.py train` then evaluate | Trained model; metrics; predict works | Not executed | Not Executed |
| NFR-01 | Non-functional | Inference latency | Model on CPU | Single image | Latency < 1s (expected) | Not executed | Not Executed |

### 9.5.1 Notes
- "Not Executed" means tests have not been run in this project
- The code has testable units but no automated test suite appears to exist
- Actual results column shows "Not executed" since tests have not been run
- Status column uses "Not Executed" for consistency

---

## 9.6 Coverage and Quality Gaps

### 9.6.1 Existing Automated Tests
- **None**. No pytest/unittest files found in the project.
- No test execution commands (pytest, python -m unittest) observed in the codebase or README.

### 9.6.2 Missing Tests (Critical Gaps)

1. **Dataset sample loading**: No test verifies that BreadfruitDataset correctly matches filenames to annotation labels. This is critical since training depends on correct label lookup.

2. **Transform correctness**: No test verifies that train transforms include augmentation and test transforms do not.

3. **Multi-task head output**: No test verifies the output shape of BreadfruitClassifier.forward() matches expectations.

4. **Loss function**: No test verifies MultiTaskLoss weighted combination correctness.

5. **Edge cases**: No tests for single-class mode (no annotation file), empty dataset, or malformed inputs.

6. **Integration**: No end-to-end tests connecting data loading → training → evaluation → prediction.

7. **Pipeline phase tests**: No tests for any of the 9 pipeline phases.

### 9.6.3 Important Untested Modules

- `breadfruit_classifier/datasets.py`: _load_samples() critical path not tested
- `breadfruit_pipeline/src/`: All 9 phases have no unit tests
- `breadfruit_classifier/predict.py`: predict_single() and predict_folder() not tested
- `breadfruit_classifier/gradcam.py`: GradCAM.generate() not tested

### 9.6.4 Error-Handling Gaps

- Corrupt image files: try/except in predict_folder() but not comprehensively
- Missing checkpoint: error messages exist but no test verifies correct behavior
- Malformed annotation CSV: falls back to single-class mode; no test verifies this path

### 9.6.5 Potential Integration Failures

- Annotation CSV with wrong column names → KeyError (not handled gracefully)
- Mismatched image filenames between CSV and directory → some samples lose labels
- Device mismatch (CPU model vs CUDA evaluation) → potential errors

### 9.6.6 Recommended Additional Tests (if testing is authorized)

1. **Pytest suite** with test files for each module:
   - `tests/test_datasets.py`: sample loading, transform correctness
   - `tests/test_model.py`: forward pass, parameter counting, loss computation
   - `tests/test_predict.py`: predict_single, predict_folder
   - `tests/test_gradcam.py`: heatmap generation
   - `tests/test_pipeline.py`: individual phase execution

2. **Run existing evaluation**: Execute `python main.py evaluate --checkpoint breadfruit_model/best_model.pth` to verify test metrics match reported values (maturity 1.0000, quality 0.9487, avg 0.9744).

3. **Test with sample images**: Use the Streamlit web interface to classify sample images and verify results.

4. **Test pipeline phases individually**: Run `python run_pipeline.py` with specific phase configurations to verify each phase.
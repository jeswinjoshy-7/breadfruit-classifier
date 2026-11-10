#!/usr/bin/env python3
"""Breadfruit Dataset Engineering Pipeline - Main Entry Point."""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import time
import logging

from breadfruit_pipeline.core.config import get_config
from breadfruit_pipeline.core.logger import setup_logging
from breadfruit_pipeline.phases.phase1_inspection import DatasetInspector
from breadfruit_pipeline.phases.phase2_duplicates import DuplicateDetector
from breadfruit_pipeline.phases.phase3_quality import QualityAnalyzer
from breadfruit_pipeline.phases.phase4_object_validation import ObjectValidator
from breadfruit_pipeline.phases.phase5_standardization import ImageStandardizer
from breadfruit_pipeline.phases.phase6_organization import DatasetOrganizer
from breadfruit_pipeline.phases.phase7_analysis import DatasetAnalyzer
from breadfruit_pipeline.phases.phase8_reports import ReportGenerator
from breadfruit_pipeline.phases.phase9_validation import FinalValidator

logger = logging.getLogger("pipeline")


def run_pipeline(config_path: str = None):
    start_time = time.time()
    
    config = get_config(config_path)
    base_dir = Path(config.output.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = base_dir / config.output.structure.logs
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = str(log_dir / "pipeline.log")
    setup_logging(log_file=log_file, level=getattr(logging, config.logging.level.upper(), logging.INFO))
    
    logger.info("=" * 60)
    logger.info("BREADFRUIT DATASET ENGINEERING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Output base: {base_dir}")
    
    phase_results = {}
    
    for dataset_key, dataset_config in vars(config.datasets).items():
        setattr(dataset_config, 'key', dataset_key)

    # --- Phase 1: Inspection ---
    if config.phase1_inspection.enabled:
        logger.info("--- Phase 1: Dataset Inspection ---")
        inspector = DatasetInspector(config_path)
        reports = {}
        for dataset_key, dataset_config in vars(config.datasets).items():
            report = inspector.inspect_dataset(dataset_config.path, dataset_config.name)
            reports_dir = base_dir / config.output.structure.reports
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"phase1_inspection_{dataset_key}"
            inspector.save_report(report, str(report_path))
            reports[dataset_key] = report
        phase_results["inspection"] = reports
        logger.info(f"Phase 1 complete: inspected {sum(r.valid_images for r in reports.values())} valid images")
    else:
        logger.info("Phase 1: SKIPPED")

    # --- Phase 2: Duplicates ---
    if config.phase2_duplicates.enabled and "inspection" in phase_results:
        logger.info("--- Phase 2: Duplicate Detection ---")
        detector = DuplicateDetector(config_path)
        reject_dir = base_dir / config.output.structure.rejected.root
        dup_reports = {}
        for dataset_key, report in phase_results["inspection"].items():
            valid_paths = [r.path for r in report.images if not r.is_corrupted and not r.is_zero_byte]
            if valid_paths:
                dup_report = detector.detect_duplicates(valid_paths, dataset_key, str(reject_dir))
                reports_dir = base_dir / config.output.structure.reports
                reports_dir.mkdir(parents=True, exist_ok=True)
                detector.save_report(dup_report, str(reports_dir / f"phase2_duplicates_{dataset_key}"))
                dup_reports[dataset_key] = dup_report
        phase_results["duplicates"] = dup_reports
        flagged = sum(r.images_flagged for r in dup_reports.values())
        logger.info(f"Phase 2 complete: {flagged} duplicate images flagged")
    else:
        logger.info("Phase 2: SKIPPED")

    # --- Phase 3: Quality ---
    if config.phase3_quality.enabled and "inspection" in phase_results:
        logger.info("--- Phase 3: Quality Analysis ---")
        analyzer = QualityAnalyzer(config_path)
        reject_dir = base_dir / config.output.structure.rejected.root
        qual_reports = {}
        for dataset_key, report in phase_results["inspection"].items():
            valid_paths = [r.path for r in report.images if not r.is_corrupted and not r.is_zero_byte]
            if valid_paths:
                qual_report = analyzer.analyze_dataset(valid_paths, dataset_key, str(reject_dir))
                reports_dir = base_dir / config.output.structure.reports
                reports_dir.mkdir(parents=True, exist_ok=True)
                analyzer.save_report(qual_report, str(reports_dir / f"phase3_quality_{dataset_key}"))
                qual_reports[dataset_key] = qual_report
        phase_results["quality"] = qual_reports
        flagged = sum(r.flagged_images for r in qual_reports.values()) if qual_reports else 0
        logger.info(f"Phase 3 complete: {flagged} images flagged for quality")
    else:
        logger.info("Phase 3: SKIPPED")

    # --- Phase 4: Object Validation ---
    if config.phase4_object_validation.enabled and "inspection" in phase_results:
        logger.info("--- Phase 4: Object Validation ---")
        validator = ObjectValidator(config_path)
        reject_dir = base_dir / config.output.structure.rejected.root
        obj_reports = {}
        for dataset_key, report in phase_results["inspection"].items():
            valid_paths = [r.path for r in report.images if not r.is_corrupted and not r.is_zero_byte]
            if valid_paths:
                obj_report = validator.validate_dataset(valid_paths, dataset_key, str(reject_dir))
                reports_dir = base_dir / config.output.structure.reports
                reports_dir.mkdir(parents=True, exist_ok=True)
                validator.save_report(obj_report, str(reports_dir / f"phase4_object_validation_{dataset_key}"))
                obj_reports[dataset_key] = obj_report
        phase_results["object_validation"] = obj_reports
        detected = sum(r.fruit_detected_count for r in obj_reports.values()) if obj_reports else 0
        logger.info(f"Phase 4 complete: fruit detected in {detected} images")
    else:
        logger.info("Phase 4: SKIPPED")

    # --- Phase 5: Standardization ---
    if config.phase5_standardization.enabled and "inspection" in phase_results:
        logger.info("--- Phase 5: Standardization ---")
        standardizer = ImageStandardizer(config_path)
        std_reports = {}
        for dataset_key, report in phase_results["inspection"].items():
            valid_paths = [r.path for r in report.images if not r.is_corrupted and not r.is_zero_byte]
            if valid_paths:
                std_dir = base_dir / "standardized"
                std_report = standardizer.standardize_dataset(valid_paths, dataset_key, str(std_dir))
                reports_dir = base_dir / config.output.structure.reports
                reports_dir.mkdir(parents=True, exist_ok=True)
                standardizer.save_report(std_report, str(reports_dir / f"phase5_standardization_{dataset_key}"))
                std_reports[dataset_key] = std_report
        phase_results["standardization"] = std_reports
        success = sum(r.successful for r in std_reports.values()) if std_reports else 0
        logger.info(f"Phase 5 complete: {success} images standardized to 224x224")
    else:
        logger.info("Phase 5: SKIPPED")

    # --- Phase 6: Organization ---
    if config.phase6_organization.enabled and "standardization" in phase_results:
        logger.info("--- Phase 6: Organization ---")
        organizer = DatasetOrganizer(config_path)
        dataset_info = {}
        for ds_key in vars(config.datasets):
            ds_cfg = getattr(config.datasets, ds_key)
            dataset_info[ds_key] = {"name": ds_cfg.name}
        org_results = organizer.organize_datasets(dataset_info)
        reports_dir = base_dir / config.output.structure.reports
        reports_dir.mkdir(parents=True, exist_ok=True)
        organizer.generate_structure_report(str(reports_dir / "dataset_structure.json"))
        phase_results["organization"] = org_results
        total_clean = sum(r.clean_images for r in org_results.values())
        logger.info(f"Phase 6 complete: {total_clean} images organized into clean directories")
    else:
        logger.info("Phase 6: SKIPPED")

    # --- Phase 7: Analysis ---
    if config.phase7_analysis.enabled and "inspection" in phase_results:
        logger.info("--- Phase 7: Analysis ---")
        analyzer = DatasetAnalyzer(config_path)
        analysis_reports = {}
        for dataset_key, report in phase_results["inspection"].items():
            valid_paths = [r.path for r in report.images if not r.is_corrupted and not r.is_zero_byte]
            if valid_paths:
                quality_data = None
                if "quality" in phase_results and dataset_key in phase_results["quality"]:
                    quality_data = {dataset_key: phase_results["quality"][dataset_key].results}
                analysis = analyzer.analyze_dataset(valid_paths, dataset_key, quality_data)
                reports_dir = base_dir / config.output.structure.reports
                reports_dir.mkdir(parents=True, exist_ok=True)
                analyzer.save_analysis(analysis, str(reports_dir / f"phase7_analysis_{dataset_key}"))
                analysis_reports[dataset_key] = analysis
        phase_results["analysis"] = analysis_reports
        logger.info(f"Phase 7 complete: analysis generated for {len(analysis_reports)} datasets")
    else:
        logger.info("Phase 7: SKIPPED")

    # --- Phase 8: Reports ---
    if config.phase8_reports.enabled:
        logger.info("--- Phase 8: Report Generation ---")
        report_gen = ReportGenerator(config_path)
        report_paths = report_gen.generate_all_reports(phase_results)
        phase_results["reports"] = report_paths
        logger.info(f"Phase 8 complete: {len(report_paths)} reports generated")
    else:
        logger.info("Phase 8: SKIPPED")

    # --- Phase 9: Validation ---
    if config.phase9_validation.enabled:
        logger.info("--- Phase 9: Final Validation ---")
        validator = FinalValidator(config_path)
        validation_reports = validator.validate_all_datasets(str(base_dir))
        reports_dir = base_dir / config.output.structure.reports
        reports_dir.mkdir(parents=True, exist_ok=True)
        for name, v_report in validation_reports.items():
            validator.save_report(v_report, str(reports_dir / f"phase9_validation_{name}"))
        phase_results["validation"] = validation_reports
        all_passed = all(r.overall_passed for r in validation_reports.values())
        logger.info(f"Phase 9 complete: overall {'PASSED' if all_passed else 'FAILED'}")
    else:
        logger.info("Phase 9: SKIPPED")

    # --- Summary ---
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.2f}s")
    logger.info(f"Output directory: {base_dir}")
    logger.info("=" * 60)

    return phase_results


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    results = run_pipeline(config_path)
    print("\nPipeline execution complete. Check the log file and reports directory for details.")

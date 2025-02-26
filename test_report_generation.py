#!/usr/bin/env python
"""Debug script for PDF report generation."""
import json
import logging
import sys
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    # Import the report generator
    from src.analysis.report_generator import PDFReportGenerator

    # Check if analysis_report.json exists
    analysis_path = Path("models/analysis/analysis_report.json")
    if not analysis_path.exists():
        logger.error(f"Analysis report not found at {analysis_path}")
        sys.exit(1)

    # Load the analysis results
    try:
        with open(analysis_path, "r") as f:
            analysis_results = json.load(f)
        logger.info(f"Successfully loaded analysis report from {analysis_path}")

        # Print the keys in the analysis results to help debug
        logger.debug(f"Analysis result keys: {list(analysis_results.keys())}")

        # If financial_impact exists, print its keys
        if "financial_impact" in analysis_results:
            logger.debug(
                f"Financial impact keys: {list(analysis_results['financial_impact'].keys())}"
            )

            # If profit_by_threshold exists, examine its structure
            if "profit_by_threshold" in analysis_results["financial_impact"]:
                logger.debug("profit_by_threshold structure:")
                for threshold, values in analysis_results["financial_impact"][
                    "profit_by_threshold"
                ].items():
                    logger.debug(
                        f"  Threshold {threshold}: {type(threshold)}, Values: {values}"
                    )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        sys.exit(1)

    # Create the report generator
    try:
        generator = PDFReportGenerator(
            analysis_results=analysis_results, save_path=Path("models/analysis")
        )
        logger.info("Successfully created PDFReportGenerator instance")

        # Generate the report with additional debugging
        try:
            pdf_path = generator.generate_report(filename="debug_report.pdf")
            logger.info(f"Successfully generated report at {pdf_path}")
        except Exception as e:
            logger.exception(f"Error during report generation: {e}")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Error creating PDFReportGenerator: {e}")
        sys.exit(1)

except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    sys.exit(1)

#!/usr/bin/env python
"""Command-line interface for generating PDF reports from model analysis results."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.analysis.report_generator import PDFReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def main():
    """Run the report generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate a PDF report from model analysis results"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the analysis_report.json file or directory containing it",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="model_analysis_report.pdf",
        help="Output PDF filename (default: model_analysis_report.pdf)",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default="Model Analysis Report",
        help="Report title (default: 'Model Analysis Report')",
    )
    parser.add_argument(
        "--subtitle",
        "-s",
        type=str,
        default="Comprehensive Evaluation and Interpretation",
        help="Report subtitle (default: 'Comprehensive Evaluation and Interpretation')",
    )

    args = parser.parse_args()

    # Handle input path
    input_path = Path(args.input)
    if input_path.is_dir():
        input_path = input_path / "analysis_report.json"

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    # Load analysis results
    try:
        with open(input_path, "r") as f:
            analysis_results = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {input_path}: {e}")

        # Try to fix the JSON file
        try:
            logger.info("Attempting to fix JSON serialization issues...")
            with open(input_path, "r") as f:
                content = f.read()

            # Try to load and re-save with the custom encoder
            with open(input_path, "r") as f:
                raw_data = json.load(
                    f, object_hook=lambda d: {k: v for k, v in d.items()}
                )

            with open(input_path, "w") as f:
                json.dump(raw_data, f, indent=2, cls=NumpyEncoder)

            # Try loading again
            with open(input_path, "r") as f:
                analysis_results = json.load(f)

            logger.info("Successfully fixed JSON serialization issues")
        except Exception as fix_error:
            logger.error(f"Failed to fix JSON file: {fix_error}")
            return 1

    # Determine save path (same directory as input file)
    save_path = input_path.parent

    # Create PDF report generator
    report_generator = PDFReportGenerator(
        analysis_results=analysis_results, save_path=save_path
    )

    # Generate PDF report
    try:
        pdf_path = report_generator.generate_report(filename=args.output)
        logger.info(f"PDF report generated successfully: {pdf_path}")
        return 0
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

"""PDF report generator for model analysis results using ReportLab."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generates professional PDF reports for model analysis results."""

    def __init__(self, analysis_results: Dict, save_path: Path):
        """Initialize the PDF report generator.

        Args:
            analysis_results: Dictionary containing all analysis results
            save_path: Path to save the generated PDF report
        """
        self.analysis_results = analysis_results
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Initialize styles
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            name="TitleStyle",
            parent=self.styles["Heading1"],
            alignment=TA_CENTER,
            spaceAfter=12,
        )
        self.heading_style = ParagraphStyle(
            name="HeadingStyle",
            parent=self.styles["Heading2"],
            spaceAfter=6,
        )
        self.subheading_style = ParagraphStyle(
            name="SubheadingStyle",
            parent=self.styles["Heading3"],
            spaceAfter=6,
        )
        self.normal_style = self.styles["Normal"]
        self.normal_style.alignment = TA_LEFT

        # Initialize document elements
        self.elements = []

    def _add_title_page(self, title: str, subtitle: Optional[str] = None) -> None:
        """Add a title page to the report.

        Args:
            title: Main report title
            subtitle: Optional subtitle
        """
        self.elements.append(Spacer(1, 2 * 72))  # 2-inch spacer
        self.elements.append(Paragraph(title, self.title_style))

        if subtitle:
            self.elements.append(Spacer(1, 0.5 * 72))  # 0.5-inch spacer
            self.elements.append(Paragraph(subtitle, self.styles["Heading2"]))

        # Add date
        date_str = datetime.now().strftime("%B %d, %Y")
        self.elements.append(Spacer(1, 1 * 72))  # 1-inch spacer
        self.elements.append(Paragraph(f"Generated on: {date_str}", self.normal_style))

        self.elements.append(PageBreak())

    def _add_section_header(self, title: str) -> None:
        """Add a section header to the report.

        Args:
            title: Section title
        """
        self.elements.append(Spacer(1, 0.25 * 72))  # 0.25-inch spacer
        self.elements.append(Paragraph(title, self.heading_style))
        self.elements.append(Spacer(1, 0.1 * 72))  # 0.1-inch spacer

    def _add_subsection_header(self, title: str) -> None:
        """Add a subsection header to the report.

        Args:
            title: Subsection title
        """
        self.elements.append(Paragraph(title, self.subheading_style))

    def _add_paragraph(self, text: str) -> None:
        """Add a paragraph to the report.

        Args:
            text: Paragraph text
        """
        self.elements.append(Paragraph(text, self.normal_style))
        self.elements.append(Spacer(1, 0.1 * 72))  # 0.1-inch spacer

    def _add_image(
        self, image_path: Path, width: int = 500, height: Optional[int] = None
    ) -> None:
        """Add an image to the report.

        Args:
            image_path: Path to the image file
            width: Image width in points
            height: Optional image height in points
        """
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return

        img = Image(str(image_path), width=width, height=height)
        self.elements.append(img)
        self.elements.append(Spacer(1, 0.2 * 72))  # 0.2-inch spacer

    def _add_table(
        self,
        data: List[List[Union[str, int, float]]],
        colWidths: Optional[List[int]] = None,
    ) -> None:
        """Add a table to the report.

        Args:
            data: Table data as a list of rows
            colWidths: Optional list of column widths
        """
        table = Table(data, colWidths=colWidths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                    ("ALIGN", (0, 1), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        self.elements.append(table)
        self.elements.append(Spacer(1, 0.2 * 72))  # 0.2-inch spacer

    def _add_data_quality_section(self) -> None:
        """Add data quality analysis section to the report."""
        if "data_quality" not in self.analysis_results:
            return

        self._add_section_header("Data Quality Analysis")

        data_quality = self.analysis_results["data_quality"]

        # Add missing values summary
        self._add_subsection_header("Missing Values Summary")
        missing_data = [["Feature", "Missing Count", "Missing Percentage"]]
        for feature, values in data_quality["missing_values"].items():
            missing_data.append(
                [feature, str(values["count"]), f"{values['percentage']:.2f}%"]
            )
        self._add_table(missing_data)

        # Add data types summary
        self._add_subsection_header("Data Types Summary")
        dtypes_data = [["Feature", "Data Type"]]
        for feature, dtype in data_quality["data_types"].items():
            dtypes_data.append([feature, dtype])
        self._add_table(dtypes_data)

        # Add descriptive statistics
        if "descriptive_stats" in data_quality:
            self._add_subsection_header("Descriptive Statistics")
            self._add_paragraph("Summary statistics for numerical features:")
            stats_data = [["Feature", "Mean", "Median", "Std Dev", "Min", "Max"]]
            for feature, stats in data_quality["descriptive_stats"].items():
                stats_data.append(
                    [
                        feature,
                        f"{stats['mean']:.2f}",
                        f"{stats['median']:.2f}",
                        f"{stats['std']:.2f}",
                        f"{stats['min']:.2f}",
                        f"{stats['max']:.2f}",
                    ]
                )
            self._add_table(stats_data)

    def _add_feature_relevance_section(self) -> None:
        """Add feature relevance analysis section to the report."""
        if "feature_relevance" not in self.analysis_results:
            return

        self._add_section_header("Feature Relevance Analysis")

        feature_relevance = self.analysis_results["feature_relevance"]

        # Add feature importance
        self._add_subsection_header("Feature Importance")
        importance_data = [["Feature", "Importance Score"]]
        for feature, score in feature_relevance["feature_importance"].items():
            importance_data.append([feature, f"{score:.4f}"])
        self._add_table(importance_data)

        # Add feature importance plot if available
        feature_importance_plot = self.save_path / "feature_importance.png"
        if feature_importance_plot.exists():
            self._add_image(feature_importance_plot)

    def _add_class_balance_section(self) -> None:
        """Add class balance analysis section to the report."""
        if "class_balance" not in self.analysis_results:
            return

        self._add_section_header("Class Balance Analysis")

        class_balance = self.analysis_results["class_balance"]

        # Add class distribution
        self._add_subsection_header("Class Distribution")
        class_data = [["Class", "Count", "Percentage"]]
        for cls, values in class_balance["class_distribution"].items():
            class_data.append(
                [cls, str(values["count"]), f"{values['percentage']:.2f}%"]
            )
        self._add_table(class_data)

        # Add class distribution plot if available
        class_dist_plot = self.save_path / "class_distribution.png"
        if class_dist_plot.exists():
            self._add_image(class_dist_plot)

    def _add_model_performance_section(self) -> None:
        """Add model performance analysis section to the report."""
        if "model_performance" not in self.analysis_results:
            return

        self._add_section_header("Model Performance Analysis")

        model_perf = self.analysis_results["model_performance"]

        # Add classification metrics
        self._add_subsection_header("Classification Metrics")
        metrics_data = [["Metric", "Value"]]
        for metric, value in model_perf["classification_metrics"].items():
            metrics_data.append([metric.replace("_", " ").title(), f"{value:.4f}"])
        self._add_table(metrics_data)

        # Add confusion matrix
        self._add_subsection_header("Confusion Matrix")
        cm = model_perf["confusion_matrix"]
        cm_data = [
            ["", "Predicted Negative", "Predicted Positive"],
            ["Actual Negative", str(cm["true_negatives"]), str(cm["false_positives"])],
            ["Actual Positive", str(cm["false_negatives"]), str(cm["true_positives"])],
        ]
        self._add_table(cm_data)

        # Add ROC curve plot if available
        roc_plot = self.save_path / "roc_curve.png"
        if roc_plot.exists():
            self._add_image(roc_plot)

    def _add_financial_impact_section(self) -> None:
        """Add financial impact analysis section to the report."""
        if "financial_impact" not in self.analysis_results:
            return

        self._add_section_header("Financial Impact Analysis")

        financial = self.analysis_results["financial_impact"]

        # Add financial metrics
        self._add_subsection_header("Financial Metrics")
        metrics_data = [["Metric", "Value"]]
        metrics_data.append(["Total Profit", f"${financial['total_profit']:.2f}"])
        metrics_data.append(
            ["Opportunity Loss", f"${financial['opportunity_loss']:.2f}"]
        )
        metrics_data.append(["ROI", f"{financial['roi']:.2f}%"])
        self._add_table(metrics_data)

        # Add profit by threshold if available
        if "profit_by_threshold" in financial:
            self._add_subsection_header("Profit by Threshold")
            threshold_data = [["Threshold", "Profit", "ROI"]]
            for threshold, values in financial["profit_by_threshold"].items():
                threshold_data.append(
                    [
                        f"{float(threshold):.2f}",
                        f"${values['profit']:.2f}",
                        f"{values['roi']:.2f}%",
                    ]
                )
            self._add_table(threshold_data)

        # Add financial impact plots if available
        financial_plot = self.save_path / "financial_impact.png"
        if financial_plot.exists():
            self._add_image(financial_plot)

        detailed_plot = self.save_path / "financial_impact_detailed.png"
        if detailed_plot.exists():
            self._add_image(detailed_plot)

    def generate_report(self, filename: str = "model_analysis_report.pdf") -> Path:
        """Generate the PDF report with all analysis sections.

        Args:
            filename: Name of the output PDF file

        Returns:
            Path to the generated PDF file
        """
        # Create the document
        pdf_path = self.save_path / filename
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        # Add title page
        self._add_title_page(
            "Model Analysis Report", "Comprehensive Evaluation and Interpretation"
        )

        # Add executive summary
        self._add_section_header("Executive Summary")
        self._add_paragraph(
            "This report presents a comprehensive analysis of the predictive model "
            "performance, including data quality assessment, feature relevance, "
            "class balance, model performance metrics, and financial impact analysis. "
            "The insights provided in this report aim to support data-driven decision making."
        )

        # Add analysis sections
        self._add_data_quality_section()
        self._add_feature_relevance_section()
        self._add_class_balance_section()
        self._add_model_performance_section()
        self._add_financial_impact_section()

        # Build the document
        doc.build(self.elements)

        logger.info(f"PDF report generated successfully: {pdf_path}")
        return pdf_path

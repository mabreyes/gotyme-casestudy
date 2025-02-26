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

# Import the LLM Description Generator
from src.analysis.llm_description_generator import LLMDescriptionGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generates professional PDF reports for model analysis results."""

    def __init__(
        self,
        analysis_results: Dict,
        save_path: Path,
        use_llm_descriptions: bool = True,
        model_path: Optional[str] = None,
        use_transformers: bool = False,
    ):
        """Initialize the PDF report generator.

        Args:
            analysis_results: Dictionary containing all analysis results
            save_path: Path to save the generated PDF report
            use_llm_descriptions: Whether to use LLM-generated descriptions
            model_path: Path to LLM model file (for llama-cpp)
            use_transformers: Whether to use transformers instead of llama-cpp
        """
        self.analysis_results = analysis_results
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.use_llm_descriptions = use_llm_descriptions

        # Initialize the LLM description generator if enabled
        self.llm_generator = None
        if use_llm_descriptions:
            try:
                self.llm_generator = LLMDescriptionGenerator(
                    model_path=model_path, use_transformers=use_transformers
                )
                logger.info("LLM description generator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM description generator: {e}")
                logger.warning("Falling back to static descriptions")
                self.use_llm_descriptions = False

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

        # Style for LLM-generated descriptions
        self.llm_description_style = ParagraphStyle(
            name="LLMDescriptionStyle",
            parent=self.styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=10,
            leading=12,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=12,
            borderWidth=0.5,
            borderColor=colors.grey,
            borderPadding=10,
            borderRadius=5,
            backColor=colors.lightgrey,
        )

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
        self, image_path: Path, width: int = 400, height: Optional[int] = None
    ) -> None:
        """Add an image to the report.

        Args:
            image_path: Path to the image file
            width: Image width in points
            height: Optional image height in points
        """
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            self._add_paragraph(f"[Image not available: {image_path.name}]")
            return

        # Skip problematic images that are known to cause layout issues
        if (
            hasattr(self, "problematic_images")
            and image_path.name in self.problematic_images
        ):
            logger.info(f"Skipping problematic image: {image_path.name}")
            self._add_paragraph(
                f"[Image '{image_path.name}' is available in the project directory but not shown in the PDF "
                "due to formatting constraints.]"
            )
            return

        try:
            # Define max width (letter page is 612 points wide, with margins we have less)
            max_width = (
                450  # Conservative max width that works well with default margins
            )

            # Get original image dimensions to maintain aspect ratio
            from PIL import Image as PILImage

            try:
                with PILImage.open(str(image_path)) as img:
                    img_width, img_height = img.size
                    aspect_ratio = img_height / img_width

                    # Scale to fit width within max_width
                    scaled_width = min(width, max_width)
                    scaled_height = int(scaled_width * aspect_ratio)

                    logger.info(
                        f"Scaling image {image_path.name} to {scaled_width}x{scaled_height}"
                    )
            except Exception as e:
                logger.warning(
                    f"Could not determine image dimensions for {image_path.name}: {e}"
                )
                # Fallback to provided dimensions
                scaled_width = min(width, max_width)
                scaled_height = height

            # Add the image with controlled dimensions
            img = Image(str(image_path), width=scaled_width, height=scaled_height)
            self.elements.append(img)
            self.elements.append(Spacer(1, 0.2 * 72))  # 0.2-inch spacer
        except Exception as e:
            logger.warning(f"Error adding image {image_path}: {e}")
            # Add a placeholder text instead
            self._add_paragraph(f"[Image could not be displayed: {image_path.name}]")

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

        # Add missing values summary if available
        if "missing_values" in data_quality:
            self._add_subsection_header("Missing Values Summary")
            missing_data = [["Feature", "Missing Count", "Missing Percentage"]]
            for feature, values in data_quality["missing_values"].items():
                # Handle both dictionary format and integer format
                if isinstance(values, dict):
                    # Dictionary format: {"count": X, "percentage": Y}
                    count = str(values["count"])
                    percentage = f"{values['percentage']:.2f}%"
                else:
                    # Integer format: values is just a count
                    count = str(values)
                    # Calculate percentage (assume total records is 100 if not available)
                    total_records = data_quality.get("total_records", 100)
                    percentage = (
                        f"{(values / total_records * 100):.2f}%"
                        if total_records > 0
                        else "0.00%"
                    )

                missing_data.append([feature, count, percentage])
            self._add_table(missing_data)

        # Add data types summary if available
        if "data_types" in data_quality:
            self._add_subsection_header("Data Types Summary")
            dtypes_data = [["Feature", "Data Type"]]
            for feature, dtype in data_quality["data_types"].items():
                dtypes_data.append([feature, dtype])
            self._add_table(dtypes_data)

        # Add descriptive statistics if available
        if "descriptive_stats" in data_quality:
            self._add_subsection_header("Descriptive Statistics")
            self._add_paragraph("Summary statistics for numerical features:")
            stats_data = [["Feature", "Mean", "Median", "Std Dev", "Min", "Max"]]
            for feature, stats in data_quality["descriptive_stats"].items():
                try:
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
                except (KeyError, TypeError) as e:
                    # Handle missing or invalid statistics
                    logger.warning(f"Invalid stats for feature {feature}: {e}")
                    stats_data.append([feature, "N/A", "N/A", "N/A", "N/A", "N/A"])
            self._add_table(stats_data)

    def _add_llm_description(self, description: str) -> None:
        """Add an LLM-generated description with special formatting.

        Args:
            description: The LLM-generated description text
        """
        if description:
            self.elements.append(Paragraph(description, self.llm_description_style))
            self.elements.append(Spacer(1, 0.1 * 72))

    def _add_feature_relevance_section(self) -> None:
        """Add feature relevance analysis section to the report."""
        if "feature_relevance" not in self.analysis_results:
            return

        self._add_section_header("Feature Relevance Analysis")

        feature_relevance = self.analysis_results["feature_relevance"]

        # Add mutual information if available
        if "mutual_information" in feature_relevance:
            self._add_subsection_header("Mutual Information with Target")
            mi_data = [["Feature", "Mutual Information Score"]]
            # Sort by mutual information score in descending order for better readability
            sorted_mi = sorted(
                feature_relevance["mutual_information"].items(),
                key=lambda x: float(x[1])
                if isinstance(x[1], (int, float, str))
                and str(x[1]).replace(".", "", 1).isdigit()
                else 0,
                reverse=True,
            )
            for feature, score in sorted_mi:
                try:
                    mi_data.append([feature, f"{float(score):.4f}"])
                except (ValueError, TypeError):
                    mi_data.append([feature, str(score)])
            self._add_table(mi_data)

            # Add explanation of mutual information
            self._add_paragraph(
                "Mutual Information measures the amount of information obtained about the target "
                "variable when observing each feature. Higher values indicate stronger relevance "
                "to the prediction task."
            )

        # Add feature importance if available
        if "feature_importance" in feature_relevance:
            self._add_subsection_header("Feature Importance")
            importance_data = [["Feature", "Importance Score"]]

            # Sort by importance score in descending order for better readability
            sorted_importance = sorted(
                feature_relevance["feature_importance"].items(),
                key=lambda x: float(x[1])
                if isinstance(x[1], (int, float, str))
                and str(x[1]).replace(".", "", 1).isdigit()
                else 0,
                reverse=True,
            )

            for feature, score in sorted_importance:
                try:
                    importance_data.append([feature, f"{float(score):.4f}"])
                except (ValueError, TypeError):
                    importance_data.append([feature, str(score)])
            self._add_table(importance_data)

            # Add explanation of feature importance
            self._add_paragraph(
                "Feature importance scores indicate how useful each feature was in the construction "
                "of the model. Features with higher importance contributed more to the prediction."
            )

            # Add LLM-generated feature importance description
            if (
                self.use_llm_descriptions
                and self.llm_generator
                and len(sorted_importance) > 0
            ):
                try:
                    llm_description = (
                        self.llm_generator.generate_feature_importance_description(
                            feature_relevance["feature_importance"]
                        )
                    )
                    self._add_subsection_header("Feature Importance Insights")
                    self._add_llm_description(llm_description)
                except Exception as e:
                    logger.warning(
                        f"Failed to generate LLM feature importance description: {e}"
                    )

            # Add feature importance plot if available
            feature_importance_plot = self.save_path / "feature_importance.png"
            if feature_importance_plot.exists():
                self._add_image(feature_importance_plot)

        # Add high correlations if available
        if "high_correlations" in feature_relevance:
            self._add_subsection_header("Highly Correlated Features")
            corr_data = [["Feature 1", "Feature 2", "Correlation"]]

            for corr_info in feature_relevance["high_correlations"]:
                if isinstance(corr_info, (list, tuple)) and len(corr_info) >= 3:
                    try:
                        corr_data.append(
                            [
                                str(corr_info[0]),
                                str(corr_info[1]),
                                f"{float(corr_info[2]):.4f}",
                            ]
                        )
                    except (ValueError, TypeError, IndexError):
                        corr_data.append(
                            [
                                str(corr_info[0]) if len(corr_info) > 0 else "Unknown",
                                str(corr_info[1]) if len(corr_info) > 1 else "Unknown",
                                str(corr_info[2]) if len(corr_info) > 2 else "Unknown",
                            ]
                        )

            if len(corr_data) > 1:  # Only add table if we have data
                self._add_table(corr_data)

                # Add explanation of high correlations
                self._add_paragraph(
                    "Highly correlated features provide similar information and may introduce redundancy "
                    "in the model. Correlation values close to 1 or -1 indicate strong linear relationships "
                    "between features."
                )

        # Add correlation matrix section
        correlation_matrix_plot = self.save_path / "correlation_matrix.png"
        if correlation_matrix_plot.exists():
            self._add_subsection_header("Feature Correlation Matrix")

            self._add_paragraph(
                "The correlation matrix visualizes the pairwise correlation between numerical features. "
                "Darker red indicates strong positive correlation, darker blue indicates strong negative correlation, "
                "and light colors indicate weak correlation."
            )

            # The _add_image method will automatically handle problematic images
            self._add_image(correlation_matrix_plot)

    def _add_class_balance_section(self) -> None:
        """Add class balance analysis section to the report."""
        if "class_balance" not in self.analysis_results:
            return

        self._add_section_header("Class Balance Analysis")

        class_balance = self.analysis_results["class_balance"]

        # Add class counts if available
        if "class_counts" in class_balance:
            self._add_subsection_header("Class Counts")
            class_counts_data = [["Class", "Count"]]
            for cls, count in class_balance["class_counts"].items():
                class_counts_data.append([cls, str(count)])
            self._add_table(class_counts_data)

        # Add class proportions if available
        if "class_proportions" in class_balance:
            self._add_subsection_header("Class Proportions")
            class_proportions_data = [["Class", "Proportion"]]
            for cls, proportion in class_balance["class_proportions"].items():
                try:
                    class_proportions_data.append([cls, f"{float(proportion):.2%}"])
                except (ValueError, TypeError):
                    class_proportions_data.append([cls, str(proportion)])
            self._add_table(class_proportions_data)

        # Add class distribution if available
        if "class_distribution" in class_balance:
            self._add_subsection_header("Class Distribution")
            class_data = [["Class", "Count", "Percentage"]]
            for cls, values in class_balance["class_distribution"].items():
                # Handle both dictionary format and integer format
                if isinstance(values, dict):
                    # Dictionary format: {"count": X, "percentage": Y}
                    count = str(values["count"])
                    percentage = f"{values['percentage']:.2f}%"
                else:
                    # Integer format: values is just a count
                    count = str(values)
                    # Calculate percentage (assume total is sum of all values)
                    total = sum(
                        val if isinstance(val, int) else val.get("count", 0)
                        for val in class_balance["class_distribution"].values()
                    )
                    percentage = (
                        f"{(values / total * 100):.2f}%" if total > 0 else "0.00%"
                    )

                class_data.append([cls, count, percentage])
            self._add_table(class_data)

        # Add interpretation of class balance
        imbalance_message = (
            "Class imbalance can significantly impact model performance. "
        )
        if (
            "class_proportions" in class_balance
            or "class_distribution" in class_balance
        ):
            imbalance_message += "The distribution shown above indicates the relative frequency of each class in the dataset."
        self._add_paragraph(imbalance_message)

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

        # Add LLM-generated model performance description if available
        if self.use_llm_descriptions and self.llm_generator:
            try:
                llm_description = (
                    self.llm_generator.generate_model_performance_description(
                        model_perf
                    )
                )
                self._add_subsection_header("Performance Summary")
                self._add_llm_description(llm_description)
            except Exception as e:
                logger.warning(
                    f"Failed to generate LLM model performance description: {e}"
                )

        # Add optimal threshold if available
        if "optimal_threshold" in model_perf:
            self._add_subsection_header("Optimal Threshold")
            threshold_data = [["Metric", "Value"]]
            try:
                threshold_data.append(
                    [
                        "Optimal Threshold",
                        f"{float(model_perf['optimal_threshold']):.4f}",
                    ]
                )
            except (ValueError, TypeError):
                threshold_data.append(
                    ["Optimal Threshold", str(model_perf["optimal_threshold"])]
                )

            # Add threshold metrics if available
            if "threshold_metrics" in model_perf:
                metrics = model_perf["threshold_metrics"]
                for metric, value in metrics.items():
                    try:
                        threshold_data.append(
                            [metric.replace("_", " ").title(), f"{float(value):.4f}"]
                        )
                    except (ValueError, TypeError):
                        threshold_data.append(
                            [metric.replace("_", " ").title(), str(value)]
                        )

            self._add_table(threshold_data)

            # Add explanation about optimal threshold
            self._add_paragraph(
                "The optimal threshold is the probability cutoff that maximizes model performance "
                "by balancing the trade-off between true positive rate and false positive rate. "
                "This threshold can be adjusted based on business requirements."
            )

        # Add classification metrics if available
        if "classification_metrics" in model_perf:
            self._add_subsection_header("Classification Metrics")
            metrics_data = [["Metric", "Value"]]
            for metric, value in model_perf["classification_metrics"].items():
                try:
                    metrics_data.append(
                        [metric.replace("_", " ").title(), f"{float(value):.4f}"]
                    )
                except (ValueError, TypeError):
                    metrics_data.append([metric.replace("_", " ").title(), str(value)])
            self._add_table(metrics_data)

            # Add explanation of classification metrics
            metrics_explanation = (
                "Classification metrics provide a quantitative assessment of the model's predictive performance. "
                "Accuracy measures overall correctness, precision indicates the reliability of positive predictions, "
                "recall measures the ability to find all positive samples, and F1 score is the harmonic mean of "
                "precision and recall."
            )
            self._add_paragraph(metrics_explanation)

        # Add confusion matrix if available
        if "confusion_matrix" in model_perf:
            cm = model_perf["confusion_matrix"]
            if isinstance(cm, dict) and all(
                k in cm
                for k in [
                    "true_negatives",
                    "false_positives",
                    "false_negatives",
                    "true_positives",
                ]
            ):
                self._add_subsection_header("Confusion Matrix")
                cm_data = [
                    ["", "Predicted Negative", "Predicted Positive", "Total"],
                    [
                        "Actual Negative",
                        str(cm["true_negatives"]),
                        str(cm["false_positives"]),
                        str(cm["true_negatives"] + cm["false_positives"]),
                    ],
                    [
                        "Actual Positive",
                        str(cm["false_negatives"]),
                        str(cm["true_positives"]),
                        str(cm["false_negatives"] + cm["true_positives"]),
                    ],
                    [
                        "Total",
                        str(cm["true_negatives"] + cm["false_negatives"]),
                        str(cm["false_positives"] + cm["true_positives"]),
                        str(
                            cm["true_negatives"]
                            + cm["false_positives"]
                            + cm["false_negatives"]
                            + cm["true_positives"]
                        ),
                    ],
                ]
                self._add_table(cm_data)

                # Add explanation of confusion matrix
                cm_explanation = (
                    "The confusion matrix shows the count of correct and incorrect predictions. "
                    "True Negatives (TN) and True Positives (TP) represent correct predictions, "
                    "while False Positives (FP) and False Negatives (FN) represent incorrect predictions. "
                    "The model correctly predicted "
                    f"{cm['true_positives'] + cm['true_negatives']} out of "
                    f"{cm['true_positives'] + cm['true_negatives'] + cm['false_positives'] + cm['false_negatives']} instances."
                )
                self._add_paragraph(cm_explanation)

        # Add ROC curve plot if available
        roc_plot = self.save_path / "roc_curve.png"
        if roc_plot.exists():
            self._add_image(roc_plot)
            self._add_paragraph(
                "The ROC (Receiver Operating Characteristic) curve shows the trade-off between "
                "true positive rate and false positive rate at different classification thresholds. "
                "A perfect classifier would have an area under the curve (AUC) of 1.0, while a random "
                "classifier would have an AUC of 0.5."
            )

        # Add optimal threshold ROC plot if available
        roc_optimal_plot = self.save_path / "roc_optimal_threshold.png"
        if roc_optimal_plot.exists():
            self._add_image(roc_optimal_plot)
            self._add_paragraph(
                "The plot above highlights the optimal threshold point on the ROC curve, "
                "which represents the best balance between true positive rate and false positive rate."
            )

    def _add_financial_impact_section(self) -> None:
        """Add financial impact analysis section to the report."""
        if "financial_impact" not in self.analysis_results:
            return

        self._add_section_header("Financial Impact Analysis")

        financial = self.analysis_results["financial_impact"]

        # Add LLM-generated financial impact description
        if self.use_llm_descriptions and self.llm_generator:
            try:
                llm_description = (
                    self.llm_generator.generate_financial_impact_description(financial)
                )
                self._add_subsection_header("Financial Impact Summary")
                self._add_llm_description(llm_description)
            except Exception as e:
                logger.warning(
                    f"Failed to generate LLM financial impact description: {e}"
                )

        # Add campaign size if available
        if "campaign_size" in financial:
            self._add_subsection_header("Campaign Overview")
            campaign_data = [["Metric", "Value"]]
            campaign_data.append(["Campaign Size", f"{financial['campaign_size']:,}"])
            self._add_table(campaign_data)

        # Add financial metrics
        self._add_subsection_header("Financial Metrics")
        metrics_data = [["Metric", "Value"]]

        # Add total profit if available
        if "total_profit" in financial:
            try:
                profit_value = financial["total_profit"]
                metrics_data.append(["Total Profit", f"${profit_value:.2f}"])
            except (TypeError, ValueError):
                # Handle non-numeric values
                metrics_data.append(["Total Profit", str(financial["total_profit"])])

        # Add opportunity loss if available
        if "opportunity_loss" in financial:
            try:
                loss_value = financial["opportunity_loss"]
                metrics_data.append(["Opportunity Loss", f"${loss_value:.2f}"])
            except (TypeError, ValueError):
                metrics_data.append(
                    ["Opportunity Loss", str(financial["opportunity_loss"])]
                )

        # Add ROI if available
        if "roi" in financial:
            try:
                roi_value = financial["roi"]
                metrics_data.append(["ROI", f"{roi_value:.2f}%"])
            except (TypeError, ValueError):
                metrics_data.append(["ROI", str(financial["roi"])])

        if len(metrics_data) > 1:  # Only add table if we have data
            self._add_table(metrics_data)

        # Add profit by risk band if available
        if "profit_by_risk_band" in financial:
            self._add_subsection_header("Profit by Risk Band")
            risk_band_data = [["Risk Band", "Profit"]]
            for risk_band, profit in financial["profit_by_risk_band"].items():
                try:
                    risk_band_data.append([risk_band, f"${float(profit):.2f}"])
                except (ValueError, TypeError):
                    risk_band_data.append([risk_band, str(profit)])

            if len(risk_band_data) > 1:  # Only add table if we have data
                self._add_table(risk_band_data)

        # Add scaled confusion matrix if available
        if "scaled_confusion_matrix" in financial:
            self._add_subsection_header("Scaled Confusion Matrix")
            cm = financial["scaled_confusion_matrix"]
            if isinstance(cm, dict) and all(
                k in cm
                for k in [
                    "true_negatives",
                    "false_positives",
                    "false_negatives",
                    "true_positives",
                ]
            ):
                cm_data = [
                    ["", "Predicted Negative", "Predicted Positive", "Total"],
                    [
                        "Actual Negative",
                        f"{cm['true_negatives']:,}",
                        f"{cm['false_positives']:,}",
                        f"{cm['true_negatives'] + cm['false_positives']:,}",
                    ],
                    [
                        "Actual Positive",
                        f"{cm['false_negatives']:,}",
                        f"{cm['true_positives']:,}",
                        f"{cm['false_negatives'] + cm['true_positives']:,}",
                    ],
                    [
                        "Total",
                        f"{cm['true_negatives'] + cm['false_negatives']:,}",
                        f"{cm['false_positives'] + cm['true_positives']:,}",
                        f"{cm['true_negatives'] + cm['false_positives'] + cm['false_negatives'] + cm['true_positives']:,}",
                    ],
                ]
                self._add_table(cm_data)

                # Add interpretation of scaled confusion matrix
                self._add_paragraph(
                    "The scaled confusion matrix shows the predicted distribution of customers "
                    f"in a campaign of {financial.get('campaign_size', 'N/A')} customers. "
                    f"True positives ({cm['true_positives']:,}) represent correctly targeted customers, "
                    f"while false positives ({cm['false_positives']:,}) represent customers incorrectly targeted. "
                    f"False negatives ({cm['false_negatives']:,}) represent missed opportunities."
                )

        # Add profit by threshold if available
        if "profit_by_threshold" in financial:
            self._add_subsection_header("Profit by Threshold")
            threshold_data = [["Threshold", "Profit", "ROI"]]

            for threshold, values in financial["profit_by_threshold"].items():
                try:
                    if isinstance(values, dict):
                        # Handle dictionary format
                        profit = values.get("profit", 0)
                        roi = values.get("roi", 0)
                        threshold_data.append(
                            [f"{float(threshold):.2f}", f"${profit:.2f}", f"{roi:.2f}%"]
                        )
                    else:
                        # Handle non-dictionary values
                        threshold_data.append(
                            [f"{float(threshold):.2f}", str(values), "N/A"]
                        )
                except (ValueError, TypeError) as e:
                    # Handle conversion errors
                    logger.warning(f"Error processing threshold {threshold}: {e}")
                    threshold_data.append([str(threshold), "Error", "Error"])

            if len(threshold_data) > 1:  # Only add table if we have data
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

        # Define potentially problematic images to handle with extra care
        # Note: With our improved scaling, these might work now, but we keep the list
        # in case some images still cause issues
        self.problematic_images = []

        # Previously problematic images that we'll now try to include with proper scaling
        large_complex_images = [
            "correlation_matrix.png",
            "financial_impact_detailed.png",
        ]

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

        # Add information about LLM-enhanced descriptions if enabled
        if self.use_llm_descriptions and self.llm_generator:
            llm_info = (
                "This report includes AI-generated insights based on the analysis data. "
                "These insights appear in highlighted text boxes throughout the report and "
                "are intended to provide additional context and interpretation of the results."
            )
            self._add_paragraph(llm_info)

        # Note about image handling
        image_note = (
            "All visualization images have been automatically scaled to fit this report while maintaining "
            "their aspect ratio for optimal viewing. Some highly detailed visualizations may be better viewed "
            "in their original form in the project's models/plots directory."
        )
        self._add_paragraph(image_note)

        # Safely add each analysis section with error handling
        sections = [
            ("Data Quality", self._add_data_quality_section),
            ("Feature Relevance", self._add_feature_relevance_section),
            ("Class Balance", self._add_class_balance_section),
            ("Model Performance", self._add_model_performance_section),
            ("Financial Impact", self._add_financial_impact_section),
        ]

        for section_name, section_method in sections:
            try:
                section_method()
            except Exception as e:
                logger.error(f"Error generating {section_name} section: {e}")
                self._add_section_header(f"{section_name} (Error)")
                self._add_paragraph(
                    f"An error occurred while generating this section: {str(e)}. "
                    "The data may be incomplete or in an unexpected format."
                )
                # Continue with other sections despite errors

        try:
            # Build the document
            doc.build(self.elements)
            logger.info(f"PDF report generated successfully: {pdf_path}")
            return pdf_path
        except Exception as e:
            logger.error(f"Error building PDF document: {e}")

            # Try to identify which element caused the problem and remove it
            if "too large on page" in str(e) and "filename=" in str(e):
                # Extract the filename from the error message
                import re

                match = re.search(r"filename=([^)]+)", str(e))
                if match:
                    problematic_file = match.group(1)
                    logger.warning(f"Identified problematic file: {problematic_file}")
                    problematic_image = problematic_file.split("/")[-1]
                    self.problematic_images.append(problematic_image)

                    # Add the problematic image to our list for future reference
                    if problematic_image not in large_complex_images:
                        large_complex_images.append(problematic_image)

                    logger.warning(
                        f"Will retry without problematic image: {problematic_image}"
                    )

                    # Create a new set of elements without the problematic images
                    filtered_elements = []
                    for element in self.elements:
                        skip = False
                        if hasattr(element, "filename"):
                            for img_name in self.problematic_images:
                                if img_name in element.filename:
                                    skip = True
                                    logger.info(
                                        f"Skipping problematic image: {img_name}"
                                    )
                                    break
                        if not skip:
                            filtered_elements.append(element)

                    # Try to build the document again with filtered elements
                    try:
                        doc = SimpleDocTemplate(
                            str(pdf_path),
                            pagesize=letter,
                            rightMargin=72,
                            leftMargin=72,
                            topMargin=72,
                            bottomMargin=72,
                        )
                        doc.build(filtered_elements)
                        logger.info(
                            f"PDF report generated successfully (without problematic images): {pdf_path}"
                        )
                        return pdf_path
                    except Exception as e3:
                        logger.error(
                            f"Failed to generate report even without problematic images: {e3}"
                        )

            # If all else fails, create a simplified report
            try:
                # Create a new document with just critical elements
                simplified_elements = []
                simplified_elements.append(
                    Paragraph("Model Analysis Report (Simplified)", self.title_style)
                )
                simplified_elements.append(Spacer(1, 0.5 * 72))
                simplified_elements.append(
                    Paragraph("Error in Report Generation", self.styles["Heading2"])
                )
                simplified_elements.append(Spacer(1, 0.2 * 72))
                simplified_elements.append(
                    Paragraph(
                        f"The full report could not be generated due to an error: {str(e)}. "
                        "This is a simplified version with basic information.",
                        self.normal_style,
                    )
                )

                # Add a note about the problematic images
                if self.problematic_images:
                    simplified_elements.append(Spacer(1, 0.2 * 72))
                    simplified_elements.append(
                        Paragraph("Problematic Images:", self.styles["Heading3"])
                    )
                    simplified_elements.append(Spacer(1, 0.1 * 72))
                    problematic_text = "The following images could not be included in the report due to formatting constraints: "
                    problematic_text += ", ".join(self.problematic_images)
                    simplified_elements.append(
                        Paragraph(problematic_text, self.normal_style)
                    )

                # Save the simplified report
                simple_doc = SimpleDocTemplate(
                    str(pdf_path),
                    pagesize=letter,
                    rightMargin=72,
                    leftMargin=72,
                    topMargin=72,
                    bottomMargin=72,
                )
                simple_doc.build(simplified_elements)
                logger.info(f"Simplified PDF report generated successfully: {pdf_path}")
                return pdf_path
            except Exception as e2:
                logger.error(f"Failed to generate even a simplified report: {e2}")
                raise

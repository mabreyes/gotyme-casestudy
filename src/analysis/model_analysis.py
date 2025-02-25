"""Model analysis module for comprehensive evaluation and interpretation."""
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder

from src.analysis.report_generator import PDFReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Handles comprehensive model analysis and generates interpretable artifacts."""

    def __init__(self, data: pd.DataFrame, model, save_path: Path):
        """Initialize the analyzer.

        Args:
            data: Input DataFrame with features and target
            model: Trained model
            save_path: Path to save analysis artifacts
        """
        self.data = data
        self.model = model
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Split features and target
        self.X = data.drop("Response", axis=1)
        self.y = data["Response"]

        # Initialize results dictionary
        self.analysis_results = {}

        # Store label encoders
        self.label_encoders = {}

    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for prediction.

        Args:
            X: Input features DataFrame

        Returns:
            Encoded features DataFrame
        """
        X_encoded = X.copy()
        categorical_features = X.select_dtypes(include=["object"]).columns

        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X_encoded[col] = self.label_encoders[col].transform(X[col])

        return X_encoded

    def analyze_data_quality(self) -> Dict:
        """Analyze data quality including missing values, outliers, and distributions."""
        results = {
            "missing_values": self.data.isnull().sum().to_dict(),
            "outliers": {},
            "distributions": {},
        }

        # Analyze numerical features
        numerical_features = self.data.select_dtypes(
            include=["float64", "int64"]
        ).columns
        for col in numerical_features:
            # Calculate outliers using IQR method
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (
                (self.data[col] < (Q1 - 1.5 * IQR))
                | (self.data[col] > (Q3 + 1.5 * IQR))
            ).sum()
            results["outliers"][col] = outliers

            # Save distribution plots
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.data, x=col, hue="Response")
            plt.title(f"Distribution of {col} by Response")
            plt.savefig(self.save_path / f"distribution_{col}.png")
            plt.close()

            results["distributions"][col] = {
                "mean": self.data[col].mean(),
                "std": self.data[col].std(),
                "skew": self.data[col].skew(),
            }

        self.analysis_results["data_quality"] = results
        return results

    def analyze_feature_relevance(self) -> Dict:
        """Analyze feature relevance using mutual information and correlation."""
        results = {}

        # Prepare data for mutual information calculation
        X_encoded = self._encode_categorical_features(self.X)

        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X_encoded, self.y)
        results["mutual_information"] = dict(zip(self.X.columns, mi_scores))

        # Calculate correlation matrix for numerical features
        numerical_features = self.X.select_dtypes(include=["float64", "int64"]).columns
        if len(numerical_features) > 0:
            correlation_matrix = self.X[numerical_features].corr()

            # Save correlation heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig(self.save_path / "correlation_matrix.png")
            plt.close()

            # Identify highly correlated features
            high_correlation_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) > 0.8:
                        high_correlation_pairs.append(
                            (
                                correlation_matrix.columns[i],
                                correlation_matrix.columns[j],
                                correlation_matrix.iloc[i, j],
                            )
                        )

            results["high_correlations"] = high_correlation_pairs

        self.analysis_results["feature_relevance"] = results
        return results

    def analyze_class_balance(self) -> Dict:
        """Analyze class balance and its implications."""
        class_counts = self.y.value_counts()
        class_proportions = class_counts / len(self.y)

        results = {
            "class_counts": class_counts.to_dict(),
            "class_proportions": class_proportions.to_dict(),
        }

        # Visualize class distribution
        plt.figure(figsize=(8, 6))
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title("Class Distribution")
        plt.xlabel("Response")
        plt.ylabel("Count")
        plt.savefig(self.save_path / "class_distribution.png")
        plt.close()

        self.analysis_results["class_balance"] = results
        return results

    def analyze_model_performance(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict:
        """Analyze model performance including ROC curves and optimal threshold."""
        # Encode categorical features in test data
        X_test_encoded = self._encode_categorical_features(X_test)

        y_prob = self.model.predict_proba(X_test_encoded)[:, 1]

        # Calculate ROC curve and find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        results = {
            "optimal_threshold": float(optimal_threshold),
            "threshold_metrics": {
                "tpr": float(tpr[optimal_idx]),
                "fpr": float(fpr[optimal_idx]),
            },
        }

        # Plot ROC curve with optimal threshold
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label="ROC curve")
        plt.plot([0, 1], [0, 1], "k--")
        plt.scatter(
            fpr[optimal_idx],
            tpr[optimal_idx],
            color="red",
            label=f"Optimal threshold: {optimal_threshold:.2f}",
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve with Optimal Threshold")
        plt.legend()
        plt.savefig(self.save_path / "roc_optimal_threshold.png")
        plt.close()

        self.analysis_results["model_performance"] = results
        return results

    def analyze_financial_impact(
        self,
        risk_bands: Dict[str, float],
        profit_matrix: Dict[str, Dict[str, float]],
        campaign_size: int = 10000,
    ) -> Dict:
        """Analyze financial impact of the model on a campaign.

        Args:
            risk_bands: Dictionary of risk band proportions
            profit_matrix: Profit/loss values for each risk band and outcome
            campaign_size: Size of the campaign list
        """
        # Encode categorical features for prediction
        X_encoded = self._encode_categorical_features(self.X)

        # Get predictions
        y_pred = self.model.predict(X_encoded)
        tn, fp, fn, tp = confusion_matrix(self.y, y_pred).ravel()
        total = len(self.y)

        # Scale to campaign size
        scale_factor = campaign_size / total
        tn_scaled = int(tn * scale_factor)
        fp_scaled = int(fp * scale_factor)
        fn_scaled = int(fn * scale_factor)
        tp_scaled = int(tp * scale_factor)

        # Calculate profits for each risk band
        total_profit = 0
        opportunity_loss = 0
        profit_by_risk = {}

        for risk, proportion in risk_bands.items():
            # Calculate number of customers in this risk band
            risk_tp = int(tp_scaled * proportion)
            risk_fp = int(fp_scaled * proportion)

            # Calculate profit for this risk band
            profit = (
                risk_tp * profit_matrix[risk]["success"]
                + risk_fp * profit_matrix[risk]["failure"]
            )
            profit_by_risk[risk] = profit
            total_profit += profit

            # Calculate opportunity loss
            risk_fn = int(fn_scaled * proportion)
            opportunity_loss += risk_fn * (
                profit_matrix[risk]["success"] - profit_matrix[risk]["failure"]
            )

        results = {
            "campaign_size": campaign_size,
            "total_profit": total_profit,
            "profit_by_risk_band": profit_by_risk,
            "opportunity_loss": opportunity_loss,
            "scaled_confusion_matrix": {
                "true_negatives": tn_scaled,
                "false_positives": fp_scaled,
                "false_negatives": fn_scaled,
                "true_positives": tp_scaled,
            },
        }

        # Save financial impact visualization
        self.plot_financial_impact(results)

        self.analysis_results["financial_impact"] = results
        return results

    def plot_financial_impact(self, results: Dict) -> None:
        """Create detailed financial impact visualizations."""
        plt.figure(figsize=(15, 5))

        # Plot 1: Profit by risk band
        plt.subplot(1, 3, 1)
        risk_bands = list(results["profit_by_risk_band"].keys())
        profits = list(results["profit_by_risk_band"].values())
        colors = ["red", "yellow", "green"]
        plt.bar(risk_bands, profits, color=colors)
        plt.title("Profit by Risk Band")
        plt.xlabel("Risk Band")
        plt.ylabel("Profit")

        # Plot 2: Total profit vs Opportunity loss
        plt.subplot(1, 3, 2)
        plt.bar(
            ["Total Profit", "Opportunity Loss"],
            [results["total_profit"], results["opportunity_loss"]],
            color=["blue", "orange"],
        )
        plt.title("Profit vs Opportunity Loss")
        plt.ylabel("Amount")

        # Plot 3: Scaled confusion matrix
        plt.subplot(1, 3, 3)
        cm = np.array(
            [
                [
                    results["scaled_confusion_matrix"]["true_negatives"],
                    results["scaled_confusion_matrix"]["false_positives"],
                ],
                [
                    results["scaled_confusion_matrix"]["false_negatives"],
                    results["scaled_confusion_matrix"]["true_positives"],
                ],
            ]
        )
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Scaled Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.tight_layout()
        plt.savefig(self.save_path / "financial_impact_detailed.png")
        plt.close()

    def save_analysis_report(self) -> None:
        """Save comprehensive analysis results to a JSON file."""

        # Define a custom JSON encoder to handle NumPy types
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
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super(NumpyEncoder, self).default(obj)

        try:
            with open(self.save_path / "analysis_report.json", "w") as f:
                json.dump(self.analysis_results, f, indent=2, cls=NumpyEncoder)

            logger.info(
                f"Analysis report saved to {self.save_path / 'analysis_report.json'}"
            )
        except Exception as e:
            logger.error(f"Error serializing analysis results: {e}")

            # Fallback to converting all items to Python native types
            def convert_dict(d):
                result = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        result[k] = convert_dict(v)
                    elif isinstance(v, list):
                        result[k] = [
                            convert_dict(item) if isinstance(item, dict) else item
                            for item in v
                        ]
                    elif isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                        result[k] = int(v)
                    elif isinstance(v, (np.float64, np.float32, np.float16)):
                        result[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        result[k] = v.tolist()
                    elif isinstance(v, np.bool_):
                        result[k] = bool(v)
                    else:
                        result[k] = v
                return result

            serializable_results = convert_dict(self.analysis_results)

            with open(self.save_path / "analysis_report.json", "w") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(
                f"Analysis report saved to {self.save_path / 'analysis_report.json'} (using fallback method)"
            )

    def generate_pdf_report(self, filename: str = "model_analysis_report.pdf") -> Path:
        """Generate a professional PDF report with all analysis results.

        Args:
            filename: Name of the output PDF file

        Returns:
            Path to the generated PDF file
        """
        # Make sure analysis results are saved and serialized
        if not (self.save_path / "analysis_report.json").exists():
            self.save_analysis_report()

        # Load the serialized results to ensure we have Python native types
        with open(self.save_path / "analysis_report.json", "r") as f:
            serialized_results = json.load(f)

        # Generate plots for the report if they don't exist
        self._generate_report_plots()

        # Create and use the PDF report generator
        report_generator = PDFReportGenerator(
            analysis_results=serialized_results, save_path=self.save_path
        )

        # Generate the PDF report
        pdf_path = report_generator.generate_report(filename)

        logger.info(f"PDF report generated successfully: {pdf_path}")
        return pdf_path

    def _generate_report_plots(self) -> None:
        """Generate plots for the PDF report if they don't exist."""
        # Feature importance plot
        if (
            "feature_relevance" in self.analysis_results
            and not (self.save_path / "feature_importance.png").exists()
        ):
            plt.figure(figsize=(10, 6))
            feature_importance = self.analysis_results["feature_relevance"][
                "feature_importance"
            ]
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            features, importance = zip(*sorted_features)

            plt.barh(range(len(features)), importance, align="center")
            plt.yticks(range(len(features)), features)
            plt.xlabel("Importance Score")
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig(self.save_path / "feature_importance.png")
            plt.close()

        # Class distribution plot
        if (
            "class_balance" in self.analysis_results
            and not (self.save_path / "class_distribution.png").exists()
        ):
            plt.figure(figsize=(8, 6))
            class_dist = self.analysis_results["class_balance"]["class_distribution"]
            classes = list(class_dist.keys())
            counts = [info["count"] for info in class_dist.values()]

            plt.bar(classes, counts, color=["skyblue", "salmon"])
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title("Class Distribution")
            plt.tight_layout()
            plt.savefig(self.save_path / "class_distribution.png")
            plt.close()

        # ROC curve plot
        if (
            "model_performance" in self.analysis_results
            and not (self.save_path / "roc_curve.png").exists()
        ):
            if "roc_curve" in self.analysis_results["model_performance"]:
                plt.figure(figsize=(8, 6))
                roc = self.analysis_results["model_performance"]["roc_curve"]
                fpr = roc["fpr"]
                tpr = roc["tpr"]

                plt.plot(
                    fpr,
                    tpr,
                    color="darkorange",
                    lw=2,
                    label=f'ROC curve (AUC = {roc["auc"]:.2f})',
                )
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Receiver Operating Characteristic (ROC) Curve")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(self.save_path / "roc_curve.png")
                plt.close()

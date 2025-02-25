"""Model training module."""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV

from src.analysis import ModelAnalyzer
from src.data.loader import DataLoader
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(
        self, data_path: str | Path, model_save_path: str | Path, random_state: int = 42
    ):
        """Initialize the ModelTrainer.

        Args:
            data_path: Path to the data file
            model_save_path: Path to save the trained model
            random_state: Random state for reproducibility
        """
        self.data_loader = DataLoader(data_path)
        self.model_save_path = Path(model_save_path)
        self.random_state = random_state
        self.model = None

    def train(self) -> None:
        """Train the XGBoost model with hyperparameter tuning."""
        logger.info("Loading and preprocessing data...")
        df = self.data_loader.load_data()
        X_train, X_test, y_train, y_test = self.data_loader.get_train_test_split(
            random_state=self.random_state
        )

        # Create analysis directory
        analysis_path = self.model_save_path / "analysis"
        analysis_path.mkdir(parents=True, exist_ok=True)

        # Initialize analyzer and perform pre-training analysis
        analyzer = ModelAnalyzer(df, None, analysis_path)
        logger.info("Performing pre-training analysis...")
        analyzer.analyze_data_quality()
        analyzer.analyze_feature_relevance()
        analyzer.analyze_class_balance()

        logger.info("Starting hyperparameter tuning...")
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "n_estimators": [100, 200],
            "min_child_weight": [1, 3],
            "gamma": [0, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic", random_state=self.random_state
        )

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_

        # Update analyzer with trained model and perform post-training analysis
        analyzer.model = self.model
        logger.info("Performing post-training analysis...")
        analyzer.analyze_model_performance(X_test, y_test)

        # Perform financial impact analysis
        risk_bands = {"High": 0.10, "Medium": 0.25, "Low": 0.65}
        profit_matrix = {
            "High": {"success": 285.00, "failure": -300.00},
            "Medium": {"success": 705.00, "failure": -300.00},
            "Low": {"success": 1225.00, "failure": -300.00},
        }
        analyzer.analyze_financial_impact(risk_bands, profit_matrix)

        # Evaluate the model and save metrics
        metrics = self._evaluate_model(X_test, y_test)
        logger.info(f"Model evaluation metrics: {metrics}")

        # Save analysis report
        try:
            analyzer.save_analysis_report()
            logger.info("Analysis report saved successfully")
        except Exception as e:
            logger.error(f"Error saving analysis report: {e}")
            # Create a simplified report with basic metrics
            try:
                logger.info("Attempting to create a simplified analysis report...")
                simplified_results = {
                    "model_performance": {
                        "classification_metrics": metrics,
                        "optimal_threshold": 0.5,
                    }
                }
                with open(self.analysis_path / "analysis_report.json", "w") as f:
                    json.dump(simplified_results, f, indent=2)
                logger.info("Simplified analysis report saved successfully")
            except Exception as e2:
                logger.error(f"Failed to create simplified report: {e2}")

        # Generate visualization plots
        self._generate_plots(X_test, y_test)

        # Save the model and metrics
        self._save_model_and_metrics(metrics)

    def _evaluate_model(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate the trained model.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        return metrics

    def _generate_plots(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Generate and save visualization plots.

        Args:
            X_test: Test features
            y_test: Test target
        """
        plots_dir = self.model_save_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Feature importance plot
        plot_feature_importance(
            self.model, X_test.columns, save_path=plots_dir / "feature_importance.png"
        )

        # ROC curve
        y_prob = self.model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_prob, save_path=plots_dir / "roc_curve.png")

        # Confusion matrix
        y_pred = self.model.predict(X_test)
        plot_confusion_matrix(
            y_test, y_pred, save_path=plots_dir / "confusion_matrix.png"
        )

    def _save_model_and_metrics(self, metrics: Dict[str, float]) -> None:
        """Save the trained model and evaluation metrics.

        Args:
            metrics: Dictionary containing evaluation metrics
        """
        # Create model directory if it doesn't exist
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # Save the model
        model_file = self.model_save_path / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump({"model": self.model, "data_loader": self.data_loader}, f)
        logger.info(f"Model saved to {model_file}")

        # Save the metrics
        metrics_file = self.model_save_path / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")


def main():
    """Main function to train the model."""
    data_path = Path("data/dataset.csv")
    model_save_path = Path("models")

    trainer = ModelTrainer(data_path, model_save_path)
    trainer.train()


if __name__ == "__main__":
    main()

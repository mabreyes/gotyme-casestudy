"""Prediction module."""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handles model prediction and financial impact analysis."""

    def __init__(self, model_path: str | Path):
        """Initialize the ModelPredictor.

        Args:
            model_path: Path to the saved model directory
        """
        self.model_path = Path(model_path)
        self.model = None
        self.data_loader = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model and data loader."""
        model_file = self.model_path / 'model.pkl'
        with open(model_file, 'rb') as f:
            saved_data = pickle.load(f)
            self.model = saved_data['model']
            self.data_loader = saved_data['data_loader']
        logger.info("Model loaded successfully")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Args:
            data: Input DataFrame with features

        Returns:
            Array of predictions
        """
        processed_data = self.data_loader.preprocess_new_data(data)
        return self.model.predict(processed_data)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Make probability predictions on new data.

        Args:
            data: Input DataFrame with features

        Returns:
            Array of probability predictions
        """
        processed_data = self.data_loader.preprocess_new_data(data)
        return self.model.predict_proba(processed_data)

    def analyze_financial_impact(
        self,
        data: pd.DataFrame,
        risk_distribution: Dict[str, float] = None
    ) -> Dict[str, float]:
        """Analyze financial impact of predictions.

        Args:
            data: Input DataFrame with features
            risk_distribution: Dictionary with risk band proportions

        Returns:
            Dictionary containing financial metrics
        """
        if risk_distribution is None:
            risk_distribution = {
                'High': 0.10,
                'Medium': 0.25,
                'Low': 0.65
            }

        profit_matrix = {
            'High': {'success': 285.00, 'failure': -300.00},
            'Medium': {'success': 705.00, 'failure': -300.00},
            'Low': {'success': 1225.00, 'failure': -300.00}
        }

        # Get predictions and probabilities
        y_pred = self.predict(data)
        y_prob = self.predict_proba(data)

        # Assign risk bands based on probability of success
        prob_success = y_prob[:, 1]
        risk_bands = pd.qcut(
            prob_success,
            q=[0, 0.35, 0.60, 1.0],
            labels=['Low', 'Medium', 'High']
        )

        # Calculate financial metrics
        total_profit = 0
        profit_by_risk = {'High': 0, 'Medium': 0, 'Low': 0}

        for risk in ['High', 'Medium', 'Low']:
            mask = risk_bands == risk
            n_customers = sum(mask)
            
            # Calculate profits for correct predictions
            correct_success = sum((y_pred == 1) & mask)
            correct_failure = sum((y_pred == 0) & mask)
            
            profit = (
                correct_success * profit_matrix[risk]['success'] +
                correct_failure * profit_matrix[risk]['failure']
            )
            
            profit_by_risk[risk] = profit
            total_profit += profit

        # Calculate opportunity loss
        opportunity_loss = 0
        for risk in ['High', 'Medium', 'Low']:
            mask = risk_bands == risk
            false_negatives = sum((y_pred == 0) & (y_prob[:, 1] > 0.5) & mask)
            opportunity_loss += false_negatives * (
                profit_matrix[risk]['success'] - profit_matrix[risk]['failure']
            )

        return {
            'total_profit': total_profit,
            'profit_by_risk_band': profit_by_risk,
            'opportunity_loss': opportunity_loss
        }


def main():
    """Main function to run predictions and analysis."""
    model_path = Path('models')
    predictor = ModelPredictor(model_path)
    
    # Example: Load and analyze a test dataset
    test_data = pd.read_csv('data/test_dataset.csv', delimiter='|')
    financial_impact = predictor.analyze_financial_impact(test_data)
    
    logger.info(f"Financial Impact Analysis: {json.dumps(financial_impact, indent=2)}")


if __name__ == '__main__':
    main() 
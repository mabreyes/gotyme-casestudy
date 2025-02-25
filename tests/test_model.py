"""Test module for model functionality."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.prediction.predict import ModelPredictor
from src.training.train import ModelTrainer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'Feature_ae_0': [25, 30, 35, 40],
        'Feature_dn_1': [120, 180, 240, 300],
        'Feature_cn_2': [2, 3, 4, 5],
        'Feature_ps_3': [10, 20, 30, 40],
        'Feature_ps_4': [1, 2, 3, 4],
        'Feature_ee_5': [0.1, 0.2, 0.3, 0.4],
        'Feature_cx_6': [95, 96, 97, 98],
        'Feature_cx_7': [80, 82, 84, 86],
        'Feature_em_8': [1.5, 1.6, 1.7, 1.8],
        'Feature_nd_9': [1000, 1100, 1200, 1300],
        'Feature_jd_10': ['admin', 'blue-collar', 'technician', 'admin'],
        'Feature_md_11': ['married', 'single', 'divorced', 'married'],
        'Feature_ed_12': ['primary', 'secondary', 'tertiary', 'primary'],
        'Feature_dd_13': ['no', 'yes', 'no', 'yes'],
        'Feature_hd_14': ['yes', 'no', 'yes', 'no'],
        'Feature_ld_15': ['no', 'yes', 'no', 'yes'],
        'Feature_cd_16': ['cellular', 'telephone', 'cellular', 'telephone'],
        'Feature_md_17': ['jan', 'feb', 'mar', 'apr'],
        'Feature_dd_18': ['mon', 'tue', 'wed', 'thu'],
        'Feature_pd_19': ['success', 'failure', 'success', 'failure'],
        'Response': [1, 0, 1, 0]  # Now we have 2 samples for each class
    }
    return pd.DataFrame(data)


def test_data_loader(sample_data, tmp_path):
    """Test DataLoader functionality."""
    # Save sample data
    data_path = tmp_path / 'test_data.csv'
    sample_data.to_csv(data_path, sep='|', index=False)
    
    # Initialize loader
    loader = DataLoader(data_path)
    
    # Test loading
    df = loader.load_data()
    assert len(df) == 4  # Updated to match new sample size
    assert all(col in df.columns for col in sample_data.columns)
    
    # Test preprocessing
    X, y = loader.preprocess_data(df)
    assert len(X) == 4  # Updated to match new sample size
    assert len(y) == 4  # Updated to match new sample size
    assert all(col in X.columns for col in loader.numerical_features + loader.categorical_features)


def test_model_training(sample_data, tmp_path):
    """Test model training functionality."""
    # Save sample data
    data_path = tmp_path / 'test_data.csv'
    sample_data.to_csv(data_path, sep='|', index=False)
    
    # Create model directory
    model_path = tmp_path / 'models'
    model_path.mkdir()
    
    # Train model
    trainer = ModelTrainer(data_path, model_path)
    trainer.train()
    
    # Check if model and metrics files exist
    assert (model_path / 'model.pkl').exists()
    assert (model_path / 'metrics.json').exists()
    
    # Check metrics format
    with open(model_path / 'metrics.json', 'r') as f:
        metrics = json.load(f)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics


def test_model_prediction(sample_data, tmp_path):
    """Test model prediction functionality."""
    # Save sample data and train model
    data_path = tmp_path / 'test_data.csv'
    sample_data.to_csv(data_path, sep='|', index=False)
    
    model_path = tmp_path / 'models'
    model_path.mkdir()
    
    trainer = ModelTrainer(data_path, model_path)
    trainer.train()
    
    # Initialize predictor
    predictor = ModelPredictor(model_path)
    
    # Test predictions
    predictions = predictor.predict(sample_data.drop('Response', axis=1))
    assert len(predictions) == 4  # Updated to match new sample size
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)
    
    # Test probabilities
    probabilities = predictor.predict_proba(sample_data.drop('Response', axis=1))
    assert probabilities.shape == (4, 2)  # Updated to match new sample size
    assert all(0 <= prob <= 1 for prob in probabilities.flatten())


def test_financial_analysis(sample_data, tmp_path):
    """Test financial impact analysis functionality."""
    # Save sample data and train model
    data_path = tmp_path / 'test_data.csv'
    sample_data.to_csv(data_path, sep='|', index=False)
    
    model_path = tmp_path / 'models'
    model_path.mkdir()
    
    trainer = ModelTrainer(data_path, model_path)
    trainer.train()
    
    # Initialize predictor
    predictor = ModelPredictor(model_path)
    
    # Test financial analysis
    analysis = predictor.analyze_financial_impact(sample_data.drop('Response', axis=1))
    assert 'total_profit' in analysis
    assert 'profit_by_risk_band' in analysis
    assert 'opportunity_loss' in analysis
    assert isinstance(analysis['total_profit'], (int, float))
    assert all(risk in analysis['profit_by_risk_band'] for risk in ['High', 'Medium', 'Low']) 
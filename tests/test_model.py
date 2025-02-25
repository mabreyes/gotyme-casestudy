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
    n_samples = 20
    np.random.seed(42)
    
    # Define categorical values
    job_types = ['admin', 'blue-collar', 'technician']
    marital_status = ['married', 'single', 'divorced']
    education = ['primary', 'secondary', 'tertiary']
    binary_choices = ['yes', 'no']
    contact_methods = ['cellular', 'telephone']
    months = ['jan', 'feb', 'mar', 'apr']
    days = ['mon', 'tue', 'wed', 'thu']
    outcomes = ['success', 'failure']
    
    data = {
        'Feature_ae_0': np.random.normal(35, 10, n_samples),
        'Feature_dn_1': np.random.normal(200, 50, n_samples),
        'Feature_cn_2': np.random.randint(1, 10, n_samples),
        'Feature_ps_3': np.random.normal(25, 10, n_samples),
        'Feature_ps_4': np.random.randint(0, 5, n_samples),
        'Feature_ee_5': np.random.normal(0.3, 0.1, n_samples),
        'Feature_cx_6': np.random.normal(95, 2, n_samples),
        'Feature_cx_7': np.random.normal(82, 3, n_samples),
        'Feature_em_8': np.random.normal(1.6, 0.2, n_samples),
        'Feature_nd_9': np.random.normal(1100, 100, n_samples),
        'Feature_jd_10': np.random.choice(job_types, n_samples).astype(str),
        'Feature_md_11': np.random.choice(marital_status, n_samples).astype(str),
        'Feature_ed_12': np.random.choice(education, n_samples).astype(str),
        'Feature_dd_13': np.random.choice(binary_choices, n_samples).astype(str),
        'Feature_hd_14': np.random.choice(binary_choices, n_samples).astype(str),
        'Feature_ld_15': np.random.choice(binary_choices, n_samples).astype(str),
        'Feature_cd_16': np.random.choice(contact_methods, n_samples).astype(str),
        'Feature_md_17': np.random.choice(months, n_samples).astype(str),
        'Feature_dd_18': np.random.choice(days, n_samples).astype(str),
        'Feature_pd_19': np.random.choice(outcomes, n_samples).astype(str),
        'Response': np.array([1, 0] * (n_samples // 2))
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
    assert len(df) == 20  # Updated to match new sample size
    assert all(col in df.columns for col in sample_data.columns)
    
    # Test preprocessing
    X, y = loader.preprocess_data(df)
    assert len(X) == 20  # Updated to match new sample size
    assert len(y) == 20  # Updated to match new sample size
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
    assert len(predictions) == 20  # Updated to match new sample size
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)
    
    # Test probabilities
    probabilities = predictor.predict_proba(sample_data.drop('Response', axis=1))
    assert probabilities.shape == (20, 2)  # Updated to match new sample size
    assert all(0 <= prob <= 1 for prob in probabilities.flatten())


def test_financial_analysis(sample_data, tmp_path):
    """Test financial impact analysis functionality."""
    # Save sample data and train model
    data_path = tmp_path / 'test_data.csv'
    sample_data.to_csv(data_path, sep='|', index=False)
    
    model_path = tmp_path / 'models'
    model_path.mkdir()
    
    # Train model with the same data to ensure label encoders are fitted correctly
    trainer = ModelTrainer(data_path, model_path)
    trainer.train()
    
    # Initialize predictor
    predictor = ModelPredictor(model_path)
    
    # Create test data with the same categorical values
    test_data = sample_data.copy()
    
    # Test financial analysis
    analysis = predictor.analyze_financial_impact(test_data.drop('Response', axis=1))
    
    # Verify the analysis results
    assert isinstance(analysis, dict)
    assert 'total_profit' in analysis
    assert 'profit_by_risk_band' in analysis
    assert 'opportunity_loss' in analysis
    assert isinstance(analysis['total_profit'], (int, float))
    assert isinstance(analysis['opportunity_loss'], (int, float))
    assert isinstance(analysis['profit_by_risk_band'], dict)
    assert all(risk in analysis['profit_by_risk_band'] for risk in ['High', 'Medium', 'Low']) 
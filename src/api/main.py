"""API module for model serving."""
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.prediction.predict import ModelPredictor
from src.training.train import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global variables
MODEL_PATH = Path('models')
DATA_PATH = Path('data/dataset.csv')
predictor = None


async def predict(request):
    """Handle prediction requests.

    Args:
        request: The HTTP request object

    Returns:
        JSONResponse with predictions and probabilities
    """
    try:
        # Get JSON data from request
        data = await request.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['instances'])
        
        # Make predictions
        predictions = predictor.predict(df)
        probabilities = predictor.predict_proba(df)
        
        # Format response
        response = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        return JSONResponse(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return JSONResponse(
            {'error': str(e)},
            status_code=500
        )


async def analyze_financial_impact(request):
    """Handle financial impact analysis requests.

    Args:
        request: The HTTP request object

    Returns:
        JSONResponse with financial impact analysis
    """
    try:
        # Get JSON data from request
        data = await request.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['instances'])
        
        # Get risk distribution if provided
        risk_distribution = data.get('risk_distribution')
        
        # Analyze financial impact
        financial_impact = predictor.analyze_financial_impact(
            df,
            risk_distribution
        )
        
        return JSONResponse(financial_impact)
    
    except Exception as e:
        logger.error(f"Error during financial analysis: {str(e)}")
        return JSONResponse(
            {'error': str(e)},
            status_code=500
        )


async def train(request):
    """Handle model training requests.

    Args:
        request: The HTTP request object

    Returns:
        JSONResponse with training results
    """
    try:
        global predictor
        
        # Initialize and train model
        trainer = ModelTrainer(DATA_PATH, MODEL_PATH)
        trainer.train()
        
        # Reload predictor with new model
        predictor = ModelPredictor(MODEL_PATH)
        
        # Load and return metrics
        with open(MODEL_PATH / 'metrics.json', 'r') as f:
            metrics = json.load(f)
        
        return JSONResponse({
            'message': 'Model trained successfully',
            'metrics': metrics
        })
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return JSONResponse(
            {'error': str(e)},
            status_code=500
        )


async def startup():
    """Initialize the model predictor on startup."""
    global predictor
    try:
        predictor = ModelPredictor(MODEL_PATH)
        logger.info("Model predictor initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model predictor: {str(e)}")
        raise e


# Define routes
routes = [
    Route('/predict', predict, methods=['POST']),
    Route('/analyze', analyze_financial_impact, methods=['POST']),
    Route('/train', train, methods=['POST'])
]

# Create Starlette application
app = Starlette(
    debug=True,
    routes=routes,
    on_startup=[startup]
)


def main():
    """Run the API server."""
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        log_level='info'
    )


if __name__ == '__main__':
    main() 
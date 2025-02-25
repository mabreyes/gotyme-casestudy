# Sales Offer Prediction Model

This project implements a machine learning solution to predict whether a customer will take up a sales offer when cold-called. The solution uses XGBoost for prediction and provides a REST API using Starlette for model serving.

## Project Structure

```
.
├── data/
│   └── dataset.csv         # Training data (pipe-delimited)
├── models/                 # Directory for saved models
├── src/
│   ├── api/               # API implementation
│   ├── data/             # Data loading and preprocessing
│   ├── prediction/       # Prediction and analysis
│   └── training/         # Model training
├── tests/                 # Unit tests
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── Makefile             # Build automation
├── README.md            # This file
└── pyproject.toml       # Python project configuration
```

## Features

- Data preprocessing and feature engineering
- XGBoost model training with hyperparameter tuning
- Model evaluation and metrics tracking
- Financial impact analysis
- REST API for predictions and analysis
- Docker containerization
- Automated build and deployment using Make

## Requirements

- Python 3.11+
- Docker
- Make

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sales-offer-prediction
```

2. Build the Docker images:
```bash
make build
```

## Usage

### Using Make Commands

- Train the model:
```bash
make train
```

- Start the API server:
```bash
make api
```

- Run predictions:
```bash
make predict
```

### Using the API

The API provides the following endpoints:

1. Train Model
```bash
curl -X POST http://localhost:8000/train
```

2. Make Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"Feature_ae_0": 25, ...}]}'
```

3. Analyze Financial Impact
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{"Feature_ae_0": 25, ...}],
    "risk_distribution": {
      "High": 0.10,
      "Medium": 0.25,
      "Low": 0.65
    }
  }'
```

## Model Details

### Features

The model uses the following features:
- Customer demographics (age, employment type, civil status, education)
- Telematics data (call duration, attempts, campaign history)
- Macro variables (employment rate, CPI, etc.)
- Financial indicators (credit defaults, loans)

### Performance Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Financial Impact Analysis

The analysis considers:
- Risk-based profit matrix
- Expected campaign profit
- Opportunity loss due to misclassification

## Development

### Running Tests

```bash
make test
```

### Code Formatting

```bash
make format
```

### Linting

```bash
make lint
```

## License

[Your License]

## Contributing

[Your Contributing Guidelines] 
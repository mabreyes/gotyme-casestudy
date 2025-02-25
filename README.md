# Sales Offer Prediction Model

This project implements a machine learning solution to predict whether a customer will take up a sales offer when cold-called. The solution uses XGBoost for prediction and provides a REST API using Starlette for model serving.

## Project Structure

```
.
├── data/
│   └── dataset.csv         # Training data (pipe-delimited)
├── models/                 # Directory for saved models
│   ├── analysis/          # Analysis reports and visualizations
│   └── plots/             # Performance and impact plots
├── src/
│   ├── api/               # API implementation
│   ├── data/              # Data loading and preprocessing
│   ├── analysis/          # Model analysis and evaluation
│   ├── prediction/        # Prediction and analysis
│   └── training/          # Model training
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
- Comprehensive model analysis and evaluation
- Financial impact analysis with risk-based assessment
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

### Available Commands

View all available commands:
```bash
make help
```

### Training and Analysis

1. Build the Docker images:
```bash
make build
```

2. Train the model and run analysis:
```bash
make analyze
```

This will:
- Train the XGBoost model with optimized hyperparameters
- Generate analysis reports and visualizations
- Save model artifacts in the `models` directory

3. View results:
```bash
# View analysis report
make view-analysis

# View model metrics
make view-metrics
```

Analysis outputs will be available in:
- Model analysis report: `models/analysis/analysis_report.json`
- Model file: `models/model.pkl`
- Performance metrics: `models/metrics.json`
- Visualizations: `models/plots/`

### Model Performance

Current model metrics (as of latest training):
- Accuracy: 91.49%
- Precision: 64.60%
- Recall: 53.63%
- F1 Score: 58.60%
- Optimal threshold: 0.084 (TPR: 94.40%, FPR: 16.06%)

### Financial Impact Analysis

The model includes risk-based financial impact analysis:
- Risk bands: High (10%), Medium (25%), Low (65%)
- Profit/loss matrix per risk band
- Campaign size analysis (default: 10,000 customers)
- Opportunity loss calculation

### Using the API

Start the API server:
```bash
make serve
```

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

### Development Commands

```bash
# Install dependencies
make install

# Format code
make format

# Run linting
make lint

# Run tests
make test

# Clean up generated files
make clean

# Run all checks (clean, install, format, lint, test)
make all
```

## Model Details

### Features

The model uses the following features:
- Customer demographics (age, employment type, civil status, education)
- Telematics data (call duration, attempts, campaign history)
- Macro variables (employment rate, CPI, etc.)
- Financial indicators (credit defaults, loans)

Key informative features (based on mutual information):
- Feature_dn_1 (0.078)
- Feature_em_8 (0.073)
- Feature_cx_7 (0.069)
- Feature_cx_6 (0.068)

### Data Characteristics

- Class distribution:
  - Negative class (0): 88.77%
  - Positive class (1): 11.23%
- No missing values
- Several features with significant outliers
- High correlations between some features:
  - Feature_em_8 and Feature_ee_5 (0.972)
  - Feature_nd_9 and Feature_em_8 (0.945)
  - Feature_nd_9 and Feature_ee_5 (0.907)

### Performance Analysis

The model evaluation includes:
- ROC curves with optimal threshold analysis
- Confusion matrix visualization
- Feature importance plots
- Distribution analysis by response
- Financial impact visualization

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

# Sales Offer Prediction Model

This project implements a machine learning solution to predict whether a customer will take up a sales offer when cold-called. The solution uses XGBoost for prediction and provides a REST API using Starlette for model serving, along with comprehensive analysis and PDF report generation.

## Project Structure

```
.
├── data/
│   └── dataset.csv         # Training data (pipe-delimited)
├── models/
│   ├── analysis/          # Analysis reports and visualizations
│   │   ├── analysis_report.json  # Analysis results in JSON format
│   │   ├── model_analysis_report.pdf  # PDF report of model analysis
│   ├── plots/             # Performance and impact plots
├── src/
│   ├── api/               # API implementation
│   │   └── main.py        # FastAPI/Starlette API endpoints
│   ├── data/              # Data loading and preprocessing
│   │   └── loader.py      # Data loading and transformation
│   ├── analysis/          # Model analysis and evaluation
│   │   ├── model_analysis.py  # Analysis logic
│   │   └── report_generator.py  # PDF report generation with ReportLab
│   ├── prediction/        # Prediction and analysis
│   │   └── predictor.py    # Model prediction logic
│   └── training/          # Model training
│       └── trainer.py      # XGBoost model training
├── tests/                 # Unit tests
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── Makefile             # Build automation
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── pyproject.toml       # Python project configuration
```

## Features

- **Data preprocessing and feature engineering**:
  - Automatic handling of categorical variables
  - Feature scaling and normalization
  - Missing value imputation

- **XGBoost model training with hyperparameter tuning**:
  - Optimized for imbalanced class distribution
  - Cross-validation for robust evaluation
  - Early stopping to prevent overfitting

- **Comprehensive model analysis and evaluation**:
  - Classification metrics (accuracy, precision, recall, F1)
  - ROC curve analysis with optimal threshold selection
  - Feature importance ranking
  - Confusion matrix visualization

- **Financial impact analysis with risk-based assessment**:
  - Profit/loss calculation based on campaign costs and returns
  - Risk band distribution analysis
  - Opportunity loss calculation
  - ROI optimization

- **✨ AI-POWERED INSIGHTS & DESCRIPTIONS ✨**:
  - Automatic translation of complex metrics into business language
  - LLM-generated explanations of model performance and financial impact
  - Context-aware analysis of data quality and feature importance
  - Makes reports accessible to non-technical stakeholders

- **Detailed PDF report generation**:
  - Professional reports with ReportLab
  - Auto-scaled visualizations
  - Data quality summaries
  - Feature relevance analysis
  - Performance metrics and interpretations
  - Integration with AI-generated insights

- **REST API for predictions and analysis**:
  - `/train` endpoint for model training
  - `/predict` endpoint for making predictions
  - `/analyze` endpoint for financial impact analysis

- **Flexible deployment options**:
  - Docker containerization
  - Automated build and deployment using Make

## Requirements

- Python 3.11+
- Docker and Docker Compose
- Make
- Dependencies listed in `requirements.txt`

### Core Dependencies

- numpy, pandas: Data manipulation
- scikit-learn: Machine learning utilities
- xgboost: Core prediction model
- starlette, uvicorn: API serving
- matplotlib, seaborn: Visualization
- reportlab, pillow: PDF report generation
- pytest: Unit testing
- llama-cpp-python or transformers: LLM inference (optional, for AI-generated descriptions)

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

> **Note**: If you encounter Docker network issues, you can install dependencies locally with `pip install -r requirements.txt` and run commands without Docker.

## Usage

### Available Commands

Run `make help` to view all available commands:

```bash
make help
```

### Training and Analysis Workflow

The recommended workflow is:

1. **Build the Docker images** (if not done already):

   ```bash
   make build
   ```

2. **Train the model and run analysis**:

   ```bash
   make analyze
   ```

   This will train the XGBoost model and generate analysis outputs.

3. **Generate PDF report from analysis results**:

   ```bash
   make generate-pdf-report    # Local Python
   # OR
   make generate-docker-pdf-report    # Using Docker
   ```

4. **Combined training and report generation**:

   ```bash
   make analyze-with-report     # Local Python for report
   # OR
   make analyze-with-docker-report    # Docker for report
   ```

5. **View the results**:

   ```bash
   make view-analysis    # View JSON analysis
   make view-metrics     # View model metrics
   ```

   You can also directly view:
   - Analysis JSON: `models/analysis/analysis_report.json`
   - PDF Report: `models/analysis/model_analysis_report.pdf`
   - Performance Plots: `models/plots/`

### Fixing JSON Serialization Issues

If you encounter JSON serialization issues with NumPy types:

```bash
make fix-json
```

### Using the API

Start the API server:

```bash
make serve
# OR (equivalent command)
make api
```

### API Endpoints

#### Train Model

```bash
curl -X POST http://localhost:8000/train
```

Response:

```json
{
  "success": true,
  "message": "Model trained successfully",
  "metrics": {
    "accuracy": 0.9149,
    "precision": 0.6460,
    "recall": 0.5363,
    "f1_score": 0.5860
  }
}
```

#### Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "Feature_ae_0": 58,
        "Feature_dn_1": 0,
        "Feature_ms_2": 0,
        "Feature_ed_3": 0,
        "Feature_dn_4": 0,
        "Feature_ee_5": 1.9,
        "Feature_cx_6": 5.0,
        "Feature_cx_7": 92.8,
        "Feature_em_8": 3.8,
        "Feature_nd_9": 1.2,
        "Feature_jd_10": 234,
        "Feature_md_11": 3,
        "Feature_ed_12": 2,
        "Feature_cw_13": 4,
        "Feature_pc_14": 1,
        "Feature_hc_15": 0,
        "Feature_dc_16": 0,
        "Feature_lc_17": 0,
        "Feature_rc_18": 0,
        "Feature_pd_19": 0
      }
    ]
  }'
```

Response:

```json
{
  "predictions": [
    {
      "prediction": 0,
      "probability": 0.0782
    }
  ]
}
```

#### Analyze Financial Impact

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "Feature_ae_0": 58,
        "Feature_dn_1": 0,
        "Feature_ms_2": 0,
        "Feature_ed_3": 0,
        "Feature_dn_4": 0,
        "Feature_ee_5": 1.9,
        "Feature_cx_6": 5.0,
        "Feature_cx_7": 92.8,
        "Feature_em_8": 3.8,
        "Feature_nd_9": 1.2,
        "Feature_jd_10": 234,
        "Feature_md_11": 3,
        "Feature_ed_12": 2,
        "Feature_cw_13": 4,
        "Feature_pc_14": 1,
        "Feature_hc_15": 0,
        "Feature_dc_16": 0,
        "Feature_lc_17": 0,
        "Feature_rc_18": 0,
        "Feature_pd_19": 0
      }
    ],
    "risk_distribution": {
      "High": 0.10,
      "Medium": 0.25,
      "Low": 0.65
    }
  }'
```

### Development Commands

```bash
# Install dependencies locally
make install

# Format code with black and isort
make format

# Run linting checks
make lint

# Run tests
make test

# Clean up generated files
make clean

# Run all checks
make all
```

### Custom PDF Report Generation

Generate a custom PDF report with interactive prompts:

```bash
make generate-custom-pdf-report
```

This will prompt you for a report title, subtitle, and output filename.

### Feature Mapping

The API expects features with the following codes:

| Feature Code   | Description                                     | Type                          |
|----------------|-------------------------------------------------|-------------------------------|
| Feature_ae_0   | Age                                             | Numeric                       |
| Feature_dn_1   | Job type                                        | Categorical (numeric encoded) |
| Feature_ms_2   | Marital status                                  | Categorical (numeric encoded) |
| Feature_ed_3   | Education level                                 | Categorical (numeric encoded) |
| Feature_dn_4   | Default on credit                               | Binary (0=no, 1=yes)         |
| Feature_ee_5   | Employment rate                                 | Numeric (economic indicator)  |
| Feature_cx_6   | CPI                                             | Numeric (economic indicator)  |
| Feature_cx_7   | Consumer confidence index                       | Numeric (economic indicator)  |
| Feature_em_8   | Euribor 3 month rate                            | Numeric (economic indicator)  |
| Feature_nd_9   | Number of employees                             | Numeric (in thousands)        |
| Feature_jd_10  | Duration of last contact                        | Numeric (seconds)             |
| Feature_md_11  | Month of last contact                           | Categorical (numeric encoded) |
| Feature_ed_12  | Day of week of last contact                     | Categorical (numeric encoded) |
| Feature_cw_13  | Number of contacts during campaign              | Numeric                       |
| Feature_pc_14  | Days since previous campaign contact            | Numeric                       |
| Feature_hc_15  | Number of previous campaign contacts            | Numeric                       |
| Feature_dc_16  | Previous campaign outcome                       | Categorical (numeric encoded) |
| Feature_lc_17  | Has housing loan                                | Binary (0=no, 1=yes)         |
| Feature_rc_18  | Has personal loan                               | Binary (0=no, 1=yes)         |
| Feature_pd_19  | Previous campaign outcome was successful        | Binary (0=no, 1=yes)         |

#### Categorical Value Mappings

For proper encoding, use these numeric mappings for categorical variables:

**Job types (Feature_dn_1)**:

```python
{
    'admin': 0,
    'blue-collar': 1,
    'entrepreneur': 2,
    'housemaid': 3,
    'management': 4,
    'retired': 5,
    'self-employed': 6,
    'services': 7,
    'student': 8,
    'technician': 9,
    'unemployed': 10,
    'unknown': 11
}
```

**Marital status (Feature_ms_2)**:

```python
{
    'divorced': 0,
    'married': 1,
    'single': 2,
    'unknown': 3
}
```

**Education level (Feature_ed_3)**:

```python
{
    'basic.4y': 0,
    'basic.6y': 1,
    'basic.9y': 2,
    'high.school': 3,
    'illiterate': 4,
    'professional.course': 5,
    'university.degree': 6,
    'unknown': 7
}
```

**Month (Feature_md_11)**:

```python
{
    'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
    'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
}
```

**Day of week (Feature_ed_12)**:

```python
{
    'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4
}
```

**Previous outcome (Feature_dc_16)**:

```python
{
    'failure': 0, 'nonexistent': 1, 'success': 2
}
```

## Model Performance

Current model metrics (as of latest training):

- Accuracy: 91.49%
- Precision: 64.60%
- Recall: 53.63%
- F1 Score: 58.60%
- Optimal threshold: 0.084 (TPR: 94.40%, FPR: 16.06%)

## Financial Impact Analysis

The model includes risk-based financial impact analysis:

- Risk bands: High (10%), Medium (25%), Low (65%)
- Profit/loss matrix per risk band
- Campaign size analysis (default: 10,000 customers)
- Opportunity loss calculation

## PDF Report Generation

The project generates comprehensive PDF reports that include:

- Data quality assessment and summary statistics
- Feature relevance analysis with mutual information and importance scores
- Class balance analysis with distributions
- Model performance metrics and visualization
- Financial impact analysis with risk-band breakdown
- **AI-generated insights and descriptions for each section**

Images in the report are automatically scaled to fit the page width while maintaining their aspect ratio, ensuring optimal readability.

To view the generated PDF report:

```bash
# For Linux/macOS
open models/analysis/model_analysis_report.pdf

# For Windows
start models/analysis/model_analysis_report.pdf
```

## ✨ AI-Generated Descriptions ✨

The PDF reports now include **AI-generated insights and descriptions** powered by a large language model (LLM). These insights help translate complex metrics and analysis results into clear business explanations, making the reports more accessible to non-technical stakeholders.

### Features

- **Automatic generation of explanatory text** for all analysis sections
- **Contextual interpretations** of key metrics and trends
- **Business implications** of statistical findings
- **Visually distinct formatting** for AI-generated content
- Makes reports accessible to stakeholders without statistical expertise

### Setting Up the LLM Model

For the AI descriptions to work, you need to download the TinyLlama model file:

1. Create a directory for the model:
   ```bash
   mkdir -p models/llm
   ```

2. Download the TinyLlama model file (638MB):
   ```bash
   # Using curl
   curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -o models/llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

   # Using wget
   wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
   ```

3. Install required dependencies:
   ```bash
   # Option 1: For llama-cpp-python (recommended, faster)
   pip install llama-cpp-python

   # Option 2: For transformers backend
   pip install transformers torch
   ```

### Controlling AI Descriptions

When generating PDF reports, you can control the LLM feature with these options:

```bash
# Generate a report with AI descriptions (default if model is available)
make generate-pdf-report

# Generate a report without AI descriptions
make generate-pdf-report USE_LLM_DESCRIPTIONS=0

# Specify a custom model path
make generate-pdf-report LLM_MODEL_PATH="/path/to/your/model.gguf"

# Use transformers instead of llama-cpp
make generate-pdf-report USE_TRANSFORMERS=1
```

These options can also be passed as environment variables:

```bash
# Set environment variables before calling make
export USE_LLM_DESCRIPTIONS=1
export LLM_MODEL_PATH="/path/to/your/model.gguf"
export USE_TRANSFORMERS=0

make generate-pdf-report
```

### AI Insights in Docker

You can also use AI-generated insights when generating reports with Docker:

```bash
# Generate with default settings
make generate-docker-pdf-report

# Generate with custom settings
make generate-docker-pdf-report USE_LLM_DESCRIPTIONS=1 LLM_MODEL_PATH="/app/models/llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Ensure all requirements are installed: `pip install -r requirements.txt`
   - For image handling issues, verify Pillow is installed

2. **Docker Build Failures**:
   - Network issues: Try running commands without Docker
   - Memory issues: Increase Docker's allocated memory in settings

3. **Report Generation Issues**:
   - If images don't appear: Check `models/plots` directory for PNG files
   - For JSON serialization errors: Run `make fix-json` to fix the analysis report

4. **API Request Formatting**:
   - Ensure all features are included and properly encoded
   - Refer to the feature mapping table for correct encodings

### Getting Help

For additional issues, please:

1. Check the logs in the terminal output
2. Examine the generated JSON files for any error messages
3. Create an issue in the repository with detailed information about the problem

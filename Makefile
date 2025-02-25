.PHONY: install format lint test clean train analyze predict api build all help generate-pdf-report generate-docker-pdf-report fix-json

install:
	pip install -e .

format:
	black src tests
	isort src tests

lint:
	black --check src tests
	isort --check-only src tests
	mypy src tests

test:
	docker-compose run --rm training pytest tests -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf models/analysis/* models/plots/*

build:
	docker-compose build

train:
	docker-compose run --rm training python -m src.training.train

analyze:
	@echo "Running model analysis..."
	-docker-compose run --rm training python -m src.training.train
	@if [ -f models/analysis/analysis_report.json ]; then \
		echo "Analysis results available in models/analysis/analysis_report.json"; \
		echo "Visualizations available in models/plots/"; \
		echo "Run 'make generate-pdf-report' to create a PDF report of the analysis"; \
	else \
		echo "Warning: Analysis report not generated. There may have been an error."; \
		echo "Run 'make fix-json' to attempt to fix the JSON file."; \
	fi

# Fix JSON serialization issues
fix-json:
	@if [ -f models/analysis/analysis_report.json ]; then \
		python -m src.fix_json models/analysis/analysis_report.json; \
	else \
		echo "Analysis report not found. Run 'make analyze' first."; \
	fi

generate-pdf-report:
	@if [ -f models/analysis/analysis_report.json ]; then \
		python -m src.generate_report -i models/analysis; \
		echo "PDF report generated at models/analysis/model_analysis_report.pdf"; \
	else \
		echo "Analysis report not found. Run 'make analyze' first."; \
	fi

# Generate PDF report using Docker
generate-docker-pdf-report:
	@if [ -f models/analysis/analysis_report.json ]; then \
		docker-compose run --rm report; \
		echo "PDF report generated at models/analysis/model_analysis_report.pdf"; \
	else \
		echo "Analysis report not found. Run 'make analyze' first."; \
	fi

# Custom PDF report with specified title and output name
generate-custom-pdf-report:
	@if [ -f models/analysis/analysis_report.json ]; then \
		read -p "Enter report title: " title; \
		read -p "Enter report subtitle: " subtitle; \
		read -p "Enter output filename: " filename; \
		python -m src.generate_report -i models/analysis -t "$$title" -s "$$subtitle" -o "$$filename"; \
		echo "Custom PDF report generated at models/analysis/$$filename"; \
	else \
		echo "Analysis report not found. Run 'make analyze' first."; \
	fi

# Automatically generate PDF report after analysis
analyze-with-report: analyze generate-pdf-report

# Automatically generate PDF report after analysis using Docker
analyze-with-docker-report: analyze generate-docker-pdf-report

predict:
	docker-compose run --rm api python -m src.prediction.predict

api:
	docker-compose up api

view-analysis:
	@if [ -f models/analysis/analysis_report.json ]; then \
		cat models/analysis/analysis_report.json; \
	else \
		echo "Analysis report not found. Run 'make analyze' first."; \
	fi

view-metrics:
	@if [ -f models/metrics.json ]; then \
		cat models/metrics.json; \
	else \
		echo "Metrics file not found. Run 'make analyze' first."; \
	fi

serve: api

all: clean install format lint test

help:
	@echo "Available commands:"
	@echo "  make install      - Install project dependencies"
	@echo "  make format      - Format code using black and isort"
	@echo "  make lint        - Run linting checks"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean up generated files"
	@echo "  make build       - Build Docker images"
	@echo "  make train       - Train the model with analysis"
	@echo "  make analyze     - Run model analysis"
	@echo "  make fix-json    - Fix JSON serialization issues in analysis report"
	@echo "  make analyze-with-report - Run model analysis and generate PDF report"
	@echo "  make analyze-with-docker-report - Run model analysis and generate PDF report using Docker"
	@echo "  make generate-pdf-report - Generate PDF report from analysis results"
	@echo "  make generate-docker-pdf-report - Generate PDF report using Docker"
	@echo "  make generate-custom-pdf-report - Generate custom PDF report with interactive prompts"
	@echo "  make predict     - Run predictions"
	@echo "  make api         - Start the API server"
	@echo "  make serve       - Alias for 'make api'"
	@echo "  make view-analysis - View the analysis report"
	@echo "  make view-metrics  - View model metrics"
	@echo "  make all         - Run clean, install, format, lint, and test"
	@echo "  make help        - Show this help message"

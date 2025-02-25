.PHONY: install format lint test clean train analyze predict api build all help

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

analyze: train
	@echo "Analysis results available in models/analysis/analysis_report.json"
	@echo "Visualizations available in models/plots/"

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
	@echo "  make predict     - Run predictions"
	@echo "  make api         - Start the API server"
	@echo "  make serve       - Alias for 'make api'"
	@echo "  make view-analysis - View the analysis report"
	@echo "  make view-metrics  - View model metrics"
	@echo "  make all         - Run clean, install, format, lint, and test"
	@echo "  make help        - Show this help message"

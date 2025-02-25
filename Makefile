.PHONY: install format lint test clean train predict api all

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

train:
	docker-compose run --rm training

predict:
	docker-compose run --rm api python -m src.prediction.predict

api:
	docker-compose up api

all: clean install format lint test 
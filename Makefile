.PHONY: help install install-dev test test-cov lint format docker-build docker-run docker-compose-up clean

help:
	@echo "Experimentation Suite - Available Commands:"
	@echo ""
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make test             Run all tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make lint             Run linters (flake8, pylint)"
	@echo "  make format           Format code (black, isort)"
	@echo "  make run-ui           Run Streamlit UI"
	@echo "  make run-api          Run FastAPI server"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run Docker container"
	@echo "  make docker-compose   Start all services with docker-compose"
	@echo "  make clean            Clean temporary files"
	@echo ""

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=modules --cov=utils --cov-report=html --cov-report=term

test-unit:
	pytest tests/ -m unit -v

test-integration:
	pytest tests/ -m integration -v

lint:
	flake8 modules/ utils/ tests/ --max-line-length=100
	pylint modules/ utils/ --max-line-length=100 || true

format:
	black modules/ utils/ tests/ api_server.py
	isort modules/ utils/ tests/ api_server.py

run-ui:
	streamlit run app.py

run-api:
	python api_server.py

run-demo:
	python demo.py

docker-build:
	docker build -t experimentation-suite:latest .

docker-run:
	docker run -p 8501:8501 -p 8000:8000 experimentation-suite:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-compose-logs:
	docker-compose logs -f

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf build dist

all: clean install-dev format lint test

.DEFAULT_GOAL := help

.PHONY: help setup test benchmark run-api run-ui docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make setup       - Install dependencies in virtual environment"
	@echo "  make test        - Run unit and integration tests"
	@echo "  make benchmark   - Run golden set RAG benchmark evaluations"
	@echo "  make run-api     - Launch FastAPI server with SSE streaming (port 8000)"
	@echo "  make run-ui      - Launch Streamlit analytics dashboard (port 8501)"
	@echo "  make docker-up   - Start FastAPI and Streamlit via Docker Compose"
	@echo "  make docker-down - Stop Docker Compose containers"
	@echo "  make clean       - Remove cached files and pycache"

setup:
	source .venv/bin/activate && pip install -r requirements.txt && pip install pytest

test:
	source .venv/bin/activate && PYTHONPATH=. pytest tests/ -v

benchmark:
	source .venv/bin/activate && PYTHONPATH=. python -m src.evaluation.run_evals

run-api:
	source .venv/bin/activate && PYTHONPATH=. uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	source .venv/bin/activate && PYTHONPATH=. streamlit run ui/streamlit_app.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +


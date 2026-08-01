.PHONY: install format lint fetch_data train retrain test api docker-build docker-run pipeline all

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install black isort flake8 mypy

format:
	isort src/*.py app/*.py tests/*.py
	black src/*.py app/*.py tests/*.py

lint:
	flake8 src/*.py app/*.py tests/*.py
	mypy src/*.py app/*.py tests/*.py --ignore-missing-imports

fetch_data:
	python src/data_loader.py

train:
	python src/model_trainer.py

retrain:
	python src/retrain.py

test:
	python -m pytest tests/ -v

api:
	uvicorn app.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t montesinho-api .

docker-run:
	docker run -p 7860:7860 montesinho-api

pipeline: fetch_data train test

all: install format lint pipeline
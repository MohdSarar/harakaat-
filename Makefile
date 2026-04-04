.PHONY: install test lint download prepare train evaluate serve clean docker

# ---- Setup ----
install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# ---- Data ----
download:
	python scripts/download_data.py --dataset all

prepare:
	python scripts/prepare_data.py --config configs/default.yaml

# ---- Training ----
train:
	python scripts/train.py --config configs/default.yaml

train-transformer:
	python scripts/train.py --config configs/transformer.yaml

# ---- Evaluation ----
evaluate:
	python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt

# ---- API ----
serve:
	python scripts/serve.py --port 8000 --checkpoint checkpoints/best.pt

# ---- Testing ----
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# ---- Linting ----
lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

# ---- Docker ----
docker:
	docker build -t arabic-diacritization .

docker-run:
	docker run -p 8000:8000 arabic-diacritization

# ---- Cleanup ----
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf src/__pycache__ src/*/__pycache__
	rm -rf checkpoints/checkpoint_epoch_*.pt
	rm -rf logs/
	find . -name "*.pyc" -delete

clean-data:
	rm -rf data/normalized data/aligned data/splits data/metadata data/lexicons

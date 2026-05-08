.PHONY: install test lint format app train help

PY ?= python3
DATA ?= notebooks/balanced_reviews.csv

help:
	@echo "Targets:"
	@echo "  install   Install pinned dependencies from requirements.txt"
	@echo "  test      Run the pytest suite"
	@echo "  lint      Run ruff check"
	@echo "  format    Run ruff format (writes changes)"
	@echo "  app       Launch the Streamlit dashboard"
	@echo "  train     Fine-tune RoBERTa on \$$DATA (default: $(DATA))"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -v

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

app:
	$(PY) -m streamlit run app.py

train:
	$(PY) -m src.train --data $(DATA)

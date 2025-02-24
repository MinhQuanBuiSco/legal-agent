.PHONY: style quality

# Variables

export PYTHONPATH = src

check_dirs := src tests

venv:
	uv venv --python=3.11

install: venv
	. .venv/bin/activate && uv pip install -e .

style:
	ruff format --line-length 320  $(check_dirs)
	isort $(check_dirs)

quality:
	ruff check --line-length 119 $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 --max-line-length 4000 $(check_dirs)	
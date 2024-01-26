DIRS_PYTHON := .

.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo
	@echo "Targets:"
	@echo "  help    This help (default target)"
	@echo "  deps    Install all dependencies"
	@echo "  format  Format source code"
	@echo "  lint    Run lint checks"
	@echo "  jupyterlab Run jupyterlab"

.PHONY: deps
deps:
	pipenv install --dev

.PHONY: format
format:	\
	format-isort \
	format-black

.PHONY: format-isort
format-isort:
	pipenv run isort --profile=black $(DIRS_PYTHON)

.PHONY: format-black
format-black:
	pipenv run black --line-length 100 $(DIRS_PYTHON)

.PHONY: lint
lint: \
	lint-isort \
	lint-black \
	lint-flake8 \
	lint-mypy

.PHONY: lint-isort
lint-isort:
	pipenv run isort --profile=black --check-only --diff $(DIRS_PYTHON)

.PHONY: lint-black
lint-black:
	pipenv run black --check --line-length 100 --diff $(DIRS_PYTHON)

.PHONY: lint-flake8
flake8:
	pipenv run flake8 --max-line-length 100 $(DIRS_PYTHON)

.PHONY: lint-mypy
lint-mypy:
	MYPYPATH=$(PWD)/stubs pipenv run mypy $(DIRS_PYTHON)

.PHONY: jupyterlab
jupyterlab:
	cp utils/minimal.ipynb tmp.ipynb && \
	PYTHON=. pipenv run \
		jupyter lab \
			--ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://127.0.0.1:8888 \
			tmp.ipynb

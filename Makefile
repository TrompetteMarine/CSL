.PHONY: help install verify reproduce figures tables paper clean
export PYTHONPATH := .
SCALE ?= standard

help:
	@echo "make install    - pip install -r requirements.txt"
	@echo "make verify     - run the pytest verification suite (19 tests)"
	@echo "make reproduce  - simulate all stages + render figures/tables (SCALE=$(SCALE))"
	@echo "make figures    - render figures only (needs existing results)"
	@echo "make tables     - render tables only (needs existing results)"
	@echo "make paper      - build the manuscript PDF (needs pdflatex/bibtex)"
	@echo "make clean      - remove caches and simulation outputs"

install:
	pip install -r requirements.txt

verify:
	python -m pytest tests/ -q

reproduce:
	python scripts/run_all.py $(SCALE)
	python scripts/make_figures.py $(SCALE)
	python scripts/make_neuro_figures.py
	python scripts/make_tables.py $(SCALE)

figures:
	python scripts/make_figures.py $(SCALE)
	python scripts/make_neuro_figures.py

tables:
	python scripts/make_tables.py $(SCALE)

paper:
	cd paper && pdflatex -interaction=nonstopmode endogenous_credibility.tex && \
	  bibtex endogenous_credibility && \
	  pdflatex -interaction=nonstopmode endogenous_credibility.tex && \
	  pdflatex -interaction=nonstopmode endogenous_credibility.tex

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache code/results results

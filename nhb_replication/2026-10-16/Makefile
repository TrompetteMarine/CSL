PYTHON ?= python
ARCHIVES_DIR ?= downloads
REPOSITORY ?= TrompetteMarine/decision-generated-credibility
TAG ?= v2026.10.16

.PHONY: help install check syntax test verify-records engine-check data metadata verify-data full-audit statistics figures workbooks verify-lean

help:
	@printf '%s\n' 'make check          Syntax, archive safety tests, saved Lean records and 18 dense-engine comparisons' 'make data           Download and verify the three raw-array/configuration release assets' 'make metadata       Download configuration metadata only' 'make verify-data    Hash all restored payload files (664 arrays)' 'make full-audit     Recompute all saved-array, graph and precision checks (requires make data)' 'make statistics     Recompute Monte Carlo intervals and covariance calculations' 'make figures        Rebuild standalone vector figures (requires data, statistics and TeX)' 'make workbooks      Rebuild portable source-data Excel workbooks (optional requirements)' 'make verify-lean    Fresh Lean kernel build with the pinned toolchain and dependencies'

install:
	$(PYTHON) -m pip install -r requirements.txt

syntax:
	$(PYTHON) tools/check_syntax.py

test:
	$(PYTHON) -m unittest discover -s tests -v

verify-records:
	$(PYTHON) tools/verify_package.py --require-source-manifest

engine-check:
	$(PYTHON) audit/check_engine.py

check: syntax test verify-records engine-check

data:
	$(PYTHON) tools/prepare_data.py --download --archives-dir "$(ARCHIVES_DIR)" --repository "$(REPOSITORY)" --tag "$(TAG)"

metadata:
	$(PYTHON) tools/prepare_data.py --download --metadata-only --archives-dir "$(ARCHIVES_DIR)" --repository "$(REPOSITORY)" --tag "$(TAG)"

verify-data:
	$(PYTHON) tools/verify_package.py --require-source-manifest --with-data

full-audit: verify-data engine-check
	$(PYTHON) audit/recompute.py
	$(PYTHON) audit/check_graphs.py
	$(PYTHON) audit/check_precision.py

statistics:
	$(PYTHON) statistics/analyse_uncertainty.py
	$(PYTHON) statistics/conditional_noise.py
	$(PYTHON) statistics/write_statistics.py
	$(PYTHON) statistics/verify_statistics.py

figures:
	$(PYTHON) figures/rebuild_figures.py
	$(PYTHON) tools/build_tikz.py
	$(PYTHON) figures/export_source_data.py
	$(PYTHON) figures/final_figure_audit.py

workbooks:
	$(PYTHON) source_data/prepare_workbook_inputs.py
	$(PYTHON) source_data/build_workbooks.py --verify

verify-lean:
	$(PYTHON) formal/verify_all.py

#!/usr/bin/env python3
"""Compile the four standalone diagrams; no manuscript TeX is present."""
from pathlib import Path
import shutil
import subprocess
import sys

import fitz

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
SOURCES = ["main01_concept", "main06_population_limits", "supp09_signal_anatomy", "supp10_noise_propagation"]

if shutil.which("pdflatex") is None:
    raise SystemExit("pdflatex is required: install a TeX distribution with standalone, TikZ and Latin Modern packages.")
for name in SOURCES:
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"],
                            cwd=FIGURES, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-6000:], file=sys.stderr)
        raise SystemExit(result.returncode)
    with fitz.open(FIGURES / (name + ".pdf")) as document:
        if len(document) != 1:
            raise SystemExit(f"Standalone diagram has {len(document)} pages: {name}")
        document[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False).save(FIGURES / (name + ".png"))
    print(f"Compiled {name}.pdf and 300-dpi preview.")

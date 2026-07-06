"""Driver: regenerate every result, figure and table for a given scale.

Usage:
    python scripts/run_all.py [preview|standard|publication]

Running from a clean checkout regenerates all results from fixed seeds, then
renders every manuscript figure and table; nothing is produced or edited by
hand. Resumable stages (phase, quotient) are looped until they report
completion, so this single command works both locally (one pass) and under a
wall-clock budget (set ECSL_BUDGET, in seconds, to force chunked runs).
"""
import os
import subprocess
import sys

import numpy as np

import common

PY = sys.executable
HERE = os.path.dirname(__file__)


def run(script, scale):
    subprocess.run([PY, os.path.join(HERE, script), scale], check=True)


def loop_until_done(script, result_file, scale, key="done"):
    path = f"{common.RESULTS}/{result_file}"
    for _ in range(200):
        run(script, scale)
        if os.path.exists(path):
            d = np.load(path)
            if key in d and bool(np.all(d[key])):
                return
    raise RuntimeError(f"{script} did not complete")


def main():
    scale = common.get_scale()
    print(f"== regenerating all results at scale '{scale}' ==")
    # non-resumable simulation stages
    run("run_baseline.py", scale)      # baseline_<scale>.{npz,json}
    run("run_ablations.py", scale)     # ablations_<scale>.{npz,json}
    run("run_confidence.py", scale)    # confidence_<scale>.{npz,json}
    run("run_early_lead.py", scale)    # early_lead_<scale>.npz    (endogenous early-lead robustness)
    # resumable / budgeted sweeps — MUST precede run_robustness: its R6 grid-resolution
    # check reads phase_<scale>.npz, so the phase grid has to exist first.
    loop_until_done("run_phase.py",    f"phase_{scale}.npz",         scale)
    loop_until_done("run_quotient.py", f"quotient_grid_{scale}.npz", scale)
    run("run_robustness.py", scale)    # robustness_<scale>.json   (needed by fig7 + tables; R6 uses the phase grid)
    # figures and tables
    run("make_figures.py",       scale)   # Figs 2-7 (Monte-Carlo)
    run("make_neuro_figures.py", scale)   # neuro-mechanism figures
    run("make_tables.py",        scale)   # results / ablation / robustness tables
    print("== done: figures in paper/figures, tables in paper/tables ==")


if __name__ == "__main__":
    main()

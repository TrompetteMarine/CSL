# CSL — Cognitive Social Learning

Reference implementation and replication package for
**"Endogenous Credibility in Social Learning."**

Confidence is generated endogenously by a drift–diffusion decision process and
then acts as a *credibility* weight in social transmission. Because confidence
can attach to correct or incorrect early choices, the same network links can
transmit correction or amplify error. The central question is:

> **When does credibility-weighted social transmission make connectivity
> corrective, and when does it make it distortive?**

Every figure and table in the manuscript is regenerated from fixed seeds by the
commands below. No result is produced or edited by hand.

---

## 1. Install

Python 3.10+.

```bash
git clone https://github.com/<owner>/CSL.git
cd CSL
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `pyyaml`, `pytest`.

The package imports as `credibility` and is run with the repository root on the
path (`PYTHONPATH=.`, as in the commands below). An editable install is also
supported:

```bash
pip install -e .
```

## 2. Verify (do this first)

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

19 tests must pass. They form the verification contract: exact drift–diffusion
identities (choice probability `logistic(2va/σ²)`, mean first-passage time
`(a/v)·tanh(av/σ²)`), the confidence-map bounds, the balanced-block quotient
identity, common-random-number ablation invariances, and Wilson/bootstrap
interval coverage.

## 3. Reproduce all results

```bash
# 1) simulate every stage at the standard profile (~10 min, single-threaded)
PYTHONPATH=. python scripts/run_all.py standard

# 2) render figures and tables
PYTHONPATH=. python scripts/make_figures.py       standard
PYTHONPATH=. python scripts/make_neuro_figures.py            # Figs 2,3 (neuro)
PYTHONPATH=. python scripts/make_dynamics_figures.py 21      # Figs 6-8 (mean-field)
PYTHONPATH=. python scripts/make_tables.py        standard
```

The `make_dynamics_figures.py` script is analytic (no Monte Carlo): it draws the
bifurcation diagram, the two-community phase portrait, and the deterministic
`21x21` (lambda, Gamma) phase map from `credibility/meanfield.py`.

### Maximum-capacity Monte-Carlo phase diagram
The stochastic `21x21` phase sweep at the publication profile (600 reps/cell) is
heavy but **resumable at the cell level** — run it repeatedly (or set a
per-invocation wall budget) until it prints `ALL DONE`:

```bash
ECSL_BUDGET=1800 PYTHONPATH=. python scripts/run_phase.py publication   # re-run to continue
```

Or with the Makefile:

```bash
make verify       # pytest
make reproduce    # run_all + figures + tables at the standard profile
make paper        # build the manuscript PDF (needs a LaTeX toolchain)
```

Profiles (`configs/scales.json`): `preview` (fast smoke test), `standard`
(paper-grade), `publication` (large sweeps, hours). Mechanism parameters are
identical across profiles; only replicate counts and grid resolutions change.

### Reproducibility guarantees
- **All randomness is seeded.** Each stage fixes its own base seed.
- **Ablations use common random numbers (CRN):** one environment draw is shared
  across mechanism variants, so differences are causal, not sampling noise.
- **Resumable sweeps.** `run_all.py` loops the phase/quotient stages until they
  report completion; set `ECSL_BUDGET` (seconds) to force chunked runs under a
  wall-clock budget.

## 4. Repository layout

```
credibility/            core model (importable package)
  ddm.py                drift–diffusion: exact choice prob. & first-passage time
  confidence.py         endogenous confidence maps (decision-time & balance)
  model.py              agent updates: confidence-gated private + social learning
  network.py            community blocks, balanced-block quotient
  meanfield.py          deterministic N->inf map: bifurcation, flow, phase map
  metrics.py            regime classification, group regret
  uncertainty.py        Wilson intervals, percentile & paired bootstrap
  scenarios.py          named environments (efficient / wrong / polarised)
  config.py             Params dataclass, profiles (SCALE)
scripts/                one stage per file + run_all driver, figure/table makers
tests/                  verification suite (pytest)
configs/                default parameters and profile definitions
paper/                  manuscript (.tex/.bib/.pdf) and all figures/tables
```

## 5. Model in one paragraph

Each trial, an agent forms a value contrast `Δ = Q(1) − Q(0)`, which drives a
drift `v = βΔ` in a bounded accumulator. The first boundary hit gives the choice;
the drift magnitude and decision time give an endogenous confidence
`C = σ(κ₁|v|/a − κ₂ log(1+τ/τ₀))`. Confidence gates plasticity two ways: it sets
a valence-asymmetric private learning rate (optimistic when low, revision-driven
when high), and it weights social transmission through an anticipatory field and
a retrospective social prediction error, with credibility weight
`wᵢ = Cᵢ / Σⱼ Cⱼ`. On balanced community blocks the dynamics admit an exact
low-dimensional quotient; the local amplification threshold
`λ⋆ = σ²/(βa C₀ b̄)` marks where an early confident lead locks in. See the
manuscript for definitions, propositions (proofs in Appendix A), the
identification strategy, and the neuroeconomic reading.

## 6. Citation

See `CITATION.cff`. Please cite the manuscript and this repository (release tag /
archive DOI to be added on publication).

## 7. License

MIT — see `LICENSE`.

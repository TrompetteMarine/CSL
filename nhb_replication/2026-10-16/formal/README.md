# Exact proof audit and supplementary Lean source

This directory contains the preserved reviewed declarations, additional finite-model proofs, exact type reports, and transitive axiom reports for the revision dated 16 October 2026.

The main scientific claims are not all Lean theorems. The coverage ledger distinguishes a complete formal statement from a conventional proof that uses a formalised auxiliary identity. In particular, finite-horizon stochastic approximation, Feller invariant-law existence, sparse local convergence and empirical laws, the Watts–Strogatz probability argument, and the joint limit still require the written proofs and their stated hypotheses. Numerical records are independently checked computations, not Lean proofs.

## Reproduce

Use Lean 4.28.0, as specified by `project/lean-toolchain`. The Lake manifest fixes mathlib to `8f9d9cff6bd728b17a24e163c9402775d9e6a365` and fixes all eight transitive repositories. With the `lean` and `lake` binaries on PATH:

```bash
cd project
lake update
lake exe cache get
cd ..
python verify_all.py
```

The verification script refuses changed protected interfaces, changed reviewed module bytes, changed dependency pins, tracked changes in dependency repositories, forbidden proof bypasses, missing type/axiom records, or any logical axiom beyond `propext`, `Classical.choice` and `Quot.sound`. It rebuilds all source libraries before producing its acceptance summary. All newly introduced results are in `project/RepairSupplement/`.

The local execution environment required a narrow `readlink` compatibility shim because `/proc/<getpid>/exe` was absent while `/proc/self/exe` existed. This changes executable discovery only. The Lean kernel and proof sources were not changed. Ordinary installations do not need this shim. Container extraction used `TAR_OPTIONS=--no-same-owner` because archived file owners were outside the container's user mapping; this does not affect proof checking.

The toolchain and dependency caches are intentionally excluded from the source deliverable. `verification_summary.json`, `declaration_audit.json`, `baseline_contract_audit.json`, `dependency_audit.json`, `execution_steps.json`, and `logs/` record what was actually checked. `ReviewAllTypes.lean` and `ReviewAllAxioms.lean` can be rerun independently from the project directory.

## Scope of the source modules

| Module | Scope |
| --- | --- |
| `RepairCore/Algebra.lean` | Exact block quotient, self exclusion, convex gain bounds, row-stochastic and broadcast identities |
| `RepairTasks/Analysis.lean` | Signed readout derivative at the origin under continuity of the score, zero-floor case, oscillation and self-exclusion ratio bounds |
| `RepairProb/P1.lean` | Conditional signed mean and fixed-private-score Jacobian |
| `RepairProb/P2.lean` | Realised-sign Bernoulli drift and private-only zero-drift target |
| `RepairProb/P3.lean` | Exploration floor, finite avoidance bound, repeated sampling under explicit bounded-state and filtration hypotheses |
| `RepairSupplement/Finite.lean` | Literal update bounds, admissible peer masses, signed averages, deterministic score invariance, two-community modes, signal scaling, withdrawal and finite-population expression bound, Bernoulli variance algebra |
| `RepairSupplement/PeerRegularity.lean` | Actual coordinate derivatives and derivative bounds for the peer increment; exact self-exclusion bound |
| `RepairSupplement/LimitsAux.lean` | Deterministic recurrence and finite-cone infrastructure; does not by itself establish a probabilistic population limit |

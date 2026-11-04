# formal_diss — machine-checked algebraic core of the dissonance-gated learning model

Seven theorems certifying the algebraic skeleton of the model in which legitimacy-gated
reinforcement learning and threshold delegitimation are the margin-wise gradient steps of
one cognitive-dissonance functional.

## Files

| File | Theorems | Frozen in |
|---|---|---|
| `DissCore.lean` | T1 (exact quadratic expansion of the dissonance functional), T2 (drift sign ⟺ cleared inequality), T2b (cleared inequality ⟺ A < A* under positivity), T3 (between-community gap ⟹ geometric decay (1−2ψΓ)^t) | KGi-DISS-I002 |
| `DissBridge.lean` | T4 (branch-weighted E[ΔL] > 0 ⟺ cleared inequality, region R1), T5a (gated contraction Q′−μ = (1−αL)(Q−μ)), T5b (n-step geometric form) | KGi-DISS-I003 |
| `verification_record_core.json`, `verification_record_bridge.json` | compile logs, axiom closures, file hashes at freeze time | |
| `DissCore.sha256`, `DissBridge.sha256` | content hashes of the frozen sources | |
| `expected_output_core.txt`, `expected_output_bridge.txt` | expected `#print axioms` output | |

Chaining T4 → T2 → T2b certifies end-to-end: **E[ΔL] > 0 ⟺ A < A\*(μ)** in the generic
drift region, where A\*(μ) = (τ^F + (ν_R/ν_F)·μ/(1−μ))/φ⁻.

## Scope

Algebra only, over ℚ (the rational parameter space used in all experiments). Nothing
stochastic, network-level, or behavioural is certified by these files; those claims rest on
the cited stochastic-approximation literature and on frozen simulation evidence (see the
paper's Supplementary Information).

## How to check

Official Lean 4 release v4.28.0, **core library only** (no mathlib, no lake project):

```
lean DissCore.lean
lean DissBridge.lean
```

Each file ends with `#print axioms` for every theorem; the expected axiom closure of every
declaration is exactly `[propext, Classical.choice, Quot.sound]`, with no `sorry`.
Re-verified before this commit (2026-11-04) with the same toolchain.

## Provenance

Produced and frozen under the KGi protocol (freeze-before-run, hash-frozen evidence
bundles) during iterations KGi-DISS-I002 and KGi-DISS-I003 of the dissonance-gated social
learning project. The verification records in this directory are verbatim copies of the
frozen bundle records.

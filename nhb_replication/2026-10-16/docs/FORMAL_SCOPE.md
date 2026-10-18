# Formal verification scope

The retained project pins Lean 4.28.0 and mathlib revision `8f9d9cff6bd728b17a24e163c9402775d9e6a365`. It preserves the reviewed source bytes, protected interfaces and nine Git dependency revisions.

The saved kernel/type/axiom records contain 93 theorem or lemma declarations and 24 definitions. Their transitive logical dependencies are restricted to `propext`, `Classical.choice` and `Quot.sound`. Exact declaration types and transitive axiom output are retained in `formal/logs/`; structured declaration and dependency audits accompany them as JSON records.

These records cover finite-update bounds and deterministic invariance, restricted two-community calculations, withdrawal bounds, peer-increment derivatives, signal rescaling, variance algebra and deterministic recurrence/locality infrastructure. They do not establish a human-confidence interpretation or validate numerical experiments.

Complete large-community stochastic coupling, Feller/invariant-law arguments, sparse local convergence, empirical laws of large numbers, graph-limit probability estimates and joint sparse/large-community limits are not all formalised. New statistical covariance discussions do not enlarge the saved Lean declaration count.

`make check` validates source hashes and existing audit records. To perform a **fresh kernel build**, install the pinned toolchain, fetch the dependencies at the revisions in `formal/project/lake-manifest.json`, then run:

```bash
cd formal/project
lake build RepairCore RepairTasks RepairProb RepairSupplement
cd ../..
python formal/verify_all.py
```

The verification driver rejects changed dependency revisions and tracked dependency edits. Check the manifest after dependency restoration; do not accept an updated dependency pin to make a failing build pass. A clean restored dependency checkout is required before the driver's checks.

`CSL_LEAN_BIN` can select an installed Lean binary directory. Historical sandbox compatibility binaries and cached toolchains are excluded from the repository. Ordinary Lean installations do not require those sandbox workarounds. No file in this repository makes a fresh-build claim merely by replaying a historical log.

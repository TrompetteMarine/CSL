/- Statement-fidelity audit for DissAggregation.lean (KGi-DISS-I039).
   Checked by concatenation:  cat DissAggregation.lean DissAggregationAudit.lean > /tmp/x.lean ; lean /tmp/x.lean
   (core Lean has no project here, so the audit is appended rather than imported).
   Each `example` restates a target with the definitions written out by hand and closes it
   with the certified theorem; a mismatch between the two statements would not type-check. -/

-- T6 with `gain` expanded and the hypotheses substituted.
example (kappa q delta : Rat) : q + kappa * (1 - 1 * (1 - 0)) * delta = q :=
  T6_freeze kappa q delta 1 0 rfl rfl

-- T8 with the covariance term written as (mean of products) - (product of means).
example (p g1 g2 d1 d2 : Rat) :
    (p * (g1 * d1) + (1 - p) * (g2 * d2)) - (p * g1 + (1 - p) * g2) * (p * d1 + (1 - p) * d2)
      = p * (1 - p) * ((g1 - g2) * (d1 - d2)) := by
  have h := T8_aggregation_identity p g1 g2 d1 d2
  grind

-- T11 at t = 2 written without `cpow`: (p x + (1-p) y)^2 ≤ p x^2 + (1-p) y^2.
example (p x y : Rat) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    (p * x + (1 - p) * y) * ((p * x + (1 - p) * y) * 1) ≤ p * (x * (x * 1)) + (1 - p) * (y * (y * 1)) :=
  T11_horizon p x y hp0 hp1 hx hy 2

-- The hypotheses of T13 are satisfiable at the frozen parameter values
-- (nuF = 1/20, nuR = 1/80, tau = 3/10, phi_plus = 1/4, rho = 1/500, chi = 2/5), and the
-- conclusion is instantiated on a concrete six-trial history with a higher board.
def frozenTrials : List (Bool × Rat × Rat) :=
  [(false, 1/4, 18/25), (false, 1/4, 18/25), (true, 1/4, 18/25),
   (false, 1/4, 18/25), (false, 1/4, 18/25), (false, 1/4, 18/25)]

example :
    (run (1/20) (1/80) (3/10) (1/4) (1/500) (2/5) false (4/5, 1) frozenTrials).1
      ≤ (run (1/20) (1/80) (3/10) (1/4) (1/500) (2/5) true (4/5, 1) frozenTrials).1
    ∧ (run (1/20) (1/80) (3/10) (1/4) (1/500) (2/5) true (4/5, 1) frozenTrials).2
      ≤ (run (1/20) (1/80) (3/10) (1/4) (1/500) (2/5) false (4/5, 1) frozenTrials).2 :=
  T13_reference_monotone (1/20) (1/80) (3/10) (1/4) (1/500) (2/5)
    (by grind) (by grind) (by grind) (by grind) (by grind) (by grind) (by grind)
    frozenTrials
    (by
      intro x hx
      simp only [frozenTrials, List.mem_cons, List.mem_nil_iff, or_false] at hx
      rcases hx with h | h | h | h | h | h <;> (subst h; show (1/4 : Rat) ≤ 18/25; grind))
    (4/5, 1) (4/5, 1)
    (by show (0 : Rat) ≤ 4/5; grind) (by show (4/5 : Rat) ≤ 4/5; grind)
    (by show (4/5 : Rat) ≤ 1; grind) (by show (1 : Rat) ≤ 1; grind)

-- The kernel as defined here, evaluated on that history, for comparison with the Python
-- generator's formulas (exact rationals).
#eval run (1/20) (1/80) (3/10) (1/4) (1/500) (2/5) false (4/5, 1) frozenTrials
#eval run (1/20) (1/80) (3/10) (1/4) (1/500) (2/5) true (4/5, 1) frozenTrials

-- Necessity of phi_plus ≤ tau: with phi_plus = 1 a win far above a low aspiration erodes
-- legitimacy more than the same win above a higher aspiration, so the order reverses.
example :
    ¬ (legStep (1/20) (1/80) (3/10) 1 1 (1/2) 1 ≤ legStep (1/20) (1/80) (3/10) 1 1 0 1) := by
  simp only [legStep, diss, clip01]
  grind

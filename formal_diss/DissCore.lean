/- KGi-DISS-I002 / M4 — core-Lean formalization of the algebraic skeleton.
   Lean 4.28.0, NO external dependencies (mathlib cache host blocked this session).
   Statements over `Rat`; they hold over any linear ordered field — Rat covers the
   model's rational parameter space. Scope: ALGEBRA only — no stochastics, no
   behavioural interpretation certified. -/

/-- T1 (gradient/nesting identity): exact quadratic expansion of the evidence-conflict
    dissonance term (L/2)(R−Q)². The linear coefficient in the revision h is
    −L·(R−Q): the belief-margin gradient step recovers the WP2 gated update
    δ_eff = L·(R−Q). -/
theorem T1_gradient_identity (L R Q h : Rat) :
    (L/2) * (R - (Q + h))^2 - (L/2) * (R - Q)^2
      = -(L * (R - Q)) * h + (L/2) * h^2 := by
  grind

/-- T2 (drift sign, cleared form): the expected legitimacy drift
    −ν_F(1−μ)(φ⁻A−τ) + ν_R μ is positive iff
    ν_F(1−μ)φ⁻·A < ν_F(1−μ)τ + ν_R μ. Pure rearrangement; no hypotheses needed. -/
theorem T2_drift_sign_cleared (nuF nuR mu A tau phim : Rat) :
    (0 < -(nuF * (1 - mu) * (phim * A - tau)) + nuR * mu)
      ↔ nuF * (1 - mu) * phim * A < nuF * (1 - mu) * tau + nuR * mu := by
  grind

/-- T2b (divided form): with ν_F > 0, μ ∈ (0,1), φ⁻ > 0, the cleared condition is
    equivalent to A < A* := (τ + (ν_R/ν_F)·(μ/(1−μ)))/φ⁻. -/
theorem T2b_Astar_divided (nuF nuR mu A tau phim : Rat)
    (hF : 0 < nuF) (hmu0 : 0 < mu) (hmu1 : mu < 1) (hp : 0 < phim) :
    (nuF * (1 - mu) * phim * A < nuF * (1 - mu) * tau + nuR * mu)
      ↔ A < (tau + (nuR / nuF) * (mu / (1 - mu))) / phim := by
  have h1mu : (0 : Rat) < 1 - mu := by grind
  have hX : 0 < nuF * (1 - mu) * phim := Rat.mul_pos (Rat.mul_pos hF h1mu) hp
  have hid : nuF * (1 - mu) * phim * ((tau + (nuR / nuF) * (mu / (1 - mu))) / phim)
      = nuF * (1 - mu) * tau + nuR * mu := by grind
  rw [← hid]
  exact Rat.mul_lt_mul_left hX

/-- Self-contained power (avoids instance dependence). -/
def cpow (c : Rat) : Nat → Rat
  | 0 => 1
  | n + 1 => c * cpow c n

/-- T3 (relative-mode / benchmark gap decay): if D(t+1) = (1 − 2ψΓ)·D(t) then
    D(t) = (1 − 2ψΓ)^t · D(0) — geometric decay at rate 2ψΓ (WP2 gap lemma re-proved). -/
theorem T3_gap_decay (psi Gam : Rat) (D : Nat → Rat)
    (hrec : ∀ t, D (t + 1) = (1 - 2 * psi * Gam) * D t) :
    ∀ t, D t = cpow (1 - 2 * psi * Gam) t * D 0 := by
  intro t
  induction t with
  | zero => simp [cpow]
  | succ n ih =>
      rw [hrec n, ih]
      show (1 - 2 * psi * Gam) * (cpow (1 - 2 * psi * Gam) n * D 0)
            = cpow (1 - 2 * psi * Gam) (n + 1) * D 0
      rw [cpow]
      grind

#print axioms T1_gradient_identity
#print axioms T2_drift_sign_cleared
#print axioms T2b_Astar_divided
#print axioms T3_gap_decay

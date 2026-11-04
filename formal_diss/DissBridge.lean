/- KGi-DISS-I003 / M2 — expectation bridge and gated contraction.
   Lean 4.28.0 core (no dependencies), statements over Rat as in I002's DissCore.lean.
   T4 closes the gap flagged in I002: Prop 2's E[ΔL] drift at Bernoulli outcomes,
   in the generic binding region, IS the T2 cleared condition — now formal.
   T5 is the algebraic core of WP2-P2 (gated contraction / slowdown). -/

/-- T4 (finite-outcome expectation bridge). In the generic region (failure discomfort
    binds: φ⁻A − τ > 0; success acceptable: φ⁺(1−A) ≤ τ), the branch-weighted expected
    legitimacy step with failure prob (1−μ) and success prob μ,
    E[ΔL] = (1−μ)·(−ν_F(φ⁻A−τ)) + μ·ν_R,
    is positive iff the T2 cleared condition ν_F(1−μ)φ⁻A < ν_F(1−μ)τ + ν_R μ holds.
    (Together with I002's T2b this yields E[ΔL] > 0 ↔ A < A*.) -/
theorem T4_expectation_bridge (nuF nuR mu A tau phim : Rat) :
    (0 < (1 - mu) * (-(nuF * (phim * A - tau))) + mu * nuR)
      ↔ nuF * (1 - mu) * phim * A < nuF * (1 - mu) * tau + nuR * mu := by
  grind

/-- T5a (one-step gated contraction): the gated value update pulls the chosen-arm value
    toward the arm mean with modulus (1 − αL):  Q' = Q + αL(m − Q) ⟹ Q' − m = (1−αL)(Q−m). -/
theorem T5a_gated_contraction (a L Q m : Rat) :
    (Q + a * L * (m - Q)) - m = (1 - a * L) * (Q - m) := by
  grind

/-- Self-contained power (same as I002). -/
def cpow (c : Rat) : Nat → Rat
  | 0 => 1
  | n + 1 => c * cpow c n

/-- T5b (n-step geometric form): if every step is the gated update at fixed gain αL,
    the deviation from the mean decays geometrically with ratio (1 − αL):
    Q_t − m = (1−αL)^t (Q_0 − m). This is the algebraic core of WP2-P2 (escape time
    ~ 1/(αL); divergence as L → 0). -/
theorem T5b_geometric (a L m : Rat) (Q : Nat → Rat)
    (hrec : ∀ t, Q (t + 1) = Q t + a * L * (m - Q t)) :
    ∀ t, Q t - m = cpow (1 - a * L) t * (Q 0 - m) := by
  intro t
  induction t with
  | zero => simp [cpow]
  | succ n ih =>
      rw [hrec n]
      show (Q n + a * L * (m - Q n)) - m = cpow (1 - a * L) (n + 1) * (Q 0 - m)
      rw [T5a_gated_contraction, ih, cpow]
      grind

#print axioms T4_expectation_bridge
#print axioms T5a_gated_contraction
#print axioms T5b_geometric

import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Exact finite identities and inequalities for the repaired social learner.
    No stochastic limit, calibration claim, or full-system bifurcation is stated. -/

namespace CSLRepair

open scoped BigOperators

variable {C : Type*} [Fintype C]

def act (B : C → C → ℝ) (z : C → ℝ) (c : C) : ℝ :=
  ∑ d, B c d * z d

def RowStochastic (B : C → C → ℝ) : Prop :=
  (∀ c d, 0 ≤ B c d) ∧ (∀ c, ∑ d, B c d = 1)

def macroMix (B : C → C → ℝ) (ω : C → ℝ) (θ : ℝ) : C → C → ℝ :=
  fun c d => (1 - θ) * B c d + θ * ω d

/-- F1: Uniform within-block weights give an exact realised finite quotient. -/
theorem block_aggregate (n : C → ℕ) (B : C → C → ℝ)
    (f : (Σ c, Fin (n c)) → ℝ) (c : C) :
    (∑ j : (Σ d, Fin (n d)), B c j.1 / (n j.1 : ℝ) * f j) =
      ∑ d, B c d * ((∑ j : Fin (n d), f ⟨d, j⟩) / (n d : ℝ)) := by
  rw [Fintype.sum_sigma]
  apply Finset.sum_congr rfl
  intro d hd
  change (∑ j : Fin (n d), (B c d / (n d : ℝ)) * f ⟨d, j⟩) =
    B c d * ((∑ j : Fin (n d), f ⟨d, j⟩) / (n d : ℝ))
  rw [← Finset.mul_sum]
  ring

/-- F2: Self-exclusion subtracts precisely one contribution, without renormalising. -/
theorem self_exclusion {I : Type*} [Fintype I] [DecidableEq I]
    (w f : I → ℝ) (i : I) :
    (∑ j ∈ Finset.univ.erase i, w j * f j) =
      (∑ j, w j * f j) - w i * f i := by
  have hs := Finset.sum_erase_add (s := Finset.univ)
    (f := fun j => w j * f j) (by simp : i ∈ Finset.univ)
  linarith

/-- F3: This proves boundedness of the actual convex update from its coefficients. -/
theorem convex_update_bounds (q r R w α g : ℝ)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hR0 : 0 ≤ R) (hR1 : R ≤ 1)
    (hα : 0 ≤ α) (hg : 0 ≤ g) (hr0 : 0 ≤ r) (hrw : r ≤ w)
    (hbudget : α + g * w ≤ 1) :
    0 ≤ (1 - α - g * w) * q + α * R + g * r ∧
      (1 - α - g * w) * q + α * R + g * r ≤ 1 := by
  have hc : 0 ≤ 1 - α - g * w := by linarith
  have hqr := mul_le_mul_of_nonneg_left hq1 hc
  have hRR := mul_le_mul_of_nonneg_left hR1 hα
  have hrr := mul_le_mul_of_nonneg_left hrw hg
  constructor
  · positivity
  · nlinarith

/-- F4: The regularised mean-score gain cannot exceed its amplitude b. -/
theorem peer_gain_bounds (b c w ε : ℝ) (hb : 0 ≤ b) (hc : 0 ≤ c)
    (hcw : c ≤ w) (hε : 0 < ε) :
    0 ≤ b * c / (w + ε) ∧ b * c / (w + ε) ≤ b := by
  have hw : 0 ≤ w := le_trans hc hcw
  have hd : 0 < w + ε := by positivity
  constructor
  · positivity
  · apply (div_le_iff₀ hd).2
    have hbc := mul_le_mul_of_nonneg_left hcw hb
    nlinarith [mul_nonneg hb (le_of_lt hε)]

/-- F5: The sufficient gain budget used in the implementation. -/
theorem gain_budget (α αmax b g w : ℝ)
    (hα : α ≤ αmax) (hg0 : 0 ≤ g) (hgb : g ≤ b)
    (_hw0 : 0 ≤ w) (hw1 : w ≤ 1) (hbudget : αmax + b ≤ 1) :
    α + g * w ≤ 1 := by
  have hgw := mul_le_mul_of_nonneg_left hw1 hg0
  nlinarith

def alphaPlus (d amin amax : ℝ) : ℝ := amax - (amax - amin) * d
def alphaMinus (d amin amax : ℝ) : ℝ := amin + (amax - amin) * d
def privateDrift (q μ d amin amax : ℝ) : ℝ :=
  μ * alphaPlus d amin amax * (1 - q) -
    (1 - μ) * alphaMinus d amin amax * q

/-- F6: Exact rational counterexample to using the sign of the mean error. -/
theorem realised_sign_counterexample :
    privateDrift (1/2) (11/20) (9/10) (1/20) (2/5) = -(47/800) ∧
    alphaPlus (9/10) (1/20) (2/5) * ((11/20) - (1/2)) = (17/4000) := by
  norm_num [privateDrift, alphaPlus, alphaMinus]

/-- F7: A row-stochastic field preserves every common signed mass. -/
theorem common_field (B : C → C → ℝ) (hB : RowStochastic B) (r : ℝ) (c : C) :
    act B (fun _ => r) c = r := by
  simp only [act, ← Finset.sum_mul, hB.2 c, one_mul]

/-- F8: Shared homogeneous local gain has a network-independent common multiplier. -/
theorem common_linear_multiplier (B : C → C → ℝ) (hB : RowStochastic B)
    (lam k r : ℝ) (c : C) :
    lam * k * act B (fun _ => r) c = lam * k * r := by
  rw [common_field B hB r c]

/-- F9: A fixed-total pairwise/macro mixture remains row stochastic. -/
theorem macro_stochastic (B : C → C → ℝ) (hB : RowStochastic B)
    (ω : C → ℝ) (hω0 : ∀ d, 0 ≤ ω d) (hω1 : ∑ d, ω d = 1)
    (θ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) :
    RowStochastic (macroMix B ω θ) := by
  constructor
  · intro c d
    exact add_nonneg (mul_nonneg (sub_nonneg.mpr hθ1) (hB.1 c d))
      (mul_nonneg hθ0 (hω0 d))
  · intro c
    simp only [macroMix, Finset.sum_add_distrib, ← Finset.mul_sum, hB.2 c, hω1]
    ring

/-- F10: Exact separation of pairwise and broadcast fields. -/
theorem macro_action (B : C → C → ℝ) (ω z : C → ℝ) (θ : ℝ) (c : C) :
    act (macroMix B ω θ) z c =
      (1 - θ) * act B z c + θ * (∑ d, ω d * z d) := by
  simp only [act, macroMix, add_mul, Finset.sum_add_distrib,
    mul_assoc, ← Finset.mul_sum]

/-- F11: A common broadcast cancels exactly from every field difference. -/
theorem macro_pairwise_difference (B : C → C → ℝ) (ω z : C → ℝ)
    (θ : ℝ) (c d : C) :
    act (macroMix B ω θ) z c - act (macroMix B ω θ) z d =
      (1 - θ) * (act B z c - act B z d) := by
  rw [macro_action, macro_action]
  ring

/-- F13: The absolute row-distance underlying Dobrushin scales exactly. -/
theorem macro_row_distance (B : C → C → ℝ) (ω : C → ℝ)
    (θ : ℝ) (hθ : θ ≤ 1) (c d : C) :
    (∑ j, |macroMix B ω θ c j - macroMix B ω θ d j|) =
      (1 - θ) * (∑ j, |B c j - B d j|) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  have he : macroMix B ω θ c j - macroMix B ω θ d j =
      (1 - θ) * (B c j - B d j) := by
    simp only [macroMix]
    ring
  rw [he, abs_mul, abs_of_nonneg (sub_nonneg.mpr hθ)]

/-- F14: A row-stochastic field contracts the componentwise supremum bound. -/
theorem field_sup_bound (B : C → C → ℝ) (hB : RowStochastic B)
    (z : C → ℝ) (r : ℝ) (hz : ∀ d, |z d| ≤ r) (c : C) :
    |act B z c| ≤ r := by
  calc
    |act B z c| ≤ ∑ d, |B c d * z d| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ d, B c d * |z d| := by
      apply Finset.sum_congr rfl
      intro d hd
      rw [abs_mul, abs_of_nonneg (hB.1 c d)]
    _ ≤ ∑ d, B c d * r := by
      apply Finset.sum_le_sum
      intro d hd
      exact mul_le_mul_of_nonneg_left (hz d) (hB.1 c d)
    _ = r := common_field B hB r c

/-- F12: Adding macro intensity, rather than reallocating it, increases total gain. -/
theorem additive_common_field (B : C → C → ℝ) (hB : RowStochastic B)
    (ω : C → ℝ) (hω1 : ∑ d, ω d = 1) (lpair lmacro r : ℝ) (c : C) :
    act (fun c d => lpair * B c d + lmacro * ω d) (fun _ => r) c =
      (lpair + lmacro) * r := by
  simp only [act, add_mul, Finset.sum_add_distrib,
    mul_assoc, ← Finset.mul_sum, ← Finset.sum_mul, hB.2 c, hω1]
  ring

end CSLRepair

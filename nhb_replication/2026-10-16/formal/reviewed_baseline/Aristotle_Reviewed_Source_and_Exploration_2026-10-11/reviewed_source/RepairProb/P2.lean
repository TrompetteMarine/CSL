import RepairCore
import Mathlib.Probability.Notation
import Mathlib.MeasureTheory.Function.ConditionalExpectation.Real
import Mathlib.MeasureTheory.Function.ConditionalExpectation.PullOut

/-! P2: realised-sign Bernoulli expectation and the private-only zero-drift target.

The gain is chosen from the sign of the *realised* prediction error `R - q`,
never from the sign of its expectation. -/

namespace CSLRepair

open MeasureTheory

/-- The realised private increment of the selected arm: gain `alphaPlus` for a
nonnegative realised error, gain `alphaMinus` for a negative realised error. -/
noncomputable def realisedIncrement (q d amin amax r : ℝ) : ℝ :=
  (if 0 ≤ r - q then alphaPlus d amin amax else alphaMinus d amin amax) * (r - q)

/-- On a reward in `{0,1}` and a score in `[0,1]`, the realised increment is affine in `R`. -/
lemma realisedIncrement_eq_affine (q d amin amax r : ℝ) (hr : r = 0 ∨ r = 1)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    realisedIncrement q d amin amax r =
      r * (alphaPlus d amin amax * (1 - q)) - (1 - r) * (alphaMinus d amin amax * q) := by
  rcases hr with rfl | rfl
  · unfold realisedIncrement
    rcases eq_or_lt_of_le hq0 with h | h
    · subst h; simp
    · rw [if_neg (by linarith)]; ring
  · unfold realisedIncrement
    rw [if_pos (by linarith)]; ring

/-- P2 (unconditional form, for a fixed pre-trial state): if, conditional on
selecting the arm, the reward is Bernoulli with success probability `μ`, the
expected realised-sign increment is exactly `privateDrift q μ d amin amax`. -/
theorem realised_sign_expectation {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (R : Ω → ℝ) (hRm : Measurable R)
    (hR01 : ∀ ω, R ω = 0 ∨ R ω = 1) (μ : ℝ) (hμ : P.real {ω | R ω = 1} = μ)
    (q d amin amax : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    ∫ ω, realisedIncrement q d amin amax (R ω) ∂P = privateDrift q μ d amin amax := by
  have hRind : R = {ω | R ω = 1}.indicator (fun _ => (1 : ℝ)) := by
    funext ω
    rcases hR01 ω with h | h <;> simp [Set.indicator, h]
  have hmeas : MeasurableSet {ω | R ω = 1} := hRm (measurableSet_singleton 1)
  have hRint : ∫ ω, R ω ∂P = μ := by
    rw [hRind, integral_indicator hmeas]; simpa using hμ
  have hRi : Integrable R P := by
    rw [hRind]; exact (integrable_const _).indicator hmeas
  simp_rw [realisedIncrement_eq_affine q d amin amax _ (hR01 _) hq0 hq1]
  have h1R : Integrable (fun ω => 1 - R ω) P := (integrable_const _).sub hRi
  rw [integral_sub (hRi.mul_const _) (h1R.mul_const _),
    integral_mul_const, integral_mul_const, integral_sub (integrable_const _) hRi,
    hRint]
  simp [privateDrift]
  ring

/-- P2 (conditional form): let `m` be the information available before the
reward is drawn (pre-trial state plus the selected action).  If the score `q`
and decisiveness `d ∈ [0,1]` are `m`-measurable, `q ∈ [0,1]`, and the reward
`R ∈ {0,1}` has conditional success probability `μ` given `m`, then the
conditional expectation of the realised-sign increment is `privateDrift q μ d`. -/
theorem realised_sign_condexp {Ω : Type*} {m mΩ : MeasurableSpace Ω} (hm : m ≤ mΩ)
    (P : Measure Ω) [IsProbabilityMeasure P] (R : Ω → ℝ) (hRm : Measurable R)
    (hR01 : ∀ ω, R ω = 0 ∨ R ω = 1) (μ : ℝ)
    (hμ : P[{ω | R ω = 1}.indicator (fun _ => (1 : ℝ)) | m] =ᵐ[P] fun _ => μ)
    (q d : Ω → ℝ) (hqm : Measurable[m] q) (hdm : Measurable[m] d)
    (hq0 : ∀ ω, 0 ≤ q ω) (hq1 : ∀ ω, q ω ≤ 1) (hd0 : ∀ ω, 0 ≤ d ω) (hd1 : ∀ ω, d ω ≤ 1)
    (amin amax : ℝ) :
    P[fun ω => realisedIncrement (q ω) (d ω) amin amax (R ω) | m] =ᵐ[P]
      fun ω => privateDrift (q ω) μ (d ω) amin amax := by
  set X : Ω → ℝ := fun ω => alphaPlus (d ω) amin amax * (1 - q ω) with hX
  set Y : Ω → ℝ := fun ω => alphaMinus (d ω) amin amax * q ω with hY
  have hRind : R = {ω | R ω = 1}.indicator (fun _ => (1 : ℝ)) := by
    funext ω
    rcases hR01 ω with h | h <;> simp [Set.indicator, h]
  have hmeas : MeasurableSet {ω | R ω = 1} := hRm (measurableSet_singleton 1)
  have hRi : Integrable R P := by
    rw [hRind]; exact (integrable_const _).indicator hmeas
  have hRc : P[R | m] =ᵐ[P] fun _ => μ := by rw [hRind]; exact hμ
  have hXm : StronglyMeasurable[m] X := by
    apply Measurable.stronglyMeasurable
    simp only [hX, alphaPlus]; fun_prop
  have hYm : StronglyMeasurable[m] Y := by
    apply Measurable.stronglyMeasurable
    simp only [hY, alphaMinus]; fun_prop
  have hXb : ∀ ω, ‖X ω‖ ≤ (|amax| + |amax - amin|) * 1 := by
    intro ω
    simp only [hX, alphaPlus, Real.norm_eq_abs, abs_mul]
    have h1 : |1 - q ω| ≤ 1 := by rw [abs_le]; constructor <;> linarith [hq0 ω, hq1 ω]
    have h2 : |amax - (amax - amin) * d ω| ≤ |amax| + |amax - amin| := by
      calc _ ≤ |amax| + |(amax - amin) * d ω| := abs_sub _ _
        _ ≤ _ := by
          rw [abs_mul]; gcongr
          nlinarith [abs_nonneg (amax - amin), abs_le.mpr (⟨by linarith [hd0 ω], hd1 ω⟩ :
            -1 ≤ d ω ∧ d ω ≤ 1)]
    exact mul_le_mul h2 h1 (abs_nonneg _) (by positivity)
  have hYb : ∀ ω, ‖Y ω‖ ≤ (|amin| + |amax - amin|) * 1 := by
    intro ω
    simp only [hY, alphaMinus, Real.norm_eq_abs, abs_mul]
    have h1 : |q ω| ≤ 1 := by rw [abs_le]; constructor <;> linarith [hq0 ω, hq1 ω]
    have h2 : |amin + (amax - amin) * d ω| ≤ |amin| + |amax - amin| := by
      calc _ ≤ |amin| + |(amax - amin) * d ω| := abs_add_le _ _
        _ ≤ _ := by
          rw [abs_mul]; gcongr
          nlinarith [abs_nonneg (amax - amin), abs_le.mpr (⟨by linarith [hd0 ω], hd1 ω⟩ :
            -1 ≤ d ω ∧ d ω ≤ 1)]
    exact mul_le_mul h2 h1 (abs_nonneg _) (by positivity)
  have h1R : Integrable (fun ω => 1 - R ω) P := (integrable_const _).sub hRi
  have hRX : Integrable (X * R) P := by
    refine Integrable.bdd_mul hRi (hXm.mono hm).aestronglyMeasurable
      (Filter.Eventually.of_forall hXb)
  have hRY : Integrable (Y * fun ω => 1 - R ω) P := by
    refine Integrable.bdd_mul h1R (hYm.mono hm).aestronglyMeasurable
      (Filter.Eventually.of_forall hYb)
  have heq : (fun ω => realisedIncrement (q ω) (d ω) amin amax (R ω)) =
      X * R - Y * (fun ω => 1 - R ω) := by
    funext ω
    simp only [Pi.sub_apply, Pi.mul_apply, hX, hY]
    rw [realisedIncrement_eq_affine _ _ _ _ _ (hR01 ω) (hq0 ω) (hq1 ω)]
    ring
  rw [heq]
  have h1 := condExp_mul_of_stronglyMeasurable_left hXm hRX hRi
  have h2 := condExp_mul_of_stronglyMeasurable_left hYm hRY h1R
  have h3 : P[fun ω => 1 - R ω | m] =ᵐ[P] fun _ => 1 - μ := by
    have := condExp_sub (integrable_const (1 : ℝ)) hRi m (μ := P)
    filter_upwards [this, hRc] with ω h h'
    rw [show (fun ω => 1 - R ω) = (fun _ => (1:ℝ)) - R from rfl, h, Pi.sub_apply, h',
      condExp_const hm]
  filter_upwards [condExp_sub hRX hRY m, h1, h2, hRc, h3] with ω h h1 h2 hR h3
  rw [h, Pi.sub_apply, h1, h2]
  simp only [Pi.mul_apply, hR, h3, hX, hY, privateDrift]
  ring

/-- The private-only zero-drift target at fixed decisiveness. -/
noncomputable def driftTarget (μ d amin amax : ℝ) : ℝ :=
  μ * alphaPlus d amin amax /
    (μ * alphaPlus d amin amax + (1 - μ) * alphaMinus d amin amax)

/-- P2 (target): when the denominator is positive the drift is the denominator
times the signed distance to the target; hence it vanishes exactly at the
target, is positive below it and negative above it. -/
theorem privateDrift_eq_target (q μ d amin amax : ℝ)
    (hden : 0 < μ * alphaPlus d amin amax + (1 - μ) * alphaMinus d amin amax) :
    privateDrift q μ d amin amax =
      (μ * alphaPlus d amin amax + (1 - μ) * alphaMinus d amin amax) *
        (driftTarget μ d amin amax - q) := by
  unfold privateDrift driftTarget
  field_simp
  ring

/-- P2 (target): the target is the unique zero of the private drift. -/
theorem privateDrift_eq_zero_iff (q μ d amin amax : ℝ)
    (hden : 0 < μ * alphaPlus d amin amax + (1 - μ) * alphaMinus d amin amax) :
    privateDrift q μ d amin amax = 0 ↔ q = driftTarget μ d amin amax := by
  rw [privateDrift_eq_target q μ d amin amax hden, mul_eq_zero, sub_eq_zero]
  constructor
  · rintro (h | h)
    · exact absurd h hden.ne'
    · exact h.symm
  · intro h; exact Or.inr h.symm

end CSLRepair

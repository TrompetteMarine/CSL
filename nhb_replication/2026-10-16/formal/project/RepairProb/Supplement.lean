import RepairProb.P3
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Probability.Moments.Variance

/-! Supplementary checks for the probability modules.

* The four predeclared score conventions satisfy the score hypotheses used in
  P1 (measurable, mapping `[0,1]` into `[0,1]`) and the continuity at `1/2`
  used by `readout_derivative`.  So those hypotheses are not vacuous.
* R4 remark: a zero mean private gap does not give the homogeneous slope.
  For the constant convention and any `h ≠ 0`, the private gaps `±1` have mean
  zero, yet their mean readout slope is strictly below the slope at `0`. -/

namespace CSLRepair

/-- Constant convention `s(p) = 1/2`. -/
noncomputable def scoreConstant (_p : ℝ) : ℝ := 1 / 2
/-- Preference convention `s(p) = (1 + |2p-1|)/2` (has a cusp at `1/2`). -/
noncomputable def scorePreference (p : ℝ) : ℝ := (1 + |2 * p - 1|) / 2
/-- Quadratic convention `s(p) = (1 + |2p-1|^2)/2`. -/
noncomputable def scoreQuadratic (p : ℝ) : ℝ := (1 + |2 * p - 1| ^ 2) / 2
/-- Zero-floor convention `s(p) = |2p-1|`. -/
noncomputable def scoreZeroFloor (p : ℝ) : ℝ := |2 * p - 1|

lemma abs_two_mul_sub_one_le {p : ℝ} (h0 : 0 ≤ p) (h1 : p ≤ 1) : |2 * p - 1| ≤ 1 := by
  rw [abs_le]; constructor <;> linarith

/-- All four conventions are continuous (hence measurable and continuous at `1/2`)
and map `[0,1]` into `[0,1]`. -/
theorem score_conventions_admissible :
    (Continuous scoreConstant ∧ Continuous scorePreference ∧ Continuous scoreQuadratic ∧
      Continuous scoreZeroFloor) ∧
    (∀ p, 0 ≤ p → p ≤ 1 →
      (0 ≤ scoreConstant p ∧ scoreConstant p ≤ 1) ∧
      (0 ≤ scorePreference p ∧ scorePreference p ≤ 1) ∧
      (0 ≤ scoreQuadratic p ∧ scoreQuadratic p ≤ 1) ∧
      (0 ≤ scoreZeroFloor p ∧ scoreZeroFloor p ≤ 1)) := by
  refine ⟨⟨continuous_const, ?_, ?_, ?_⟩, fun p h0 h1 => ?_⟩
  · unfold scorePreference; fun_prop
  · unfold scoreQuadratic; fun_prop
  · unfold scoreZeroFloor; fun_prop
  have hu := abs_two_mul_sub_one_le h0 h1
  have hu0 := abs_nonneg (2 * p - 1)
  simp only [scoreConstant, scorePreference, scoreQuadratic, scoreZeroFloor]
  refine ⟨⟨by norm_num, by norm_num⟩, ⟨by linarith, by linarith⟩, ⟨by positivity, ?_⟩,
    ⟨hu0, hu⟩⟩
  nlinarith

lemma hasDerivAt_sigmoid (x : ℝ) : HasDerivAt sigmoid (sigmoid x * (1 - sigmoid x)) x := by
  have he : HasDerivAt (fun y => Real.exp (-y)) (-Real.exp (-x)) x := by
    simpa using (Real.hasDerivAt_exp (-x)).comp x (hasDerivAt_neg x)
  have hd : HasDerivAt (fun y => 1 + Real.exp (-y)) (-Real.exp (-x)) x := by
    simpa using he.const_add 1
  have hne : (1 + Real.exp (-x)) ≠ 0 := by positivity
  have := (hasDerivAt_const x (1 : ℝ)).div hd hne
  unfold sigmoid
  convert this using 1
  field_simp
  ring

/-- Slope of the constant-convention readout `G(x) = σ(hx) - 1/2`. -/
lemma hasDerivAt_constant_readout (h x : ℝ) :
    HasDerivAt (weightedReadout scoreConstant h)
      (h * (sigmoid (h * x) * (1 - sigmoid (h * x)))) x := by
  have hl : HasDerivAt (fun y : ℝ => h * y) h x := by
    simpa using (hasDerivAt_id x).const_mul h
  have hs := (hasDerivAt_sigmoid (h * x)).comp x hl
  have : weightedReadout scoreConstant h = fun y => sigmoid (h * y) - 1 / 2 := by
    funext y; simp [weightedReadout, scoreConstant]; ring
  rw [this]
  convert hs.sub_const (1 / 2) using 1
  ring

/-- R4 remark (counterexample to "zero mean private gap suffices"): with the
constant convention and any `h ≠ 0`, the two private gaps `1` and `-1` have
mean zero, but the mean of their readout slopes is strictly smaller than the
homogeneous slope `G'(0) = h/4` (in absolute value), so the community field
Jacobian differs from `λ G'(0) K`. -/
theorem zero_mean_gap_not_homogeneous (h : ℝ) (hh : h ≠ 0) :
    ((1 : ℝ) + (-1)) / 2 = 0 ∧
    |(h * (sigmoid (h * 1) * (1 - sigmoid (h * 1))) +
        h * (sigmoid (h * (-1)) * (1 - sigmoid (h * (-1))))) / 2| <
      |h * (sigmoid (h * 0) * (1 - sigmoid (h * 0)))| := by
  refine ⟨by norm_num, ?_⟩
  have hsym : sigmoid (h * (-1)) = 1 - sigmoid (h * 1) := by
    rw [one_sub_sigmoid]; ring_nf
  have h0 : sigmoid (h * 0) = 1 / 2 := by norm_num [sigmoid]
  rw [hsym, h0]
  set s := sigmoid (h * 1)
  have hs : s ≠ 1 / 2 := by
    intro hs
    have : Real.exp (-(h * 1)) = 1 := by
      simp only [s, sigmoid] at hs
      have hp : 0 < 1 + Real.exp (-(h * 1)) := by positivity
      field_simp at hs
      simp only [mul_one] at hs ⊢
      linarith
    rw [Real.exp_eq_one_iff] at this
    exact hh (by linarith)
  have hlt : s * (1 - s) < 1 / 4 := by
    have : 0 < (s - 1 / 2) ^ 2 := by
      have : s - 1 / 2 ≠ 0 := sub_ne_zero.mpr hs
      positivity
    nlinarith
  have hpos : 0 < s * (1 - s) := mul_pos (sigmoid_pos _) (by linarith [sigmoid_lt_one (h * 1)])
  have e : (h * (s * (1 - s)) + h * ((1 - s) * (1 - (1 - s)))) / 2 = h * (s * (1 - s)) := by
    ring
  rw [e, abs_mul, abs_of_pos hpos, abs_mul]
  have hha : 0 < |h| := abs_pos.mpr hh
  norm_num
  nlinarith

/-- R3 / A1 lifted to the network: for the explicitly frozen, homogeneous,
private-gap-zero map `F(z)_c = G(λ (K z)_c)`, with `G = weightedReadout score h`
and `score` merely continuous at `1/2`, `F` is Fréchet differentiable at `0`
and its derivative `J` has `J v = λ k K v` with `k = score(1/2) h / 2`.
This is the frozen-map linearisation only, not a bifurcation statement for the
full stochastic learner. -/
theorem frozen_map_jacobian {C : Type*} [Fintype C] (K : C → C → ℝ) (lam h : ℝ)
    (score : ℝ → ℝ) (hs : ContinuousAt score (1 / 2)) :
    HasFDerivAt (fun z : C → ℝ => fun c => weightedReadout score h (lam * act K z c))
      (fieldJacobian (fun _ => 1) K lam (fun _ => score (1 / 2) * h / 2)) 0 ∧
    ∀ v c, fieldJacobian (fun _ => 1) K lam (fun _ => score (1 / 2) * h / 2) v c =
      lam * (score (1 / 2) * h / 2) * act K v c := by
  constructor
  · have hd := field_jacobian (fun _ : C => 1) K lam (weightedReadout score h) (fun _ => 0) 0
      (fun _ => score (1 / 2) * h / 2) (fun i => by
        simpa [choiceGap, act] using readout_derivative score h hs)
    convert hd using 1
    funext z c
    simp [choiceGap]
  · intro v c
    rw [fieldJacobian_apply]
    simp

open MeasureTheory ProbabilityTheory in
/-- R4 counterexample (individual variance does not collapse): an isolated,
always-selected fixed-gain learner `Q' = (1-α) Q + α R` with `0 < α < 1`,
reward `R ∈ {0,1}` with `P(R = 1) = μ ∈ (0,1)` independent of `Q`, and `Q`
with mean `μ` and variance `v* = α μ (1-μ) / (2-α)`, has `Q'` with the same
mean and the same variance `v* > 0`.  Thus `v*` is a stationary variance of
this recursion, it is strictly positive, and it does not involve any
population size. -/
theorem fixed_gain_stationary_variance {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (Q R : Ω → ℝ)
    (hQ : MemLp Q 2 P) (hRm : Measurable R) (hR01 : ∀ ω, R ω = 0 ∨ R ω = 1)
    (hind : IndepFun Q R P) (α μ : ℝ) (hα0 : 0 < α) (hα1 : α < 1)
    (hμ0 : 0 < μ) (hμ1 : μ < 1) (hRμ : P.real {ω | R ω = 1} = μ)
    (hQmean : ∫ ω, Q ω ∂P = μ) (hQvar : variance Q P = α * μ * (1 - μ) / (2 - α)) :
    0 < α * μ * (1 - μ) / (2 - α) ∧
    ∫ ω, ((1 - α) * Q ω + α * R ω) ∂P = μ ∧
    variance (fun ω => (1 - α) * Q ω + α * R ω) P = α * μ * (1 - μ) / (2 - α) := by
  have hRind : R = {ω | R ω = 1}.indicator (fun _ => (1 : ℝ)) := by
    funext ω
    rcases hR01 ω with h | h <;> simp [Set.indicator, h]
  have hmeas : MeasurableSet {ω | R ω = 1} := hRm (measurableSet_singleton 1)
  have hRint : ∫ ω, R ω ∂P = μ := by
    rw [hRind, integral_indicator hmeas]; simpa using hRμ
  have hRL : MemLp R 2 P := by
    rw [hRind]; exact (memLp_const 1).indicator hmeas
  have hRsq : (fun ω => R ω ^ 2) = R := by
    funext ω; rcases hR01 ω with h | h <;> simp [h]
  have hRvar : variance R P = μ * (1 - μ) := by
    rw [variance_eq_sub hRL]
    simp only [Pi.pow_apply]
    rw [hRsq, hRint]; ring
  refine ⟨div_pos (mul_pos (mul_pos hα0 hμ0) (by linarith)) (by linarith), ?_, ?_⟩
  · rw [integral_add ((hQ.integrable one_le_two).const_mul _)
      ((hRL.integrable one_le_two).const_mul _), integral_const_mul, integral_const_mul,
      hQmean, hRint]
    ring
  · have hind' : IndepFun (fun ω => (1 - α) * Q ω) (fun ω => α * R ω) P :=
      hind.comp (φ := fun x => (1 - α) * x) (ψ := fun x => α * x)
        (by fun_prop) (by fun_prop)
    have hadd := hind'.variance_add (hQ.const_mul _) (hRL.const_mul _)
    have e : (fun ω => (1 - α) * Q ω + α * R ω) =
        (fun ω => (1 - α) * Q ω) + (fun ω => α * R ω) := rfl
    rw [e, hadd, variance_const_mul, variance_const_mul, hQvar, hRvar]
    have : (2 - α) ≠ 0 := by linarith
    field_simp
    ring

end CSLRepair
